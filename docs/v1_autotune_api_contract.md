# Autotune API Contract

This document defines how MX8 autotune works, what it controls, and what it never changes.

Primary implementation: `crates/mx8-py/src/lib.rs` (startup cap derivation and runtime adaptation loop).


## The problem autotune solves

Choosing the right `max_inflight_bytes`, `prefetch_batches`, and `max_queue_batches` for a given machine, model, and dataset requires trial and error. Set them too low and you starve the GPU. Set them too high and you risk OOM or oscillation.

Autotune removes that guesswork for most users. You tell MX8 how much RAM to use and which profile matches your workload, and it derives safe starting values and adapts them at runtime based on observed pressure and throughput.


## API modes

The simple path lets MX8 do the work:

```python
loader = mx8.load(
    "s3://bucket/train@refresh",
    ram_gb=24,
    profile="balanced",
    tune=True,
)
```

The constrained path keeps autotune active but pins specific caps:

```python
loader = mx8.load(
    "s3://bucket/train@refresh",
    ram_gb=24,
    profile="throughput",
    tune=True,
    constraints=mx8.Constraints(
        max_inflight_bytes=512 * 1024 * 1024,
        max_ram_bytes=24 * 1024 * 1024 * 1024,
    ),
)
```

The manual path disables adaptation entirely:

```python
loader = mx8.load(
    "s3://bucket/train@refresh",
    ram_gb=24,
    tune=False,
    runtime=mx8.RuntimeConfig(
        prefetch_batches=8,
        max_queue_batches=32,
        want=4,
    ),
)
```


## Contract rules

`profile` sets the safety defaults including minimum headroom requirements. `constraints` overrides specific profile safety defaults. `tune=True` adjusts only runtime knobs — `prefetch_batches`, `max_queue_batches`, and `want` — within the safety constraints. Autotune never increases hard caps. When both `runtime` and `tune=True` are provided, the runtime values are treated as starting points and clamped to constraints.


## Startup flow

When `mx8.load` (or `mx8.run`) is called with autotune enabled, MX8 reads the node RAM limit by taking the minimum of physical RAM and any cgroup limit. It determines local rank count from the runtime context, selects default reservation ratios from the profile, warms up briefly to estimate the process baseline RSS, then derives `max_ram_bytes` and `max_inflight_bytes`. It emits a compact startup summary with the computed values.

Cap derivation uses this chain:

`node_budget_bytes = profile.node_fraction * node_ram_limit - profile.node_reserve_bytes`

`per_rank_budget_bytes = floor(node_budget_bytes / local_ranks)`

`derived_max_ram_bytes = profile.rss_fraction * per_rank_budget_bytes`

`derived_max_inflight_bytes = min(profile.inflight_fraction * derived_max_ram_bytes, derived_max_ram_bytes - base_rss_bytes - profile.rss_guard_bytes)`

Explicit `constraints` values override the derived values. Hard profile safety rails clamp any result that falls outside bounds.


## Runtime adaptation

The adaptation loop runs on a fixed cadence with cooldown between adjustments. It reads data-wait time, queue depth, inflight bytes, RSS headroom, and step-time jitter. It adjusts `prefetch_batches`, `max_queue_batches`, and `want`.

The jitter SLO target is `batch_payload_bytes_p95_over_p50 <= 1.25`. When jitter is above target, the loop tightens the byte band. When it is below target and the pipeline has headroom, it relaxes.

The increase path only fires when all of the following are true: the data-wait ratio is above the target threshold, RSS and inflight headroom are above the profile minimum, and queue starvation indicators are present.

The decrease path fires when any of the following are true: RSS is near `max_ram_bytes`, inflight is near `max_inflight_bytes`, or there is persistent queue saturation without a throughput gain.

Each adjustment is bounded by a step-size percentage, subject to a cooldown window, and limited to one knob change per interval to avoid oscillation.


## Mix and video autotune

The mix runtime autotune runs on a tick-based cadence controlled by `MX8_MIX_AUTOTUNE_PERIOD_TICKS` (default 32 ticks). It adjusts the shared mix cap within safe bounds.

The video runtime autotune runs every `MX8_VIDEO_AUTOTUNE_PERIOD_BATCHES` delivered batches (default 16). It adapts `max_inflight_bytes` for the video pipeline within the configured profile rails.


## Stats and proof logs

`loader.stats()` exposes the effective runtime state: `effective.max_ram_bytes`, `effective.max_inflight_bytes`, `effective.prefetch_batches`, `effective.max_queue_batches`, `effective.want`, `observed.process_rss_bytes`, `observed.inflight_bytes`, and autotune decision fields.

Proof events emitted to the `mx8_proof` log target: `autotune_startup_caps_selected`, `autotune_runtime_adjustment`, `autotune_cap_clamped`, `autotune_disabled_manual_runtime`, `load_runtime_autotune_initialized`, and `rss_cap_defaulted`.


## Failure semantics

If RSS exceeds `max_ram_bytes`, MX8 fails fast with an explicit error. If autotune cannot find valid settings within the constraints, it fails with an actionable configuration error. Manual mode (`tune=False`) never mutates runtime knobs after startup.
