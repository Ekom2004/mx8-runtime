# MX8 v1 Autotune API Contract (Planned)

Status: **active contract direction** for current loader API cleanup.

## Goals

- Make default usage frictionless for most users.
- Keep hard safety limits explicit and enforceable.
- Keep power-user controls available without polluting the default API.

## Public API shape

```python
import mx8

# Simple path (default)
loader = mx8.load(
    "s3://bucket/train@refresh",
    profile="balanced",
    autotune=True,
)
```

```python
# Constrained autotune path
loader = mx8.load(
    "s3://bucket/train@refresh",
    profile="throughput",
    autotune=True,
    constraints=mx8.Constraints(
        max_inflight_bytes=512 * 1024 * 1024,
        max_ram_bytes=24 * 1024 * 1024 * 1024,
    ),
)
```

```python
# Advanced manual path
loader = mx8.load(
    "s3://bucket/train@refresh",
    autotune=False,
    runtime=mx8.RuntimeConfig(
        prefetch_batches=8,
        max_queue_batches=32,
        want=4,
    ),
)
```

## Contract rules

1. `profile` sets defaults (including safety defaults).
2. `constraints` overrides profile safety defaults.
3. `autotune=True` adjusts only runtime knobs (`prefetch_batches`, `max_queue_batches`, `want`) within safety constraints.
4. `autotune` never increases hard caps.
5. `runtime` is explicit pinning; when `autotune=False`, runtime knobs are not modified by MX8.
6. If both `runtime` and `autotune=True` are provided, runtime values are treated as initial values and still clamped to constraints.

## Safety invariants

- Hard caps remain authoritative:
  - `max_inflight_bytes`
  - optional `max_ram_bytes`
- Autotune cannot violate cap invariants.
- Runtime adaptation uses hysteresis/cooldown to avoid oscillation.

## Startup flow (user experience)

On `mx8.load(..., profile=..., autotune=True)`:

1. Read machine limit:
   - `node_ram_limit = min(physical_ram, cgroup_limit_if_present)`.
2. Determine local concurrency:
   - `local_ranks` from runtime/distributed context (default `1`).
3. Compute initial per-rank budget from profile:
   - profile selects default reservation ratios and minimum headroom.
4. Warm up briefly (small fixed step budget) to estimate:
   - `base_rss_bytes` (model/framework/process baseline before deep buffering).
5. Set caps:
   - `max_ram_bytes` (unless explicitly overridden in `constraints`).
   - `max_inflight_bytes` (unless explicitly overridden in `constraints`).
6. Start adaptive loop for runtime knobs:
   - `prefetch_batches`, `max_queue_batches`, `want`.

Startup must emit one compact summary line with computed values.

## Cap derivation policy

Autotune computes caps from budget, then clamps:

- `node_budget_bytes = profile.node_fraction * node_ram_limit - profile.node_reserve_bytes`
- `per_rank_budget_bytes = floor(node_budget_bytes / local_ranks)`
- `derived_max_ram_bytes = profile.rss_fraction * per_rank_budget_bytes`
- `derived_max_inflight_bytes = min(profile.inflight_fraction * derived_max_ram_bytes, derived_max_ram_bytes - base_rss_bytes - profile.rss_guard_bytes)`

Then apply:

- explicit `constraints.max_ram_bytes` if provided
- explicit `constraints.max_inflight_bytes` if provided
- hard lower/upper bounds from profile safety rails

Rules:

- `max_inflight_bytes <= max_ram_bytes`
- `max_inflight_bytes >= profile.min_inflight_bytes`
- If derived values violate rails, clamp and emit warning event.

## Autotune signals

Controller inputs (periodic):

- data-wait time in training/consumer loop
- queue depth
- inflight bytes and RSS headroom
- step-time jitter

Controller outputs:

- `prefetch_batches`
- `max_queue_batches`
- `want`

## Runtime adaptation policy

Controller cadence: fixed interval (e.g., 2s) with cooldown.

Increase path (only if all are true):

- data-wait ratio above target threshold
- RSS/inflight headroom above profile minimum
- queue starvation indicators present

Decrease path (any true):

- RSS near `max_ram_bytes`
- inflight near `max_inflight_bytes`
- persistent queue saturation without throughput gain

Each change is bounded step-size (%-based) and subject to:

- cooldown window
- max one knob change per interval
- hysteresis band to avoid oscillation

## Stats / debug contract

`loader.stats()` (or debug endpoint) must include:

- `effective.max_ram_bytes`
- `effective.max_inflight_bytes`
- `effective.prefetch_batches`
- `effective.max_queue_batches`
- `effective.want`
- `observed.process_rss_bytes`
- `observed.inflight_bytes`
- `observed.data_wait_ratio`
- `observed.step_time_jitter`
- `autotune.last_decision`
- `autotune.decision_reason`
- `autotune.cooldown_remaining_ms`

Proof events (structured logs):

- `autotune_startup_caps_selected`
- `autotune_runtime_adjustment`
- `autotune_cap_clamped`
- `autotune_disabled_manual_runtime`

## Failure semantics

- If RSS exceeds `max_ram_bytes`, fail fast with explicit error (same safety behavior as v0 watchdog).
- If autotune cannot find valid settings under constraints, fail with actionable configuration error.
- Manual mode (`autotune=False`) never mutates runtime knobs.

## Persistence

Autotune may persist learned runtime settings keyed by:

- `manifest_hash`
- world size
- batch shape
- node class / hardware signature

## Observability

Expose effective runtime state in loader stats/debug APIs:

- current `prefetch_batches`, `max_queue_batches`, `want`
- current safety caps
- autotune decisions/reasons (compact event stream)

## Acceptance criteria

- Defaults-only runs complete without manual tuning on representative workloads.
- Data-wait ratio decreases versus static defaults.
- No cap breaches under autotune.
- Advanced users can disable autotune and get deterministic pinned behavior.
- Startup summary and effective config are visible without debug-only tooling.
