# Mix API Contract

This document defines the contract for `mx8.mix`, which blends multiple MX8 loaders deterministically under one shared memory envelope.


## What problem it solves

Training on multiple datasets with different sizes and importance weights is a common requirement. The naive approach — interleaving loaders manually — either loses the memory safety guarantees of each loader or requires careful hand-tuning to avoid one source starving another or the combined pipeline exceeding memory limits.

`mx8.mix` solves this by running all sources under one shared inflight cap and one deterministic scheduler. The memory guarantees hold for the combined pipeline, not per-source. Source selection is deterministic and reproducible.


## Usage

```python
import mx8

mixed = mx8.mix(
    loaders=[loader_a, loader_b],
    weights=[0.7, 0.3],
    seed=1234,
    epoch=0,
    source_exhausted="error",
    profile="balanced",
    autotune=True,
    constraints=mx8.Constraints(max_inflight_bytes=256 * 1024 * 1024),
    runtime=mx8.RuntimeConfig(prefetch_batches=1, max_queue_batches=8),
)
```

`loaders` is a list of existing `mx8.load` loaders. `weights` is a list of positive floats of the same length, normalized internally. `seed` and `epoch` are the deterministic scheduling inputs. `source_exhausted` controls what happens when a source runs out: `error` fails fast (the default, to avoid silent source drop), `allow` lets the mixer drain the remaining sources. `starvation_window` controls the starvation accounting window for the scheduler.

`profile` and `autotune` apply mix-level startup rails. `constraints` overrides the shared cap — specifically `max_inflight_bytes` and `max_ram_bytes`. `runtime` sets startup overrides for `prefetch_batches` and `max_queue_batches`. Note that `runtime.want` is not supported for `mx8.mix` — lease parallelism belongs to the distributed loader flow.


## Determinism contract

For a fixed set of source manifests, weights, seed, epoch, world size, and frozen membership, the mixed stream order is deterministic and replayable. Reproduce any run by pinning the same inputs.

On `resume_from`, mix restores scheduler/counter state from the token. If source checkpoints differ from token snapshots, mix continues in best-effort mode and reports this via `mix_resume_source_checkpoint_mismatch_total` in `mixed.stats()`.


## Memory contract

All sources share one inflight cap. The effective shared cap is `min(source_caps, mix_overrides)`. There is no per-source unbounded buffer. Backpressure is global: if the consumer slows down, all sources are throttled through the shared bounded scheduler.


## Observability

`mixed.stats()["mix_sources"]` gives per-source diagnostics including manifest IDs, delivery counters, configured knobs, and source-level metrics. Use this to verify that observed source contribution ratios match your configured weights.


## Validation gates

Run `./scripts/mix_gate.sh` to verify determinism, ratio accuracy, memory bounds, and exhaustion behavior. For stricter ratio tolerance, run `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`. For multi-rank no-overlap verification, run `./scripts/mix_multirank_gate.sh`. See `docs/mix_gate_runbook.md` for the full gate runbook.
