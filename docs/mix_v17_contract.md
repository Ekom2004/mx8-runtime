# MX8 `mx8.mix` Contract (v1.7)

This document defines the current contract for `mx8.mix(...)`.

## Goal

Enable deterministic weighted mixing across multiple MX8 loaders while preserving one shared bounded-memory runtime contract.

## Proposed API

```python
import mx8

mixed = mx8.mix(
    loaders=[loader_a, loader_b],
    weights=[0.7, 0.3],
    seed=1234,
    epoch=0,
    on_source_exhausted="error",
)
```

- `loaders`: list of existing MX8 loaders (initial scope: byte-oriented `mx8.load(...)` loaders).
- `weights`: positive floats, same length as `loaders`, normalized internally.
- `seed`: deterministic source-selection seed.
- `on_source_exhausted`: `error|allow` (default: `error`).

## Determinism Contract (failure-free)

For fixed:
- source manifests (`manifest_hash` set for each loader),
- `weights`,
- `seed`,
- `epoch`,
- `world_size` + frozen membership,

the mixed stream order is deterministic and replayable.

## Memory + Backpressure Contract

- Mixed execution uses one shared cap envelope (`max_inflight_bytes`, queue caps).
- No per-source unbounded buffering.
- Backpressure is global: if sink slows, all sources are throttled through one bounded scheduler.

## Initial Scheduling Contract

- Deterministic weighted round-robin source selection.
- Delivery remains per-batch; no global reorder inside a delivered batch.
- Source-level exhaustion defaults to fail fast with explicit error (no silent source drop).

## Planned Acceptance Gates

- Determinism gate: 3 repeated runs must produce identical digest of `(source_id, sample_id)` sequence.
- Ratio gate: observed source contribution must be within tolerance of target weights (default ±2%; strict ±1%).
- Memory gate: inflight/process bounds remain within configured caps during mixed run.
- Exhaustion gate: `on_source_exhausted=error` fails fast; `allow` drains with explicit counters.

Gate command:

- `./scripts/mix_gate.sh`
- strict mode: `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`
- smoke toggle: `MX8_SMOKE_MIX=1 ./scripts/smoke.sh`
- smoke strict toggle: `MX8_SMOKE_MIX=1 MX8_SMOKE_MIX_STRICT=1 ./scripts/smoke.sh`
- runbook: `docs/mix_gate_runbook.md`
