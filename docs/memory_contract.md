# MX8 Memory Contract (v0 + v1 direction)

This page defines exactly what MX8 memory guarantees mean.

## v0 hard guarantee

MX8 enforces a hard cap on loader-path memory with:

- `max_inflight_bytes`

This cap applies to bytes admitted into the MX8 pipeline (fetch/decode/queued delivery), with backpressure when full.

## v0 bounded knobs

Use these together:

- `max_inflight_bytes` (pipeline byte budget)
- `max_queue_batches` (delivered queue depth)
- `prefetch_batches` (read-ahead depth)
- `want` (distributed concurrent leases per node)

These protect the data path from runaway buffering.

## what v0 does not guarantee

v0 does not hard-cap total process RSS.

Process RAM includes non-MX8 allocations:

- model activations / optimizer state
- framework allocator behavior (PyTorch, Python)
- user-retained tensors/batches

So a process can still OOM even when MX8 respects `max_inflight_bytes`.

## operator usage

Treat memory safety as layered:

1. Keep `max_inflight_bytes` strict.
2. Keep `prefetch_batches` and `max_queue_batches` conservative.
3. Monitor `loader.stats()` (`inflight_bytes`, `ram_high_water_bytes`).
4. Avoid retaining batches beyond the step loop.

## v1 direction (planned)

To close the remaining OOM gap:

- `max_batch_bytes` / `max_batch_tokens`: cap per-batch workload size.
- `max_ram_bytes`: whole-process watchdog.
  - soft threshold: throttle intake + warn.
  - hard threshold: fail fast with clear error (instead of OS OOM kill).

## one-line user contract

v0: MX8 guarantees bounded loader-path memory, not total-process memory.  
v1: MX8 adds process-level RSS guardrails for full end-to-end OOM protection.
