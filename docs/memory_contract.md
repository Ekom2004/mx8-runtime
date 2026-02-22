# MX8 Memory Contract (v1.8)

This page defines exactly what MX8 memory guarantees mean in v1.8.

## Hard guarantees in v1.8

MX8 can enforce two independent hard caps:

- `max_inflight_bytes`
  - caps bytes admitted into the MX8 pipeline (fetch/decode/queued delivery).
  - backpressure engages when full.
- `max_ram_bytes` (or env `MX8_MAX_PROCESS_RSS_BYTES`)
  - caps whole-process RSS (explicit value or derived production default).
  - MX8 fails fast with an explicit error when RSS exceeds the cap.

## Bounded runtime knobs

Use these together:

- `max_inflight_bytes` (pipeline byte budget)
- `max_queue_batches` (delivered queue depth)
- `prefetch_batches` (read-ahead depth)
- `want` (distributed concurrent leases per node)

These protect the data path from runaway buffering and lease over-requesting.

## What is conditional

MX8 derives a default process RSS cap on production-facing loader APIs when no explicit cap is provided.

Residual risk remains for non-MX8 allocations (model/framework/user code): if they grow faster than expected, the process may still hit the RSS fail-fast cap and abort by design.

## Operator posture

Treat memory safety as layered:

1. Set `max_inflight_bytes` explicitly.
2. Set `max_ram_bytes` (or `MX8_MAX_PROCESS_RSS_BYTES`) to enforce a strict org-level cap; otherwise MX8 uses a derived default.
3. Keep `prefetch_batches` and `max_queue_batches` conservative for the workload.
4. Monitor `loader.stats()` (`inflight_bytes`, `process_rss_bytes`, `ram_high_water_bytes`).
5. Avoid retaining batches past the step loop.

## One-line user contract

v1.8: MX8 guarantees bounded loader-path memory and defaults to whole-process RSS fail-fast protection on production-facing loader surfaces.
