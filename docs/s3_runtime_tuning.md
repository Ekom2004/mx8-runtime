# S3 and Runtime Tuning (v0)

This page covers practical tuning for throughput, memory caps, and S3 throttling behavior.

See also: `docs/memory_contract.md` for the explicit v0/v1 memory guarantee model.

## What MX8 already does

- Hard memory/backpressure caps via:
  - `max_inflight_bytes`
  - `max_queue_batches`
  - `prefetch_batches`
- Transient S3 retry/backoff for throttling/server failures (including 429/5xx) in the fetch path.

## Why `max_inflight_bytes` (not `max_ram`) in v0

`max_inflight_bytes` is the strict cap MX8 can enforce directly: bytes admitted into the MX8 pipeline.

Process-wide RAM (`RSS`) includes memory MX8 does not own (PyTorch allocator, model activations, Python objects, fragmentation, other libs). Because of that, `max_ram` is not a hard contract in v0.

Practical meaning:

- `max_inflight_bytes` is a hard MX8 safety bound.
- `ram_high_water_bytes` is an observability metric.
- total process RSS can still OOM if non-MX8 allocations are too high.

## Main knobs

- `batch_size_samples`: how many samples per delivered batch.
- `prefetch_batches`: queued work depth before consumer.
- `max_inflight_bytes`: hard cap on bytes in-flight in the runtime pipeline.
- `max_queue_batches`: hard cap on queued delivered batches.
- `want` (`DistributedDataLoader`): max concurrent leases per node.
- `progress_interval_ms` (`DistributedDataLoader`): heartbeat/progress overhead vs responsiveness.

## Starting points

Single-node training/inference:

- `batch_size_samples=64..512`
- `prefetch_batches=2..4`
- `max_inflight_bytes=256MiB..1GiB`
- `max_queue_batches=8..64`

Distributed:

- Start with `want=1`
- Increase to `want=2..4` only after validating correctness and node stability
- Keep `progress_interval_ms` moderate (for example `250..1000`)

## How to tune safely

1. Fix correctness first (no overlap, job drains, deterministic hash where expected).
2. Raise `max_inflight_bytes` and `prefetch_batches` gradually.
3. Watch `loader.stats()` (`ram_high_water_bytes`, delivered totals).
4. Increase `want` only if nodes show idle gaps between leases.

## OOM prevention model

In v0, use layered caps:

- pipeline cap: `max_inflight_bytes`
- queue cap: `max_queue_batches`
- concurrency cap: `prefetch_batches` (plus distributed `want`)

In v1, planned additions complete the safety envelope:

- per-batch cap: `max_batch_bytes` / `max_batch_tokens`
- process cap: `max_process_rss_bytes` (whole process RAM watchdog)

## S3 throttling notes

- Throttling does not kill MX8 by itself; retries/backoff should keep progress moving.
- Very aggressive concurrency can increase throttle pressure and hurt throughput.
- For many-small-object datasets, packing to tar shards + `_mx8/manifest.tsv` is usually higher leverage than just raising concurrency.

## MinIO vs real S3

- MinIO is useful for deterministic local correctness gates.
- Do not treat MinIO localhost timings as cloud S3 production performance numbers.
