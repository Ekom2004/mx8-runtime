# S3 and Runtime Tuning

This page covers practical tuning for throughput, memory caps, and S3 behavior. For the formal memory guarantee model, see `docs/memory_contract.md`.


## What MX8 already handles for you

MX8 enforces hard memory and backpressure caps through `max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches`. Transient S3 errors — throttling, 429s, and 5xx responses — are handled automatically with retry and exponential backoff in the fetch path. You do not need to build retry logic on top of MX8.


## The main knobs

`batch_size_samples` controls how many samples are delivered per batch.

`prefetch_batches` controls how many batches the pipeline reads ahead of the consumer. More prefetch means better throughput but higher peak memory.

`max_inflight_bytes` is the hard cap on bytes currently moving through the pipeline. This is the primary memory safety control.

`max_queue_batches` caps how many delivered batches can wait in the queue before the pipeline applies backpressure.

`want` on `DistributedDataLoader` sets how many concurrent leases a node holds. More leases means more parallel S3 fetch, which increases inflight pressure.

`progress_interval_ms` on `DistributedDataLoader` controls how often progress is reported. Lower values give the coordinator more granular cursor updates at the cost of slightly more control-plane traffic.


## Starting values

For single-node training or inference, start with `batch_size_samples` between 64 and 512, `prefetch_batches` between 2 and 4, `max_inflight_bytes` between 256MB and 1GB, and `max_queue_batches` between 8 and 64.

For distributed workloads, start with `want=1`. Increase to 2 or 4 only after verifying correctness and node stability. Keep `progress_interval_ms` between 250 and 1000.


## How to tune safely

Fix correctness first. Verify no-overlap, that the job drains fully, and that the manifest hash is deterministic before touching throughput knobs.

Once correctness is confirmed, raise `max_inflight_bytes` and `prefetch_batches` gradually, watching `ram_high_water_bytes` and throughput in `loader.stats()` after each change.

Increase `want` only if nodes are showing idle gaps between leases. More concurrent leases help when fetch latency is the bottleneck, but they add memory pressure.

If you prefer not to tune manually, use `profile="balanced"` or `profile="throughput"` with `tune=True` and let MX8 derive and adapt the values for you.


## S3 throttling

S3 throttling does not kill MX8 by itself. The built-in retry and backoff logic handles transient throttle responses and keeps progress moving. Very aggressive concurrency — high `want`, many nodes, small objects — can increase throttle pressure and actually reduce throughput. If you see throttling, try reducing concurrency before increasing it.

For datasets with many small objects, packing to tar shards with a precomputed manifest is almost always higher leverage than raising concurrency. The LIST operations and per-object S3 requests that come with unpacked datasets are the real bottleneck, not the fetch bandwidth.


## External sort for large S3 snapshots

For very large prefixes where the in-memory sort during snapshot resolution exceeds available memory, set `MX8_SNAPSHOT_S3_EXTERNAL_SORT=true`. Use `MX8_SNAPSHOT_S3_SPILL_KEYS_PER_RUN` to control how many keys are held per spill run during the external sort.


## MinIO vs real S3

MinIO is useful for deterministic local correctness gates and development. It does not reflect real S3 performance characteristics — latency, throughput, and throttling behavior are all different. Do not use local MinIO timings as a baseline for production capacity planning.
