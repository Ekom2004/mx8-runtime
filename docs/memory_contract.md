# Memory Contract

This page defines exactly what MX8 memory guarantees mean in v1.8 and what remains the operator's responsibility.


## What MX8 guarantees

MX8 enforces two independent hard caps on every production-facing loader surface.

`max_inflight_bytes` caps the bytes admitted into the MX8 pipeline — everything in the fetch, decode, and queued delivery stages. When this cap is full, backpressure engages and no new fetches start. This is a strict mechanical guarantee: MX8 will not admit more bytes than you configure.

`max_ram_bytes` (or `MX8_MAX_PROCESS_RSS_BYTES`) caps the whole-process RSS. MX8 samples the process RSS periodically and fails fast with an explicit error if it exceeds the cap. This protects the process from being silently OOM-killed by the OS.

Both caps are enforced independently. You can set one or both. On production-facing loaders, MX8 derives a default process RSS cap if you do not set one explicitly.


## The runtime knobs

Four knobs work together to bound memory usage end to end:

`max_inflight_bytes` is the pipeline byte budget. Set this first.

`max_queue_batches` caps the number of delivered batches waiting to be consumed. If your consumer is slow, this prevents the queue from growing unbounded.

`prefetch_batches` controls how many batches the pipeline reads ahead. More prefetch means better throughput but more memory.

`want` (on `DistributedDataLoader`) controls how many concurrent leases this node holds. More leases means more parallel fetch, which increases inflight pressure.

Set these conservatively to start and increase based on observed throughput and `ram_high_water_bytes` from `loader.stats()`.


## What is not guaranteed

MX8 only owns the loader-path memory. The process also allocates for model weights, framework internals, Python objects, and anything else running in the same process. If those allocations grow faster than expected, the process can still hit the RSS fail-fast cap and abort — which is the correct behavior. The alternative is an OS OOM kill with no warning.

Treat the memory contract as: MX8 holds its portion of memory within your configured bounds. The rest is your code's responsibility.


## Operator checklist

Set `max_inflight_bytes` explicitly for every deployment. Do not rely on defaults in production.

Set `MX8_MAX_PROCESS_RSS_BYTES` as an org-wide policy, not per-job. This ensures consistent protection across all jobs even when individual configurations vary.

Keep `prefetch_batches` and `max_queue_batches` conservative for the workload, especially on nodes with other heavy processes running.

Monitor `inflight_bytes`, `process_rss_bytes`, and `ram_high_water_bytes` from `loader.stats()` to understand actual usage versus configured caps.

Do not retain batches past the training or inference step loop. Accumulating batches in a list outside the loop bypasses the queue cap and defeats backpressure.
