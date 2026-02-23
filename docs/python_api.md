# Python API

This page documents the API that ships in `mx8==1.8.x`. Install with `pip install mx8`. For vision and training helpers, also install `pillow numpy torch`.


## Packing and snapshot resolution

Before you can load from a dataset at production scale, you typically pack it once to create a precomputed manifest. The packer writes tar shards and a manifest TSV that MX8 uses for all subsequent runs.

`mx8.pack(pack_in, *, out, shard_mb=512, label_mode="auto", require_labels=False)` packs an S3 prefix into a packed dataset at the output prefix.

`mx8.pack_dir(in_dir, *, out, shard_mb=512, label_mode="auto", require_labels=False)` does the same for a local directory.

`mx8.resolve_manifest_hash(dataset_link, *, manifest_store=None, manifest_path=None, recursive=True, node_id=None)` resolves a dataset link to its pinned manifest hash without starting a loader.

`label_mode` accepts `auto`, `none`, or `imagefolder`.


## Core loader

`mx8.load` is the foundation for byte-oriented pipelines — ETL, inference, preprocessing. It returns batches of raw bytes with sample IDs and byte offsets.

```python
import mx8

loader = mx8.load(
    "s3://bucket/prefix/",
    batch_size_samples=64,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    sample_ids = batch.sample_ids
    payload    = batch.payload
    offsets    = batch.offsets
```

Arguments:

`dataset_link` is a plain path or prefix, optionally followed by `@refresh` to force a fresh snapshot at job start, or `@sha256:<hash>` to pin an exact snapshot.

`manifest_store` sets the manifest store root (default `~/.mx8/manifests`). Accepts a filesystem path or an S3 prefix.

`recursive` controls whether subdirectories are indexed (default `True`).

`batch_size_samples` sets how many samples per delivered batch.

`max_ram_gb` is the recommended way to set the memory budget. MX8 derives `max_inflight_bytes` and the process RSS cap from this value based on your profile.

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are advanced overrides if you need precise control over the pipeline budget.

`target_batch_bytes` and `max_batch_bytes` override the byte-aware batching defaults. MX8 derives these automatically from your inflight cap in most cases.

`start_id` and `end_id` must be set together if you want to load a specific sample ID range.

`profile` selects a preset safety/throughput balance: `safe`, `balanced`, or `throughput`.

`autotune` enables the runtime adaptation loop (default `True`). Set to `False` for fully manual control.

`constraints` accepts an `mx8.Constraints` instance to override specific cap values while keeping autotune active.

`runtime` accepts an `mx8.RuntimeConfig` instance for explicit startup values when `autotune=False`.

```python
# Autotune with explicit constraint
loader = mx8.load(
    "s3://bucket/train@refresh",
    max_ram_gb=24,
    profile="throughput",
    autotune=True,
    constraints=mx8.Constraints(max_inflight_bytes=512 * 1024 * 1024),
)

# Fully manual
loader = mx8.load(
    "s3://bucket/train@refresh",
    autotune=False,
    runtime=mx8.RuntimeConfig(prefetch_batches=8, max_queue_batches=32),
)
```

Each batch exposes `sample_ids`, `offsets`, `payload`, and `label_ids`. Call `batch.to_torch()` to get `(payload_u8, offsets_i64, sample_ids_i64)` as tensors, or `batch.to_torch_with_labels()` to include labels.


## Loader stats and monitoring

`loader.stats()` returns a plain Python dict. Poll it each step or every N steps to track pipeline health.

Key fields: `delivered_batches_total`, `delivered_samples_total`, `inflight_bytes`, `process_rss_bytes`, `ram_high_water_bytes`.

Byte-batch jitter fields: `batch_payload_bytes_p50`, `batch_payload_bytes_p95`, `batch_payload_bytes_p95_over_p50`, `batch_payload_window_size`, `batch_jitter_slo_breaches_total`, `batch_jitter_band_adjustments_total`, `batch_jitter_band_lower_pct`, `batch_jitter_band_upper_pct`.

MX8 applies a tighter internal byte band around the target batch size to reduce oscillation, emits a proof event when the jitter SLO is breached, and adaptively tightens or relaxes the band with bounded hysteresis.

A top-level `mx8.stats()` API is planned but not yet shipped in v1.8. Use `loader.stats()` directly.

To wire loader stats into Prometheus:

```python
from prometheus_client import Gauge, start_http_server
import mx8

start_http_server(9090)

inflight  = Gauge("mx8_inflight_bytes",       "MX8 pipeline inflight bytes")
rss       = Gauge("mx8_process_rss_bytes",    "MX8 process RSS bytes")
hwm       = Gauge("mx8_ram_high_water_bytes", "MX8 RAM high-water mark")
delivered = Gauge("mx8_delivered_samples",    "MX8 delivered samples total")

loader = mx8.load("s3://bucket/dataset/", max_ram_gb=12, profile="balanced")

for batch in loader:
    train_step(batch)
    s = loader.stats()
    inflight.set(s["inflight_bytes"])
    rss.set(s["process_rss_bytes"])
    hwm.set(s["ram_high_water_bytes"])
    delivered.set(s["delivered_samples_total"])
```

To wire into Datadog:

```python
from datadog import initialize, statsd
import mx8

initialize()

loader = mx8.load("s3://bucket/dataset/", max_ram_gb=12, profile="balanced")

for step, batch in enumerate(loader):
    train_step(batch)
    if step % 50 == 0:
        s = loader.stats()
        statsd.gauge("mx8.inflight_bytes",       s["inflight_bytes"])
        statsd.gauge("mx8.process_rss_bytes",    s["process_rss_bytes"])
        statsd.gauge("mx8.ram_high_water_bytes", s["ram_high_water_bytes"])
        statsd.gauge("mx8.delivered_samples",    s["delivered_samples_total"])
```

Alert when `ram_high_water_bytes` approaches your `MX8_MAX_PROCESS_RSS_BYTES` cap, when `batch_jitter_slo_breaches_total` is rising steadily, or when `inflight_bytes` is pegged at the cap for many consecutive steps.

A native Prometheus endpoint on the coordinator and agent is planned for a future release.


## Image loader

`mx8.image` delivers decoded images and labels as tensors. Use it when your dataset includes ImageFolder label hints.

```python
import mx8

loader = mx8.image(
    "s3://bucket/mx8/train/@refresh",
    batch_size_samples=64,
    max_ram_gb=12,
    resize_hw=(224, 224),
    profile="balanced",
)

print(loader.classes)  # ["cat", "dog", ...] or None

for images, labels in loader:
    pass  # images: [B,C,H,W] float32, labels: [B] int64
```

The default decode backend in v1.8 is Python/Pillow. To use the experimental Rust decode path, set `MX8_DECODE_BACKEND=rust`. The Rust path supports additional options: `MX8_DECODE_THREADS` for worker count, `MX8_RUST_JPEG_CODEC` for JPEG codec selection (`zune`, `image`, or `turbo`), and `MX8_RUST_RESIZE_BACKEND` for resize algorithm (`fast` or `image`).


## Video loader

`mx8.video` delivers decoded video clips with bounded decode and runtime rails.

```python
import mx8

loader = mx8.video(
    "s3://bucket/video_prefix/",
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    clip_ids = batch.clip_ids
    payload  = batch.payload
    offsets  = batch.offsets
```

Each batch includes `clip_ids`, `sample_ids`, `media_uris`, `clip_starts`, `offsets`, and `payload`. Batch metadata fields include `frames_per_clip`, `frame_height`, `frame_width`, `channels`, `layout`, `dtype`, `colorspace`, and `strides`. Offsets are monotonic and `offsets[-1] == len(payload)`.

The loader rejects invalid cap combinations at init — specifically when `batch_size_samples * bytes_per_clip > max_inflight_bytes`. Runtime autotune is enabled by default and adapts `max_inflight_bytes` within safe bounds. Pass `autotune=False` to disable.

The default decode backend uses a local `ffmpeg` CLI. Override the binary path with `MX8_FFMPEG_BIN`. To use the native FFI path, set `MX8_VIDEO_DECODE_BACKEND=ffi` and build with `RUSTFLAGS="--cfg mx8_video_ffi"`. If the FFI path is requested but not compiled, MX8 falls back to CLI and emits a `video_decode_backend_fallback` proof event.

`loader.stats()` for the video loader includes decode contract fields (`video_layout`, `video_dtype`, `video_colorspace`, `video_frames_per_clip`, `video_frame_height`, `video_frame_width`, `video_channels`, `video_clip_bytes`), backend selection (`video_decode_backend`), decode counters (`video_decode_attempted_clips_total`, `video_decode_succeeded_clips_total`, `video_decode_failed_total`, `video_decode_ms_total`), and autotune counters (`video_runtime_autotune_enabled`, `video_runtime_autotune_pressure`, `video_runtime_autotune_adjustments_total`).

Gate commands for the video loader: `./scripts/video_stage2b_gate.sh`, `./scripts/video_stage2b_stress_gate.sh`, `./scripts/video_stage2c_perf_gate.sh`, `./scripts/video_stage3a_backend_gate.sh`, and `./scripts/video_ga_gate.sh`.


## Distributed loader

`mx8.DistributedDataLoader` is the distributed control-plane loader for DDP-style jobs. It connects to a running coordinator and participates in the lease protocol.

```python
import mx8

loader = mx8.DistributedDataLoader(
    coord_url="http://127.0.0.1:50051",
    job_id="demo",
    node_id="node1",
    batch_size_samples=512,
    max_ram_gb=24,
    profile="balanced",
)

for batch in loader:
    payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
```

`want` sets the max number of concurrent leases this node will request. `progress_interval_ms` controls how often progress is reported to the coordinator (default 500ms). `grpc_max_message_bytes` caps the gRPC message size for manifest and control-path traffic (default 64MB).

Distributed autotune adjusts `want`, `prefetch_batches`, and `max_queue_batches` within the chosen profile rails. Pass `profile` and `autotune=True|False` to control it.

Training note: v1.8 supports distributed data delivery but is non-elastic. If a DDP rank dies, the job terminates. Lease reassignment is for inference and ETL correctness, not elastic training continuation.


## Mix API

`mx8.mix` blends multiple loaders deterministically under one shared memory envelope. Use it when you want to train on a weighted combination of datasets without managing the interleaving yourself.

```python
import mx8

loader_a = mx8.load("s3://bucket/a@refresh", batch_size_samples=32)
loader_b = mx8.load("s3://bucket/b@refresh", batch_size_samples=32)

mixed = mx8.mix(
    [loader_a, loader_b],
    weights=[0.7, 0.3],
    seed=17,
    epoch=3,
    source_exhausted="error",
    max_ram_gb=12,
    profile="balanced",
)
```

`weights` is a list of positive floats, normalized internally. `seed` and `epoch` are the deterministic scheduling inputs. `source_exhausted` controls what happens when a source runs out: `error` fails fast (the default), `allow` drains remaining sources. `starvation_window` controls the scheduler starvation accounting window.

For a fixed set of manifests, weights, seed, epoch, and frozen membership, the mixed stream order is deterministic and replayable. All sources share one inflight cap — backpressure is global. `mixed.stats()["mix_sources"]` exposes per-source diagnostics including manifest IDs, delivery counters, and configured knobs.

Gate commands: `./scripts/mix_gate.sh`, `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`, `./scripts/mix_multirank_gate.sh`.
