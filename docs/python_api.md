# MX8 Python API (v0)

This page documents the API that ships today in `mx8==0.x`.

## Install

- `pip install mx8`
- Vision/training helpers: `pip install pillow numpy torch`

## Snapshot + packing

- `mx8.resolve_manifest_hash(dataset_link, *, manifest_store_root=None, dev_manifest_path=None, recursive=True, node_id=None) -> str`
- `mx8.pack(pack_in, *, out, shard_mb=512, label_mode="auto", require_labels=False) -> dict`
  - S3 input/output packer (`s3://...`).
- `mx8.pack_dir(in_dir, *, out, shard_mb=512, label_mode="auto", require_labels=False) -> dict`
  - Local directory packer.

`label_mode` accepts `auto|none|imagefolder`.

## Core loader

Use `mx8.load(...)` for byte-oriented pipelines (ETL/inference/preprocess).

```python
import mx8

loader = mx8.load(
    "s3://bucket/prefix/",
    recursive=True,  # default
    profile="balanced",
    autotune=True,
)

for batch in loader:
    sample_ids = batch.sample_ids
    payload = batch.payload
    offsets = batch.offsets
```

### `mx8.load` args

- `dataset_link` (`plain`, `@refresh`, `@sha256:...`)
- `manifest_store_root` (default `/var/lib/mx8/manifests`)
- `dev_manifest_path` (dev-only local manifest source)
- `recursive` (default `True`; set `False` to only index top-level objects/files under the prefix)
- `batch_size_samples`
- `max_inflight_bytes`
- `max_queue_batches`
- `prefetch_batches`
- `max_process_rss_bytes`
- `start_id`, `end_id` (must be set together)
- `node_id` (for lock ownership/proof logs)
- `profile` (`safe|balanced|throughput`)
- `autotune` (`True|False`)
- `constraints` (`mx8.Constraints`)
- `runtime` (`mx8.RuntimeConfig`)

### `mx8.Constraints` and `mx8.RuntimeConfig`

```python
import mx8

loader = mx8.load(
    "s3://bucket/train@refresh",
    profile="throughput",
    autotune=True,
    constraints=mx8.Constraints(
        max_inflight_bytes=512 * 1024 * 1024,
        max_process_rss_bytes=24 * 1024 * 1024 * 1024,
    ),
)

loader_manual = mx8.load(
    "s3://bucket/train@refresh",
    autotune=False,
    runtime=mx8.RuntimeConfig(prefetch_batches=8, max_queue_batches=32),
)
```

`runtime.want` is currently consumed by `mx8.DistributedDataLoader` (not by single-node `mx8.load`).

### `PyBatch` methods/properties

- `sample_ids` (`list[int]`)
- `offsets` (`list[int]`)
- `payload` (`bytes`)
- `label_ids` (`list[int] | None`)
- `to_torch()` -> `(payload_u8, offsets_i64, sample_ids_i64)`
- `to_torch_with_labels()` -> `(payload_u8, offsets_i64, sample_ids_i64, labels_i64)`

### Loader stats

- `loader.stats()` returns:
  - `delivered_batches_total`
  - `delivered_samples_total`
  - `inflight_bytes`
  - `process_rss_bytes`
  - `ram_high_water_bytes`

Hidden operator guard (env-only): set `MX8_MAX_PROCESS_RSS_BYTES` to enforce a process RSS hard limit (fail-fast instead of OS OOM kill).

## Vision loader

Use `mx8.vision.ImageFolderLoader(...)` when manifests include ImageFolder label hints.

```python
import mx8

loader = mx8.vision.ImageFolderLoader(
    "s3://bucket/mx8/train/@refresh",
    batch_size_samples=64,
    resize_hw=(224, 224),
    max_inflight_bytes=256 * 1024 * 1024,
)

print(loader.classes)  # ["cat", "dog", ...] or None
for images, labels in loader:
    pass
```

Decode backend behavior:

- default/recommended in v0: Python/Pillow decode path
- optional Rust path for benchmarking/optimization: set `MX8_DECODE_BACKEND=rust`
- optional Rust decode worker count: set `MX8_DECODE_THREADS=<n>` (used when `MX8_DECODE_BACKEND=rust`)
- optional Rust JPEG codec: set `MX8_RUST_JPEG_CODEC=zune|image|turbo` (default: `zune`)
- optional Rust resize backend: set `MX8_RUST_RESIZE_BACKEND=fast|image` (default: `fast`)

## Video loader (v1.8 scaffold)

Use `mx8.video(...)` for clip-oriented delivery scaffolding while CPU decode contract work is in progress.

```python
import mx8

loader = mx8.video(
    "s3://bucket/video_prefix/",
    recursive=True,
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    constraints=mx8.Constraints(max_inflight_bytes=128 * 1024 * 1024),
)

for batch in loader:
    clip_ids = batch.clip_ids
    payload = batch.payload
    offsets = batch.offsets
```

Current staged contract:

- `clip_ids`, `sample_ids`, `media_uris`, `clip_starts`, `offsets`, `payload`
- batch metadata: `frames_per_clip`, `frame_height`, `frame_width`, `channels`, `layout`, `dtype`, `colorspace`, `strides`
- offsets are monotonic and `offsets[-1] == len(payload)`
- init rejects invalid cap combinations (`batch_size_samples * bytes_per_clip > max_inflight_bytes`)
- current CPU decode backend uses local `ffmpeg` CLI (`MX8_FFMPEG_BIN` override, default `ffmpeg`)

`loader.stats()` also includes video decode contract + reliability counters:

- contract: `video_layout`, `video_dtype`, `video_colorspace`, `video_frames_per_clip`, `video_frame_height`, `video_frame_width`, `video_channels`, `video_stride_t/h/w/c`, `video_clip_bytes`
- runtime decode counters: `video_decode_attempted_clips_total`, `video_decode_succeeded_clips_total`, `video_decode_failed_*_total`, `video_decode_failed_total`, `video_decode_ms_total`

Internal contract and gate checklist:

- `docs/internal_video_stage2b_contract.md`
- `./scripts/video_stage2b_gate.sh`
- `./scripts/video_stage2b_stress_gate.sh`

Runtime proof logs (target: `mx8_proof`) now include:

- `event="video_decode_batch"` for delivered decode batches
- `event="video_decode_failed"` with failure class + media URI context

## Distributed loader (DDP/local multi-rank)

`mx8.DistributedDataLoader` is the distributed control-plane loader used by DDP-style jobs.

```python
import mx8

loader = mx8.DistributedDataLoader(
    coord_url="http://127.0.0.1:50051",
    job_id="demo",
    node_id="node1",
    batch_size_samples=512,
    want=1,
)

for batch in loader:
    payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
```

Important fields:

- `want`: max concurrent leases per node.
- `progress_interval_ms`: progress report interval.
- `grpc_max_message_bytes`: gRPC message cap for manifest/control-path.
- experimental distributed autotune (env-gated):
  - `MX8_AUTOTUNE=1`
  - `MX8_AUTOTUNE_PROFILE=safe|balanced|throughput` (default `balanced`)
  - current implementation adjusts `want`, `prefetch_batches`, and `max_queue_batches` inside profile rails.

v0 training note: distributed data delivery is supported, but v0 is non-elastic (DDP rank death terminates training).

## API shape (v0 vs v1 direction)

v0 still supports explicit tuning knobs in constructor args (`max_inflight_bytes`, `max_queue_batches`, `prefetch_batches`, `want`, ...).

v0 also ships the v1-style preview for `mx8.load(...)`:

- simple path (`profile` + `autotune`)
- advanced path (`constraints` + `RuntimeConfig`)

Formal contract (single source of truth):

- `docs/v1_autotune_api_contract.md` (includes control equations, default gains, rails, and observability fields)
