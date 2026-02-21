# MX8 Python API (v0)

This page documents the API that ships today in `mx8==0.x`.

## Install

- `pip install mx8`
- Vision/training helpers: `pip install pillow numpy torch`

## Snapshot + packing

- `mx8.resolve_manifest_hash(dataset_link, *, manifest_store=None, manifest_path=None, recursive=True, node_id=None) -> str`
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
    batch_size_samples=64,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    sample_ids = batch.sample_ids
    payload = batch.payload
    offsets = batch.offsets
```

### `mx8.load` args

- `dataset_link` (`plain`, `@refresh`, `@sha256:...`)
- `manifest_store` (default `/var/lib/mx8/manifests`)
- `manifest_path` (dev-only local manifest source)
- `recursive` (default `True`; set `False` to only index top-level objects/files under the prefix)
- `batch_size_samples`
- `max_ram_gb` (recommended default cap input)
- `max_inflight_bytes`
- `max_queue_batches`
- `prefetch_batches`
- `target_batch_bytes`, `max_batch_bytes` (advanced overrides; default byte-aware batching is derived automatically)
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
    manifest_store="/var/lib/mx8/manifests",
    max_ram_gb=24,
    profile="throughput",
    autotune=True,
    constraints=mx8.Constraints(max_inflight_bytes=512 * 1024 * 1024),
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
  - byte-batch jitter metrics:
    - `batch_payload_bytes_p50`
    - `batch_payload_bytes_p95`
    - `batch_payload_bytes_p95_over_p50`
    - `batch_payload_window_size`
    - `batch_jitter_slo_breaches_total`
    - `batch_jitter_band_adjustments_total`
    - `batch_jitter_band_lower_pct`
    - `batch_jitter_band_upper_pct`

Byte-aware batching is enabled by default (MX8 auto-derives target/max batch bytes from inflight caps). MX8 applies a tighter internal byte band around the target to reduce high/low batch-byte oscillation, emits a proof event when jitter SLO is breached, and adaptively tightens/relaxes the band with bounded hysteresis.

Hidden operator guard (env-only): set `MX8_MAX_PROCESS_RSS_BYTES` to enforce a process RSS hard limit (fail-fast instead of OS OOM kill).

## Image loader

Use `mx8.image(...)` when manifests include ImageFolder label hints.

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
    pass
```

Decode backend behavior:

- default/recommended in v0: Python/Pillow decode path
- optional Rust path for benchmarking/optimization: set `MX8_DECODE_BACKEND=rust`
- optional Rust decode worker count: set `MX8_DECODE_THREADS=<n>` (used when `MX8_DECODE_BACKEND=rust`)
- optional Rust JPEG codec: set `MX8_RUST_JPEG_CODEC=zune|image|turbo` (default: `zune`)
- optional Rust resize backend: set `MX8_RUST_RESIZE_BACKEND=fast|image` (default: `fast`)

## Video loader (GA current contract)

Use `mx8.video(...)` for clip-oriented delivery with bounded decode/runtime rails.

```python
import mx8

loader = mx8.video(
    "s3://bucket/video_prefix/",
    recursive=True,
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    clip_ids = batch.clip_ids
    payload = batch.payload
    offsets = batch.offsets
```

Current contract:

- `clip_ids`, `sample_ids`, `media_uris`, `clip_starts`, `offsets`, `payload`
- batch metadata: `frames_per_clip`, `frame_height`, `frame_width`, `channels`, `layout`, `dtype`, `colorspace`, `strides`
- offsets are monotonic and `offsets[-1] == len(payload)`
- init rejects invalid cap combinations (`batch_size_samples * bytes_per_clip > max_inflight_bytes`)
- default decode backend uses local `ffmpeg` CLI (`MX8_FFMPEG_BIN` override, default `ffmpeg`)
- supports startup autotune args (`profile`, `autotune`, `max_ram_gb`, `constraints`) for cap/profile selection
- runtime autotune is enabled by default (`autotune=False` disables) and adapts `max_inflight_bytes` within safe bounds
- `runtime` overrides are currently unsupported for `mx8.video(...)`

Video decode backend selection (Stage 3A):

- `MX8_VIDEO_DECODE_BACKEND=cli` (default): CLI decode path
- `MX8_VIDEO_DECODE_BACKEND=ffi`: native-FFI decode path (compile-time gated)
- if `ffi` is requested but not compiled/available, MX8 falls back to `cli` and emits proof log `event="video_decode_backend_fallback"`

Native FFI build notes:

- native ffmpeg backend is only compiled when built with: `RUSTFLAGS="--cfg mx8_video_ffi"`
- typical prerequisites include `pkg-config` plus ffmpeg development libs
- Stage 3A parity gate command:
  - default build path: `./scripts/video_stage3a_backend_gate.sh`
  - FFI-compiled path: `RUSTFLAGS="--cfg mx8_video_ffi" ./scripts/video_stage3a_backend_gate.sh`

S3 range-streaming status:

- shipped now: deterministic Stage 2D range sidecar/planner contract + gate
- roadmap: end-to-end S3 `Range` execution + decode from remote range segments

`loader.stats()` also includes video decode contract + reliability counters:

- contract: `video_layout`, `video_dtype`, `video_colorspace`, `video_frames_per_clip`, `video_frame_height`, `video_frame_width`, `video_channels`, `video_stride_t/h/w/c`, `video_clip_bytes`
- backend: `video_decode_backend` (`cli|ffi`)
- runtime decode counters: `video_decode_attempted_clips_total`, `video_decode_succeeded_clips_total`, `video_decode_failed_*_total`, `video_decode_failed_total`, `video_decode_ms_total`
- runtime autotune counters: `video_runtime_autotune_enabled`, `video_runtime_autotune_pressure`, `video_runtime_autotune_adjustments_total`

Video gate checklist:

- `./scripts/video_stage2b_gate.sh`
- `./scripts/video_stage2b_stress_gate.sh`
- `./scripts/video_stage2b_clean_env_gate.sh`
- `./scripts/video_stage2c_perf_gate.sh`
- `./scripts/video_stage3a_backend_gate.sh`
- `./scripts/video_ga_gate.sh --quick|--full`
- `docs/video_ga_checklist.md`

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
    max_ram_gb=24,
    profile="balanced",
)

for batch in loader:
    payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
```

Important fields:

- `want`: max concurrent leases per node.
- `progress_interval_ms`: progress report interval.
- `grpc_max_message_bytes`: gRPC message cap for manifest/control-path.
- distributed autotune supports API-driven profile/autotune control:
  - `profile=safe|balanced|throughput`
  - `autotune=True|False`
  - current implementation adjusts `want`, `prefetch_batches`, and `max_queue_batches` inside profile rails.

v0 training note: distributed data delivery is supported, but v0 is non-elastic (DDP rank death terminates training).

## `mx8.mix(...)` (v1.7)

`mx8.mix` provides deterministic weighted blending of multiple loaders under one shared bounded runtime envelope.

```python
import mx8

loader_a = mx8.load("s3://bucket/a@refresh", batch_size_samples=32)
loader_b = mx8.load("s3://bucket/b@refresh", batch_size_samples=32)

mixed = mx8.mix(
    [loader_a, loader_b],
    weights=[0.7, 0.3],
    seed=17,
    epoch=3,
    source_exhausted="error",  # default: fail fast
    max_ram_gb=12,
    profile="balanced",
)
```

Arguments:
- `loaders`: list of `mx8.load(...)` loader instances.
- `weights`: positive list, same length as `loaders`.
- `seed`, `epoch`: deterministic scheduling inputs.
- `starvation_window`: scheduler starvation accounting window.
- `source_exhausted`: `error|allow` (default `error`).
- `profile`: `safe|balanced|throughput` profile rails for shared mix defaults.
- `autotune`: when `True`, applies profile defaults before explicit overrides.
- `constraints`: optional `mx8.Constraints` override for shared mix cap.
- `runtime`: optional `mx8.RuntimeConfig` startup overrides (`prefetch_batches`, `max_queue_batches`; `want` is unsupported for `mx8.mix`).

Behavior:
- deterministic replay for fixed manifests/weights/seed/epoch/membership,
- shared inflight cap safety via global mixed guard,
- fail-fast source exhaustion by default (`error`) to avoid silent source drop,
- per-source diagnostics in `mixed.stats()["mix_sources"]` (manifest, delivery counters, configured knobs, and source metrics).

Gate commands:
- `./scripts/mix_gate.sh`
- strict mode: `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`
- multi-rank no-overlap gate: `./scripts/mix_multirank_gate.sh`
- smoke toggle: `MX8_SMOKE_MIX=1 ./scripts/smoke.sh`
- smoke multi-rank toggle: `MX8_SMOKE_MIX_MULTIRANK=1 ./scripts/smoke.sh`

## API modes

MX8 exposes two modes across loader APIs:

- default mode: provide workload intent and cap (`batch_size_samples`, `max_ram_gb`, `profile`) and let MX8 autotune runtime knobs
- advanced mode: provide explicit overrides (`autotune=False`, `constraints`, `RuntimeConfig`)

Formal contract note:

- `docs/python_api.md` + `docs/memory_contract.md` are the public contract surface.
