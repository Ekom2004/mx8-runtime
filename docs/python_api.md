# Python API

This page documents the API that ships in `mx8==1.0.5`. Install with `pip install mx8`. For vision and training helpers, also install `pillow numpy torch`.


## Packing and snapshot resolution

Before you can load from a dataset at production scale, you typically pack it once to create a precomputed manifest. The packer writes tar shards and a manifest TSV that MX8 uses for all subsequent runs.

`mx8.pack(pack_in, *, out, shard_mb=512, label_mode="auto", require_labels=False, parallel_fetches=128)` packs an S3 prefix into tar shards at the output prefix and writes a manifest. `parallel_fetches` controls how many S3 GET requests run concurrently during packing (override with `MX8_PACK_PARALLEL_FETCHES`). Part size for the multipart upload defaults to 16MB and is tunable via `MX8_PACK_PART_MB`.

`mx8.pack_dir(in_dir, *, out, shard_mb=512, label_mode="auto", require_labels=False)` does the same for a local directory.

`mx8.resolve_manifest_hash(dataset_link, *, manifest_store=None, manifest_path=None, recursive=True, node_id=None)` resolves a dataset link to its pinned manifest hash without starting a loader.

`label_mode` accepts `auto`, `none`, or `imagefolder`.


## Core loader (`mx8.load`)

`mx8.load` is the primary byte-loader API.

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

Distributed attach is built into loader APIs. Shared optional args:

- `job_id` (required for distributed mode)
- `cluster_url` (explicit coordinator URL; falls back to `MX8_CLUSTER_URL` / `MX8_COORD_URL`)
- `resume_from` (checkpoint token)

Single node:

```python
loader = mx8.load("s3://bucket/prefix/", batch_size_samples=64, max_ram_gb=12)
```

Distributed (explicit attach):

```python
loader = mx8.load(
    "s3://bucket/prefix/",
    batch_size_samples=512,
    max_ram_gb=24,
    job_id="train-job-001",
    cluster_url="http://coordinator-host:50051",
)
```

Distributed (env attach):

```python
# WORLD_SIZE>1, MX8_JOB_ID, MX8_CLUSTER_URL set by launcher
loader = mx8.load("s3://bucket/prefix/", batch_size_samples=512, max_ram_gb=24)
```

Arguments:

`dataset_link` is a plain path or prefix, optionally followed by `@refresh` to force a fresh snapshot at job start, or `@sha256:<hash>` to pin an exact snapshot.

`manifest_store` sets the manifest store root (default `~/.mx8/manifests`). Pass a filesystem path.

`manifest_path` optionally points to a local manifest TSV for development/override flows.

`recursive` controls whether subdirectories are indexed (default `True`).

`batch_size_samples` sets how many samples per delivered batch.

`max_ram_gb` is the recommended way to set the memory budget. MX8 derives `max_inflight_bytes` and the process RSS cap from this value based on your profile.

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are accepted in the function signature for compatibility, but current top-level loader entrypoints derive effective values from `profile`, `constraints`, and `runtime` instead.

`target_batch_bytes` and `max_batch_bytes` override the byte-aware batching defaults. MX8 derives these automatically from your inflight cap in most cases.

`start_id` and `end_id` must be set together if you want to load a specific sample ID range.

`resume_from` (default `None`) accepts an opaque checkpoint token from `loader.checkpoint()`. When set, MX8 resumes from the token cursor within the requested range. Validation is strict: `manifest_hash`, `schema_version`, `epoch`, and `end_id` must match the current run.

`job_id` (default `None`) is required for distributed mode.

`cluster_url` (default `None`) sets the coordinator URL directly for distributed mode.

`node_id` sets node identity used in proof logs and coordinator ownership hints. In distributed attach mode, default is `rank{RANK}` when unset.

`autopack` (default `False`) — if `True` and `dataset_link` is a bare S3 prefix with no manifest, MX8 runs a full pack in-place before starting the loader. The packed shards and manifest are written to the same prefix. Subsequent runs skip the pack step automatically. Use `autopack_shard_mb` (default `512`) to control shard size.

`profile` selects a preset safety/throughput balance: `safe`, `balanced`, or `throughput`.

`autotune` enables the runtime adaptation loop (default `True`). Set to `False` for fully manual control.

`constraints` accepts an `mx8.Constraints` instance to override specific cap values while keeping autotune active.

`runtime` accepts an `mx8.RuntimeConfig` instance for explicit startup values when `autotune=False`.

Compatibility note:
- For `mx8.load`, prefer `constraints` and `runtime` for runtime shaping.
- `constraints.max_inflight_bytes` and `constraints.max_ram_bytes` are the active hard-cap knobs.
- `runtime.prefetch_batches` and `runtime.max_queue_batches` are the active startup queue knobs.

Each batch exposes `sample_ids`, `offsets`, `payload`, and `label_ids`. Call `batch.to_torch()` to get `(payload_u8, offsets_i64, sample_ids_i64)` as tensors, or `batch.to_torch_with_labels()` to include labels.


## Multi-epoch training

Loaders are **single-pass**. When all samples have been delivered the loader raises `StopIteration` and stops permanently — it does not restart on its own. Calling `next()` after exhaustion immediately raises `StopIteration` again.

For multi-epoch training create a new loader at the start of each epoch:

```python
import mx8

dataset = "s3://bucket/train@refresh"

for epoch in range(10):
    loader = mx8.load(dataset, batch_size_samples=64, max_ram_gb=8, profile="balanced")
    for batch in loader:
        train_step(batch)
```

To ensure every epoch sees the **exact same snapshot** (i.e. the dataset doesn't change mid-training), pin the manifest hash once before the loop and reuse it:

```python
import mx8

# Resolve once — captures the dataset state at job start.
manifest_hash = mx8.resolve_manifest_hash("s3://bucket/train@refresh")
pinned = f"s3://bucket/train@sha256:{manifest_hash}"

for epoch in range(10):
    loader = mx8.load(pinned, batch_size_samples=64, max_ram_gb=8, profile="balanced")
    for batch in loader:
        train_step(batch)
```

`loader.close()` shuts down the background pipeline immediately. It is called automatically by `__del__` when the loader goes out of scope, so explicit cleanup is optional but recommended in long-running processes to release RAM and network connections promptly.


## Loader stats and monitoring

`loader.stats()` returns a plain Python dict. Poll it each step or every N steps to track pipeline health.

Key fields: `delivered_batches_total`, `delivered_samples_total`, `process_rss_bytes`, `max_process_rss_bytes`, `ram_high_water_bytes`, `elapsed_seconds`.

Byte-batch jitter fields: `batch_payload_bytes_p50`, `batch_payload_bytes_p95`, `batch_payload_bytes_p95_over_p50`, `batch_payload_window_size`, `batch_jitter_slo_breaches_total`, `batch_jitter_band_adjustments_total`, `batch_jitter_band_lower_pct`, `batch_jitter_band_upper_pct`.

MX8 applies a tighter internal byte band around the target batch size to reduce oscillation, emits a proof event when the jitter SLO is breached, and adaptively tightens or relaxes the band with bounded hysteresis.

`mx8.stats(loader)` is now available as a one-shot human-readable snapshot.

`mx8.stats(loader, raw=True)` returns the raw dict from `loader.stats()` unchanged.

`mx8.stats(loader)` contract:
- Always show: `status`, `mode`, `epoch/step` (when present), `progress`, `throughput`, `memory`, `stability`.
- Conditional sections: distributed details and autotune details are shown only when those fields exist for that loader type.
- Snapshot only: this is not a live TUI stream; call repeatedly in your loop if you want a live terminal view.

```python
import mx8

loader = mx8.load("s3://bucket/dataset/", max_ram_gb=12, profile="balanced")
print(mx8.stats(loader))           # human-readable snapshot
raw = mx8.stats(loader, raw=True)  # same dict as loader.stats()
```

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


## Text loader

`mx8.text` tokenizes text samples inside Rust and emits fixed-shape token batches for direct model input.

```python
import mx8

loader = mx8.text(
    "s3://bucket/corpus@refresh",
    tokenizer="gpt2",
    sequence_length=2048,
    stride=2048,
    add_bos=False,
    add_eos=True,
    batch_size_samples=32,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    token_ids = batch.token_ids            # [B, sequence_length] int64
    attention = batch.attention_mask       # [B, sequence_length] bool (or None)
    sample_ids = batch.sample_ids          # [B] int64
```

Text loader arguments:

`tokenizer` — tokenizer reference. Supports `"gpt2"` or a local `tokenizer.json` path.

`sequence_length` — fixed output token length for each emitted row.

`stride` — step size for chunking long tokenized sequences (use `stride < sequence_length` for overlap).

`add_bos`, `add_eos` — prepend/append BOS/EOS token IDs when available in tokenizer config.

`truncate` — long-sequence handling (`"right"` or `"error"`).

`return_attention_mask` — emit `batch.attention_mask` when `True`.

`decode_error_policy` — UTF-8 decode handling (`"error"` default, `"skip"` optional).

`manifest_store`, `manifest_path`, `recursive`, `start_id`, `end_id`, `autopack`, and `autopack_shard_mb` — same semantics as `mx8.load`.

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are accepted for compatibility, but effective values are currently derived from `profile`, `constraints`, and `runtime`.

`job_id`, `cluster_url`, `resume_from` — same distributed/restore contract as `mx8.load`.

`loader.checkpoint()` returns an opaque token compatible with `mx8.text(..., resume_from=token)`.

`loader.stats()` for the text loader includes shared pipeline counters plus text fields: `text_sequence_length`, `text_stride`, and `text_return_attention_mask`.

Gate commands for the text loader: `./scripts/text_gate.sh`.


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

With the augmentation preset:

```python
loader = mx8.image(
    "s3://bucket/mx8/train/@refresh",
    augment="imagenet",
    batch_size_samples=64,
    max_ram_gb=12,
    seed=42,
)

for images, labels in loader:
    pass  # images: [B,C,H,W] float32, augmented and normalized
```

Image loader arguments:

`resize_hw` — `(height, width)` tuple to resize all images before batching (e.g. `resize_hw=(224, 224)`). If not set, images are returned at their original size. All images in a batch must have the same decoded dimensions; mixed-size batches are an error.

`augment` — optional preset alias. `augment="imagenet"` (or `"standard"`) expands defaults for common classification training: `resize_hw=(256,256)` when unset, `crop_hw=(224,224)`, `horizontal_flip_p=0.5`, `color_jitter_{brightness,contrast,saturation}=0.4`, and ImageNet normalization (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`).

`crop_hw` — optional random crop size `(height, width)` applied after decode/resize.

`horizontal_flip_p` — per-sample horizontal flip probability in `[0, 1]`.

`color_jitter_brightness`, `color_jitter_contrast`, `color_jitter_saturation` — non-negative jitter magnitudes. `color_jitter_hue` is currently accepted only as `0.0` (hue jitter is not yet implemented).

`normalize_mean`, `normalize_std` — optional per-channel normalization tuples. Pass both together. `std` values must be `> 0`.

`seed`, `epoch` — deterministic augmentation controls. For a fixed snapshot (`manifest_hash`), `seed`, `epoch`, and `sample_id`, augment choices are reproducible.

`to_float` (default `True`) — normalize pixel values to `float32` in `[0, 1]`. Set `to_float=False` to get raw `uint8` tensors.

`node_id` — same as the core loader. In distributed attach mode, default is `rank{RANK}` when unset.

`manifest_store`, `manifest_path`, `recursive`, `start_id`, `end_id`, `autopack`, and `autopack_shard_mb` — same semantics as `mx8.load`.

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are accepted for compatibility, but effective values are currently derived from `profile`, `constraints`, and `runtime`.

`job_id`, `cluster_url`, `resume_from` — same distributed/restore contract as `mx8.load`.

`loader.checkpoint()` returns an opaque token compatible with `mx8.image(..., resume_from=token)`.

`autopack` and `autopack_shard_mb` — same as the core loader.

The default decode backend in `mx8==1.0.5` is Python/Pillow. To use the experimental Rust decode path, set `MX8_DECODE_BACKEND=rust`. The Rust path supports additional options: `MX8_DECODE_THREADS` for worker count, `MX8_RUST_JPEG_CODEC` for JPEG codec selection (`zune`, `image`, or `turbo`), and `MX8_RUST_RESIZE_BACKEND` for resize algorithm (`fast` or `image`).

Augmentation order is fixed and deterministic: `decode -> resize -> crop -> flip -> jitter -> normalize`.

Gate commands for the image loader: `./scripts/image_aug_gate.sh`, `./scripts/py_local_image_pillow_gate.sh`, and `./scripts/py_minio_image_pillow_gate.sh`.


## Audio loader

`mx8.audio` decodes audio samples in Rust and emits fixed-shape mono tensors for model input.

```python
import mx8

loader = mx8.audio(
    "s3://bucket/audio@refresh",
    batch_size_samples=32,
    sample_count=16000,
    channels=1,
    sample_rate_hz=16000,
    decode_error_policy="error",
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    samples = batch.samples                  # [B, sample_count] float32
    sample_rates_hz = batch.sample_rates_hz  # [B] int64
    sample_ids = batch.sample_ids            # [B] int64
```

Audio loader arguments:

`sample_count` — fixed output frame count per emitted row. Short clips are zero-padded; long clips are truncated.

`channels` — output channels. `mx8==1.0.5` supports `channels=1` only (mono output).

`sample_rate_hz` — optional strict sample-rate check. When set, samples with mismatched decoded rates follow `decode_error_policy`.

`decode_error_policy` — decode failure handling (`"error"` default, `"skip"` optional).

`manifest_store`, `manifest_path`, `recursive`, `start_id`, `end_id`, `autopack`, and `autopack_shard_mb` — same semantics as `mx8.load`.

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are accepted for compatibility, but effective values are currently derived from `profile`, `constraints`, and `runtime`.

`job_id`, `cluster_url`, `resume_from` — same distributed/restore contract as `mx8.load`.

`loader.checkpoint()` returns an opaque token compatible with `mx8.audio(..., resume_from=token)`.

Supported formats in `mx8==1.0.5`: WAV and FLAC.

`loader.stats()` for the audio loader includes decode fields (`audio_sample_count`, `audio_channels`, `audio_expected_sample_rate_hz`, `audio_decode_samples_total`, `audio_decode_failures_total`, `audio_decoded_frames_total`) plus shared pipeline counters.

Gate command for the audio loader: `./scripts/audio_gate.sh`.


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

`mx8.video` supports `job_id`, `cluster_url`, and `resume_from`. In distributed attach mode it applies deterministic rank/world clip sharding from the launcher environment.

`manifest_store`, `manifest_path`, and `recursive` are accepted and used for snapshot resolution.

`constraints` is supported for runtime caps (`max_inflight_bytes`, `max_ram_bytes`).

`runtime` is accepted in the signature, but queue/want runtime overrides are currently rejected for `mx8.video`.

`node_id` is accepted and used as lock-owner identity during snapshot resolution.

`loader.checkpoint()` returns an opaque video token. Resume with:

```python
token = loader.checkpoint()
loader = mx8.video("s3://bucket/video_prefix/", clip_len=16, stride=8, fps=8, resume_from=token)
```

Each batch includes `clip_ids`, `sample_ids`, `media_uris`, `clip_starts`, `offsets`, and `payload`. Batch metadata fields include `frames_per_clip`, `frame_height`, `frame_width`, `channels`, `layout`, `dtype`, `colorspace`, and `strides`. Offsets are monotonic and `offsets[-1] == len(payload)`.
`batch.to_torch()` returns `(payload_u8, offsets_i64, sample_ids_i64)` where `payload_u8` has shape `B x T x H x W x C` and dtype `torch.uint8`.

The loader rejects invalid cap combinations at init — specifically when `batch_size_samples * bytes_per_clip > max_inflight_bytes`. Runtime autotune is enabled by default and adapts `max_inflight_bytes` within safe bounds. Pass `autotune=False` to disable.

The default decode backend uses a local `ffmpeg` CLI. Override the binary path with `MX8_FFMPEG_BIN`.

`MX8_VIDEO_DECODE_BACKEND` accepts `cli|auto|ffi|nvdec|nvidia` (`nvidia` is an alias of `nvdec`).
`MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT=1` enables an experimental device-output path for `batch.to_torch()`: when CUDA is available, `payload_u8` is returned as a Torch-owned CUDA tensor; otherwise it fails open to CPU tensor output.
`MX8_VIDEO_EXPERIMENTAL_DEVICE_DIRECT_WRITE=1` enables an experimental direct-write mode on top of device-output mode: MX8 first attempts `torch.ops.mx8_video.direct_write_u8(...)` (if registered; current native-op implementation uses stream-bound raw pointer copy when CUDA runtime is available), and fails open to staged-copy destination writing when unavailable.
`MX8_VIDEO_EXPERIMENTAL_DIRECT_DECODE_TO_DESTINATION=1` (requires direct-write mode + `nvdec|auto`) enables an internal decode-to-destination attempt: MX8 uses `torch.ops.mx8_video.decode_file_nvdec_into_u8(...)` to decode local media paths and stream decoded bytes directly into the Torch-owned CUDA destination tensor on the current stream, then fails open to the existing payload direct-write/staged-copy path if unsupported.
`MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY` can point to a shared library that registers `torch.ops.mx8_video.direct_write_u8` (loaded through `torch.ops.load_library(...)` on first use).
Build helper: `python crates/mx8-py/python/m8_build_video_direct_write_op.py --out-dir /tmp/mx8-direct-write-op`.
`MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT_ENFORCE_STREAM` (default `true`) enforces current-stream stability during this experimental write path and fails open to CPU tensor output on runtime stream/write errors.

Build-time flags:

- FFI: `RUSTFLAGS="--cfg mx8_video_ffi"`
- NVDEC: `RUSTFLAGS="--cfg mx8_video_nvdec"`

Current NVDEC path uses FFmpeg CUDA hwaccel flags (`-hwaccel cuda`). It requires an FFmpeg build with CUDA/NVIDIA decode support on the host.
When `nvdec`/`auto` is selected, video runtime autotune samples GPU memory pressure from `nvidia-smi` (override binary with `MX8_NVIDIA_SMI_BIN`).
Sampling is rate-limited to once every 2 seconds minimum and reuses the latest cached value between samples.

Fallback behavior is fail-open:

- `ffi` falls back to `cli` on any backend error.
- `nvdec` falls back to `ffi`, then `cli` on any backend error.
- `auto` tries `nvdec -> ffi -> cli`.

Each fallback emits a `video_decode_backend_fallback` proof event.

`loader.stats()` for the video loader includes decode contract fields (`video_layout`, `video_dtype`, `video_colorspace`, `video_frames_per_clip`, `video_frame_height`, `video_frame_width`, `video_channels`, `video_clip_bytes`), backend selection (`video_decode_backend`), fallback counters (`video_decode_backend_fallback_total`), decode counters (`video_decode_attempted_clips_total`, `video_decode_succeeded_clips_total`, `video_decode_failed_total`, `video_decode_ms_total`), device-output counters (`video_experimental_device_output_requested`, `video_experimental_device_output_active`, `video_experimental_device_output_fallback_total`), direct-write counters (`video_experimental_device_direct_write_requested`, `video_experimental_device_direct_write_active`, `video_experimental_device_direct_write_fallback_total`, `video_experimental_device_direct_write_batches_total`), decode-to-destination counters (`video_experimental_direct_decode_to_destination_requested`, `video_experimental_direct_decode_to_destination_active`, `video_experimental_direct_decode_to_destination_fallback_total`, `video_experimental_direct_decode_to_destination_batches_total`), GPU pressure counters (`video_gpu_pressure`, `video_gpu_pressure_unavailable_total`), and autotune counters (`video_runtime_autotune_enabled`, `video_runtime_autotune_pressure`, `video_runtime_autotune_adjustments_total`, `video_runtime_autotune_gpu_clamps_total`).

Gate commands for the video loader: `./scripts/video_stage2b_gate.sh`, `./scripts/video_stage2b_stress_gate.sh`, `./scripts/video_stage2c_perf_gate.sh`, `./scripts/video_stage3a_backend_gate.sh`, `./scripts/video_nvdec_fallback_gate.sh`, `./scripts/video_nvdec_compiled_fallback_gate.sh`, `./scripts/video_nvdec_pressure_gate.sh`, `./scripts/video_nvdec_throughput_gate.sh`, `./scripts/video_device_output_gate.sh`, `./scripts/video_direct_write_native_op_gate.sh`, `./scripts/video_direct_decode_to_destination_gate.sh`, `./scripts/video_direct_decode_stress_gate.sh`, and `./scripts/video_ga_gate.sh`. For strict hardware throughput enforcement, set `MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW=1`.


## Distributed loader

Distributed attach is built into `mx8.load`, `mx8.text`, `mx8.image`, and `mx8.audio` directly, and into `mx8.mix` via distributed source loaders. `mx8.video` supports deterministic rank/world sharding with the same attach arguments.

```python
import mx8

loader = mx8.load(
    "s3://bucket/prefix/",
    batch_size_samples=512,
    max_ram_gb=24,
    profile="balanced",
    job_id="train",
    cluster_url="http://coordinator-host:50051",
)
```

Equivalent env-based attach:

- `MX8_CLUSTER_URL=http://coordinator-host:50051` (or `MX8_COORD_URL`)
- `MX8_JOB_ID=train`

Then:

```python
loader = mx8.load("s3://bucket/prefix/", batch_size_samples=512, max_ram_gb=24)
```

`mx8.DistributedDataLoader` remains available as the low-level explicit API.

### Starting the coordinator

Use `mx8.coordinator()` to start a local coordinator subprocess without any manual setup. It finds the `mx8d-coordinator` binary in your PATH, starts it, waits for it to be ready, and tears it down when the context exits.

In distributed mode, the coordinator owns snapshot resolution for the job. Start the coordinator with the same `dataset_link` intent you expect workers to consume.

**Single-machine multi-GPU (`torchrun`)**

Pass `rank` so that only rank 0 starts the coordinator. All other ranks wait for it automatically — no manual barrier or broadcast needed.

```python
# train.py — every rank runs the same script
import os, mx8

rank       = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

with mx8.coordinator(rank=rank, world_size=world_size) as coord:
    loader = mx8.DistributedDataLoader(
        coord_url=coord.url,
        job_id="train",
        node_id=f"rank{rank}",
        batch_size_samples=512,
        max_ram_gb=24,
        profile="balanced",
    )
    for batch in loader:
        payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
```

Launch with: `torchrun --nproc_per_node=8 train.py`

**Multi-node**

Set `bind_host="0.0.0.0"` and `master_addr` to the coordinator machine's reachable hostname or IP. Every rank gets a `coord.url` that points at the coordinator's public address.

```python
with mx8.coordinator(
    rank=rank,
    world_size=world_size,
    bind_host="0.0.0.0",
    master_addr="node0.cluster.local",
    port=50051,
) as coord:
    loader = mx8.DistributedDataLoader(coord_url=coord.url, ...)
```

**Single process (no torchrun)**

Omit `rank` and a free port is chosen automatically:

```python
with mx8.coordinator(world_size=1) as coord:
    loader = mx8.DistributedDataLoader(coord_url=coord.url, ...)
```

`coordinator()` parameters:

`rank` — current process rank. When set, only rank 0 starts the subprocess; all other ranks wait for it to become ready. When omitted the function always starts the coordinator (single-process mode).

`world_size` — number of ranks expected in this job (default 1).

`dataset_link` — optional dataset link passed to the coordinator for snapshot resolution.

`port` — port to bind on. Defaults to 50051 when `rank` is set (all ranks must agree); auto-picked when `rank` is omitted.

`bind_host` — host the coordinator binds on (default `"127.0.0.1"`). Set to `"0.0.0.0"` for multi-node.

`master_addr` — address other ranks use to reach the coordinator. Defaults to `bind_host`. Set this to the coordinator machine's public hostname or IP for multi-node jobs.

`timeout_secs` — how long every rank waits for the coordinator to become ready (default 30).

`log` — forward the coordinator's output to stderr. Default `False`. Use `log=True` when debugging startup failures.

`coord.stop()` terminates the subprocess immediately. Called automatically on context exit and garbage collection.

### DistributedDataLoader

```python
loader = mx8.DistributedDataLoader(
    coord_url="http://127.0.0.1:50051",
    job_id="train",
    node_id=f"rank{rank}",
    batch_size_samples=512,
    max_ram_gb=24,
    profile="balanced",
)
```

`want` sets the max number of concurrent leases this node will request. `progress_interval_ms` controls how often progress is reported to the coordinator (default 500ms). `grpc_max_message_bytes` caps the gRPC message size for manifest and control-path traffic (default 64MB).

`max_inflight_bytes`, `max_queue_batches`, and `prefetch_batches` are accepted for compatibility, but effective values are currently derived from `profile`, `constraints`, and `runtime`.

`resume_from` (default `None`) accepts an opaque distributed checkpoint token from `DistributedDataLoader.checkpoint()`. The token is validated and applied by the coordinator before lease issuance. Every rank should pass the same token on restart.

Distributed resume rules:

`resume_from` must be produced by `DistributedDataLoader.checkpoint()` from the same `manifest_hash` and `epoch`.

All ranks must pass the same token content. Conflicting tokens are rejected.

`resume_from` is a startup input only. If the run has already issued leases, the coordinator rejects late resume tokens.

The token is intentionally opaque and versioned by MX8. Treat it as a byte blob and store it alongside the model checkpoint.

Distributed autotune adjusts `want`, `prefetch_batches`, and `max_queue_batches` within the chosen profile rails. Pass `profile` and `autotune=True|False` to control it.

Training note: `mx8==1.0.5` training supports epoch-boundary elasticity only. Mid-epoch rank loss is still non-elastic (the DDP process group fails and the job restarts). Add/remove nodes between epochs by launching the next epoch with a new world size.

Distributed checkpoint example:

```python
loader = mx8.DistributedDataLoader(coord_url=coord.url, job_id="train", node_id=f"rank{rank}")
token = loader.checkpoint()
# save token with model checkpoint

loader = mx8.DistributedDataLoader(
    coord_url=coord.url,
    job_id="train",
    node_id=f"rank{rank}",
    resume_from=token,
)
```

Gate commands:

`./scripts/distributed_resume_gate.sh` (same-epoch restart from checkpoint token)

`./scripts/training_epoch_boundary_gate.sh` (epoch-boundary add/remove membership)


## Mix API

`mx8.mix` blends multiple loaders deterministically under one shared memory envelope. Use it when you want to train on a weighted combination of datasets without managing the interleaving yourself.

```python
import mx8

loader_a = mx8.load("s3://bucket/a@refresh", batch_size_samples=32, max_ram_gb=6)
loader_b = mx8.load("s3://bucket/b@refresh", batch_size_samples=32, max_ram_gb=6)

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

`job_id` and `cluster_url` are accepted in the `mx8.mix` signature for API consistency, but are currently ignored by the implementation.

`mx8.mix` accepts loaders created by `mx8.load(...)` and `mx8.DistributedDataLoader(...)`.

`mixed.checkpoint()` returns an opaque token compatible with `mx8.mix(..., resume_from=token)`.

Resume contract for mix:

- `seed`, `epoch`, and source count must match.
- The mix token restores scheduler position and per-source delivery counters.
- If source checkpoints differ, MX8 continues in best-effort mode and increments `mix_resume_source_checkpoint_mismatch_total` in `mixed.stats()`.

For a fixed set of manifests, weights, seed, epoch, and frozen membership, the mixed stream order is deterministic and replayable. All sources share one inflight cap — backpressure is global. `mixed.stats()["mix_sources"]` exposes per-source diagnostics including manifest IDs, delivery counters, and configured knobs.

Gate commands: `./scripts/mix_gate.sh`, `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`, `./scripts/mix_multirank_gate.sh`.
