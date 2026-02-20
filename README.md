# MX8 Runtime (v0)

MX8 is a bounded, high-performance Rust data runtime (with Python bindings) plus a tiny per-job coordinator/agent layer for multi-node correctness.

It is optimized for inference/ETL/preprocessing today, with training support under v0 non-elastic semantics.

## What Works Today (Front-page status)

- **Zero-manifest path is live:** `mx8d-agent` runs direct-stream manifest ingest for lease execution.
- **Distributed lease flow is live:** membership barrier, lease grant/progress, expiry, and requeue/reassignment are exercised by demos/gates.
- **Bounded memory runtime is live:** inflight byte caps + queue backpressure are enforced.
- **Pinned snapshot semantics are live:** `plain`, `@refresh`, and `@sha256:` resolve to `manifest_hash`.
- **S3-compatible path is live:** MinIO gates verify resolver + runtime behavior.
- **Python path is live:** PyO3 package + smoke scripts are available.
- **Autotune preview is live:** hybrid AIMD + PID-like control adjusts `want`, `prefetch_batches`, and `max_queue_batches` within profile rails (`safe|balanced|throughput`).

## Quickstart (current API)

```python
import mx8

loader = mx8.load(
    "s3://bucket/prefix/@refresh",
    recursive=True,
    profile="balanced",
    autotune=True,
)

for batch in loader:
    # consume batch.payload / batch.offsets / batch.sample_ids
    pass
```

## Current v0 Constraints

- Training is **non-elastic** in v0: rank/node failure will terminate DDP jobs.
- Cursor semantics are at-least-once at boundaries for inference/ETL (no exactly-once guarantee in v0).
- Refresh is job-start only (no mid-run pickup).

## Quick Verify Commands

- **Main smoke gate:** `./scripts/smoke.sh`
- **Zero-manifest accelerated burn-in:** `MX8_BURNIN_RUNS=3 ./scripts/direct_stream_burnin.sh`
- **Autotune A/B DDP gate:** `MX8_TORCH_DDP_AUTOTUNE_AB=1 ./scripts/torch_ddp_gate.sh`
- **Autotune pressure simulation:** `./scripts/autotune_memory_pressure_sim.sh`
- **Python smoke:** `./scripts/py_smoke.sh`
- **Wheel + pip smoke:** `./scripts/build_wheel.sh && ./scripts/pip_wheel_smoke.sh`
- **MinIO/S3-compatible gates:** `MX8_SMOKE_MINIO=1 ./scripts/smoke.sh`

## Key Capability Areas

### Bounded Memory Runtime

MX8 is designed to be hard-capped by config (backpressure via inflight permits).

Need explicit memory rails? Use:

```python
loader = mx8.load(
    "s3://bucket/prefix/@refresh",
    profile="balanced",
    autotune=True,
    constraints=mx8.Constraints(
        max_inflight_bytes=256 * 1024 * 1024,
        max_process_rss_bytes=24 * 1024 * 1024 * 1024,
    ),
)
```

Note: `max_inflight_bytes` caps loader-path memory. Total process RSS includes model/framework/user allocations.

### Autotune (AIMD + PID-like rails)

Autotune is available as a preview path in v0 and is exposed through the Python loader API and distributed gate scripts.

- Profiles: `safe`, `balanced`, `throughput`
- Controller intent: increase throughput while staying inside explicit memory/safety constraints
- Runtime knobs adjusted by autotune:
  - `want` (distributed lease demand)
  - `prefetch_batches`
  - `max_queue_batches`
- Env controls (distributed path):
  - `MX8_AUTOTUNE=1`
  - `MX8_AUTOTUNE_PROFILE=safe|balanced|throughput` (default `balanced`)

### Packing (Many Small Objects)

For many-small-object S3 datasets, pack once into tar shards + byte-range manifest:

- Input: `s3://bucket/raw/train/`
- Output: `s3://bucket/mx8/train/`
  - `shards/shard-00000.tar`
  - `_mx8/manifest.tsv`
  - optional `_mx8/labels.tsv`

CLI example:

- `MX8_PACK_IN=s3://bucket/raw/train/ MX8_PACK_OUT=s3://bucket/mx8/train/ cargo run -p mx8-snapshot --features s3 --bin mx8-pack-s3`

Python example:

- `python -c "import mx8; mx8.pack('s3://bucket/raw/train/', out='s3://bucket/mx8/train/', shard_mb=512, label_mode='auto')"`

### Vision (ImageFolder -> Torch tensors)

```python
import mx8

loader = mx8.vision.ImageFolderLoader(
    "s3://bucket/mx8/train/@refresh",
    batch_size_samples=64,
    resize_hw=(224, 224),
)

for images, labels in loader:
    # images: [B,C,H,W] float32 in [0,1]
    # labels: [B] int64
    pass
```

### Video “Virtual Seek” Status (Shipped vs Target)

**What ships today**

- Clip-oriented video API and decode contract gates are live (v1.8 Stage 2A/2B).
- Deterministic Stage 2D range-planning contract is live (`video_range_schema_version=1` + planner gate).
- Current decode path is CPU (`ffmpeg` CLI), with bounded runtime caps.

**What is next (Virtual Seek target)**

- Execute S3 `Range` GETs directly from Stage 2D plans.
- Decode clips from fetched segments without full-object temp download on supported datasets.
- Emit proof logs/counters for plan/fetch/fallback so no-download claims are measurable.

Determinism contract for this path is pinned by `manifest_hash + sidecar schema/hash + sampling seed`.

## Current Docs

- Python API: `docs/python_api.md`
- Vision labels/layouts: `docs/vision_labels.md`
- S3/runtime tuning: `docs/s3_runtime_tuning.md`
- Memory contract: `docs/memory_contract.md`
- Troubleshooting: `docs/troubleshooting.md`
- AI agent guide: `docs/ai_agent_guide.md`
- AI agent context: `docs/ai_agent_context.json`

## Roadmap Next (after v1.6)

- **v1.7 `mx8.mix`:** deterministic weighted mixing across multiple loaders with shared global memory/backpressure caps.
- **v1.8 video-native path:** clip-level indexing/decode hardening on top of current bounded runtime.
