# MX8 Runtime

MX8 is a bounded Rust data runtime with Python bindings for ML data loading at S3 scale.

It focuses on one hard contract: keep data delivery fast, deterministic, and memory-bounded across single-node and distributed jobs.

## Why MX8

- **Bounded memory by design:** hard inflight caps + queue backpressure.
- **Deterministic snapshots:** dataset links resolve to pinned `manifest_hash`.
- **Distributed correctness rails:** lease grant/progress/expiry/requeue with proof logs.
- **S3-native workflows:** MinIO/S3-compatible gates and packing path for many-small-object datasets.
- **Simple Python surface:** `pip install mx8` and start with `mx8.load(...)`.

## Shipped Today (Main Branch)

- `mx8.load(...)` byte-oriented runtime for ETL/inference/preprocess.
- `mx8.DistributedDataLoader` for multi-rank data delivery.
- Snapshot resolver (`plain`, `@refresh`, `@sha256:`).
- Zero-manifest direct-stream ingest path in `mx8d-agent`.
- Autotune preview (`safe|balanced|throughput`) for `want`, `prefetch_batches`, `max_queue_batches`.
- Vision loader (`mx8.vision.ImageFolderLoader`) + packers (`mx8.pack`, `mx8.pack_dir`).
- Video loader (`mx8.video`) with locked decode/delivery contracts.
- Video Stage 3A backend rails: `MX8_VIDEO_DECODE_BACKEND=cli|ffi` with fallback proof log (`video_decode_backend_fallback`).

## 60-Second Quickstart

```python
import mx8

loader = mx8.load(
    "s3://bucket/prefix/@refresh",
    recursive=True,
    profile="balanced",
    autotune=True,
    constraints=mx8.Constraints(
        max_inflight_bytes=256 * 1024 * 1024,
    ),
)

for batch in loader:
    payload = batch.payload
    offsets = batch.offsets
    sample_ids = batch.sample_ids
```

Install:

- `pip install mx8`
- For vision/training helpers: `pip install pillow numpy torch`

## v0 Boundaries (Explicit)

- Training is **non-elastic** in v0 (rank/node loss ends DDP training run).
- Inference/ETL cursor semantics are at-least-once at boundaries (not exactly-once).
- `@refresh` resolution is job-start scoped (no mid-run dataset refresh).

## Operational Gates

- Main smoke: `./scripts/smoke.sh`
- Python smoke: `./scripts/py_smoke.sh`
- Wheel smoke: `./scripts/build_wheel.sh && ./scripts/pip_wheel_smoke.sh`
- Zero-manifest burn-in: `MX8_BURNIN_RUNS=3 ./scripts/direct_stream_burnin.sh`
- Mix gate: `./scripts/mix_gate.sh`
- Mix gate strict (CI profile): `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`
- Mix gate via smoke toggle: `MX8_SMOKE_MIX=1 ./scripts/smoke.sh`
- DDP autotune A/B: `MX8_TORCH_DDP_AUTOTUNE_AB=1 ./scripts/torch_ddp_gate.sh`
- Autotune pressure simulation: `./scripts/autotune_memory_pressure_sim.sh`
- Video Stage 3A backend parity: `./scripts/video_stage3a_backend_gate.sh`
- Video GA checklist (fast/full): `./scripts/video_ga_gate.sh --quick|--full`

## Video (GA)

- `mx8.video(...)` is GA for the current clip decode/delivery contract.
- Default decode backend is `ffmpeg` CLI; optional backend selector rails are available.
- S3 range-streaming optimization is still roadmap work, not required for current GA contract.

## Docs Map

- Product architecture: `ARCHITECTURE.MD`
- Python API reference: `docs/python_api.md`
- `mx8.mix` v1.7 contract draft: `docs/mix_v17_contract.md`
- `mx8.mix` gate runbook: `docs/mix_gate_runbook.md`
- Memory model/caps: `docs/memory_contract.md`
- S3/runtime tuning: `docs/s3_runtime_tuning.md`
- Troubleshooting: `docs/troubleshooting.md`
- Video GA checklist: `docs/video_ga_checklist.md`

## Near-Term Roadmap

- **Dataset mixing (`mx8.mix`)** for multi-source training/inference pipelines (contract drafted; implementation next).
- **Video-native hardening** for range-streaming and decode reliability at scale.
