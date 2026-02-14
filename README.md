# MX8 Runtime (v0)

MX8 is a high-performance Rust in-process data runtime (exposed to Python) plus a tiny per-job coordinator/agent layer for multi-node correctness and pacing.

## Quick gates

- Repo smoke (offline + online sub-gates):
  - `./scripts/smoke.sh`
- Python smoke:
  - `./scripts/py_smoke.sh`
  - `MX8_PY_SMOKE_INSTALL_TORCH=1 ./scripts/py_smoke.sh`
- MinIO (S3-compatible) gates:
  - `MX8_SMOKE_MINIO=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_DEMO2_MINIO_SCALE=1 MX8_SMOKE_MINIO_MANIFEST_STORE=1 ./scripts/smoke.sh`
- PyTorch DDP gates (local multi-process):
  - `MX8_SMOKE_TORCH_DDP=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_NODUPES=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_DETERMINISM=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_RESTART=1 ./scripts/smoke.sh`

## Training semantics (v0)

### Non-elastic
MX8 v0 does **not** keep PyTorch DDP alive after rank/node failure.

- If a rank dies, DDP will terminate the job.
- Lease reassignment is for inference/ETL correctness (keep the job draining), not for elastic training.

### Determinism
For a pinned snapshot (`manifest_hash`) and frozen membership (`world_size` barrier), sharding/shuffle is deterministic by:

- `MX8_SEED` (default: `0`)
- `MX8_EPOCH` (default: `0`)
- `MX8_SHUFFLE` (default: `false`; accepts `true/false/1/0`; set to `true` for training-style block shuffles)

### Checkpointing & resume (epoch-level)
MX8 v0 supports **epoch-level resume**.

On preemption or crash:

- Save your model checkpoint regularly (every N steps).
- Restart the job; MX8 restarts from the beginning of the current epoch.
- Data order is deterministic for the same `(manifest_hash, seed, epoch, world_size)` and frozen membership.
- Some data may be re-processed; training continues correctly.

Mid-epoch resume (dataloader `state_dict()`) is planned for v1.
