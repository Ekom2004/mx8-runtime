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
  - `MX8_SMOKE_MINIO_PACK=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_PY_VISION_PILLOW=1 ./scripts/smoke.sh` (heavier; installs torch+pillow in a temp venv)
- PyTorch DDP gates (local multi-process):
  - `MX8_SMOKE_TORCH_DDP=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_NODUPES=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_DETERMINISM=1 ./scripts/smoke.sh`
  - `MX8_SMOKE_TORCH_DDP_RESTART=1 ./scripts/smoke.sh`

## Packing (v0)

If your dataset is “many small S3 objects” (e.g. ImageFolder layout with millions of files), run `mx8-pack-s3` once to create tar shards plus a byte-range manifest:

- Input: `s3://bucket/raw/train/`
- Output: `s3://bucket/mx8/train/`
  - shards: `s3://bucket/mx8/train/shards/shard-00000.tar`
  - manifest: `s3://bucket/mx8/train/_mx8/manifest.tsv`
  - labels (optional): `s3://bucket/mx8/train/_mx8/labels.tsv` (if ImageFolder labels are enabled)

You can run the packer either via the CLI:

- `MX8_PACK_IN=s3://bucket/raw/train/ MX8_PACK_OUT=s3://bucket/mx8/train/ cargo run -p mx8-snapshot --features s3 --bin mx8-pack-s3`

Or via the Python API (after installing `mx8`):

- `python -c "import mx8; mx8.pack('s3://bucket/raw/train/', out='s3://bucket/mx8/train/', shard_mb=512, label_mode='auto')"`

Then point MX8 at the packed prefix (snapshot resolver will use the precomputed manifest, avoiding large LIST operations).

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
