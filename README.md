<div align="center">

<img src="docs/assets/mx8-banner.png" alt="MX8" width="600" />

# MX8

**The data runtime for serious ML workloads.**

Point to your data. Set a RAM limit. Train. That's it.

[![PyPI version](https://img.shields.io/pypi/v/mx8?color=blue)](https://pypi.org/project/mx8/)
[![Python](https://img.shields.io/pypi/pyversions/mx8)](https://pypi.org/project/mx8/)
[![Smoke](https://img.shields.io/badge/gates-passing-brightgreen)](#gates)

</div>

---

MX8 is a high-performance Rust data runtime exposed to Python, plus a small per-job coordinator and agent layer that prevents multi-node stampedes, enforces memory budgets, and recovers from node failures automatically.

It is built for the workloads that break other dataloaders: large-scale inference, ETL, video preprocessing, and distributed training on fleets where nodes die.

```python
import mx8

loader = mx8.load(
    "s3://your-bucket/dataset/",
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    process(batch)
```

No manifest required. No setup. MX8 scans the prefix, shuffles, caps memory, and starts delivering.

---

## Why MX8

Most data loading tools hand you the primitives and leave the hard problems to you. MX8 solves them.

**Guaranteed bounded loader memory + fail-fast RSS.** Every loader surface has hard inflight backpressure caps plus a whole-process RSS fail-fast cap. Set `max_ram_gb=12` and MX8 aborts explicitly if RSS exceeds the cap, before a silent OS OOM kill.

**Zero-manifest loading.** Point MX8 at any raw S3 prefix and it starts delivering bounded, shuffled batches immediately. No packing step, no index generation, no prep. When you are ready for production, pack once and MX8 switches to the fast path automatically.

**Kill-and-recover.** When a node dies mid-job, the coordinator detects the stale heartbeat, expires the lease, and requeues the unfinished range for another node to pick up. Inference and ETL jobs drain to completion with no manual intervention.

**Deterministic replay.** Every run is pinned to a `manifest_hash`. Same hash, same seed, same epoch, same membership — bit-identical data order. Reproduce any training run, audit any inference job.

**S3 surgical range-seek for video.** MX8 does not download whole video files. It plans byte ranges into compressed video, seeks to the right GOP boundaries, and decodes only the frames you need — under the same hard memory envelope.

**Multi-dataset mixing under one cap.** `mx8.mix` blends multiple datasets with configurable weights, deterministic scheduling, and a single shared memory budget. No source can starve another. No source can blow the cap.

---

## Install

```bash
pip install mx8
```

For vision and training:

```bash
pip install mx8 pillow numpy torch
```

---

## Quickstart

**Load from any S3 prefix (no prep required):**

```python
import mx8

loader = mx8.load(
    "s3://bucket/raw-images/",
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    process(batch.payload, batch.offsets)
```

**Image classification with PyTorch:**

```python
import mx8

mx8.pack(
    "s3://bucket/raw/train/",
    out="s3://bucket/mx8/train/",
    label_mode="imagefolder",
)

loader = mx8.image(
    "s3://bucket/mx8/train/@refresh",
    batch_size_samples=64,
    resize_hw=(224, 224),
    max_ram_gb=24,
)

for images, labels in loader:
    # images: [B, C, H, W] float32
    # labels: [B] int64
    train_step(images, labels)
```

**Video clips with bounded decode:**

```python
import mx8

loader = mx8.video(
    "s3://bucket/videos/",
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    max_ram_gb=24,
)

for batch in loader:
    process_clips(batch.payload, batch.clip_ids)
```

**Multi-dataset mixing:**

```python
import mx8

a = mx8.load("s3://bucket/dataset-a@refresh", batch_size_samples=32)
b = mx8.load("s3://bucket/dataset-b@refresh", batch_size_samples=32)

mixed = mx8.mix([a, b], weights=[0.7, 0.3], seed=17, epoch=0, max_ram_gb=24)

for batch in mixed:
    train_step(batch)
```

**Distributed multi-node:**

```python
import mx8

loader = mx8.DistributedDataLoader(
    coord_url="http://coordinator-host:50051",
    job_id="my-job",
    node_id="node-0",
    batch_size_samples=512,
    max_ram_gb=48,
    profile="throughput",
)

for batch in loader:
    payload_u8, offsets, sample_ids = batch.to_torch()
    train_step(payload_u8)
```

---

## Architecture

MX8 has three components.

The **runtime** (`mx8`) runs in-process inside your Python training or inference script. It owns the data path: Fetch → Decode/Parse → Pack → Deliver. All pipeline stages are bounded. Backpressure prevents runaway prefetch. The RSS watchdog prevents silent OOM kills.

The **coordinator** (`mx8-coordinator`) runs once per job. It resolves the dataset to a pinned snapshot, freezes initial membership, supports bounded post-freeze replacement/scale-out up to configured capacity, assigns leases to nodes, expires stale leases, and requeues unfinished ranges. It is the reason multi-node MX8 behaves like one coordinated consumer instead of many independent loaders.

The **agent** (`mx8d-agent`) runs once per node. It enforces per-node memory budgets, requests leases from the coordinator, and feeds work ranges to local runtime processes.

```
 Python process          Node (mx8d-agent)          Job (mx8-coordinator)
 + mx8 runtime    <——>   local caps/budgets   <——>   leases/ownership
 bounded pipeline         feeds WorkRanges            resolves snapshots
       |
       v
 Dataset Storage (S3 / NVMe / filesystem)
```

---

## Documentation

| Doc | Description |
| --- | --- |
| [Python API](docs/python_api.md) | Full loader API reference |
| [Deployment Guide](docs/deployment_guide.md) | Running coordinator and agents in production |
| [Zero-Manifest Loading](docs/zero_manifest.md) | Loading from raw S3 prefixes without packing |
| [Memory Contract](docs/memory_contract.md) | What MX8 guarantees about memory |
| [Autotune](docs/v1_autotune_api_contract.md) | How startup and runtime autotune work |
| [Mix API](docs/mix_v17_contract.md) | Multi-dataset blending |
| [Operator TUI](docs/tui.md) | Live job monitoring in the terminal |
| [Production Runbook](docs/prod_runbook.md) | Incident response and escalation |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and fixes |
| [Security Model](docs/security_model.md) | Auth, IAM, and network hardening |
| [CLI Reference](docs/cli_reference.md) | mx8-pack-s3, mx8-snapshot-resolve |
| [gRPC Contract](docs/grpc_contract.md) | Coordinator/agent wire protocol |
| [Env Reference](docs/env_reference.md) | All MX8_* environment variables |
| [HA Contract](docs/ha_contract.md) | Coordinator HA plan (v1.9) |
| [Compatibility Policy](docs/compatibility_policy.md) | Versioning and stability guarantees |

---

## Gates

MX8 ships with a deterministic gate suite. Run the full readiness check before any production deployment:

```bash
./scripts/prod_readiness.sh
```

Individual gates:

```bash
./scripts/smoke.sh                          # offline + online correctness
./scripts/py_smoke.sh                       # Python API smoke
./scripts/mix_gate.sh                       # mix determinism + ratio + memory
./scripts/mix_multirank_gate.sh             # multi-rank no-overlap
./scripts/tui_gate.sh                       # TUI headless probe
MX8_SMOKE_MINIO=1 ./scripts/smoke.sh       # S3-compatible (MinIO) gates
MX8_SMOKE_TORCH_DDP=1 ./scripts/smoke.sh   # PyTorch DDP multi-rank
```

---

## Building from source

Prerequisites: Rust toolchain, `cargo`.

```bash
git clone https://github.com/your-org/mx8-runtime
cd mx8-runtime
cargo build --workspace
cargo test --workspace
```

To build the Python wheel:

```bash
./scripts/build_wheel.sh
pip install dist/mx8-*.whl
```

---

## Status

| Capability | Status |
| --- | --- |
| Bounded in-process pipeline | GA |
| Zero-manifest S3 loading | GA |
| Image loader (PyTorch) | GA |
| Video loader (CPU decode) | GA |
| Video loader (GPU/NVDEC) | Planned |
| Multi-dataset mix | GA |
| Distributed coordinator + agent | GA |
| Coordinator HA (automatic failover) | v1.9 planned |
| Mid-epoch resume (`state_dict`) | v1.9 planned |
| Prometheus/OTEL metrics export | Planned |
