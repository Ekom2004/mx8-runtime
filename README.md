<div align="center">

<img src="docs/assets/mx8-banner.png" alt="MX8" width="600" />

# MX8

**The data runtime for serious ML workloads.**

Point to your data. Set a RAM limit. Train.

[![PyPI version](https://img.shields.io/pypi/v/mx8?color=blue)](https://pypi.org/project/mx8/)
[![Python](https://img.shields.io/pypi/pyversions/mx8)](https://pypi.org/project/mx8/)

</div>

---

MX8 is a high-performance Rust data runtime exposed to Python, with a small coordinator/agent control plane for multi-node execution.

It is built for large-scale inference, ETL, video preprocessing, and distributed training workloads.

## Quickstart

```python
import mx8

loader = mx8.load(
    "s3://your-bucket/dataset/",
    batch_size_samples=512,
    max_ram_gb=12,
    profile="balanced",
)

for batch in loader:
    train_step(batch)
```

## Why MX8

- Bounded memory with fail-fast RSS caps.
- Zero-manifest loading from raw S3 prefixes.
- Deterministic replay via pinned `manifest_hash`.
- Recovery-oriented distributed lease model.
- Text, image, audio, video, and mix loaders under one API family.

## API At A Glance

```python
import mx8

# bytes
loader = mx8.load("s3://bucket/data/", batch_size_samples=512, max_ram_gb=24)

# image
loader = mx8.image(
    "s3://bucket/images/",
    batch_size_samples=64,
    resize_hw=(256, 256),
    augment="imagenet",
    seed=17,
    epoch=0,
    max_ram_gb=24,
)

# text
loader = mx8.text("s3://bucket/corpus/", tokenizer="gpt2", sequence_length=2048, batch_size_samples=32, max_ram_gb=24)

# audio
loader = mx8.audio(
    "s3://bucket/audio/",
    batch_size_samples=32,
    sample_count=16000,
    channels=1,
    sample_rate_hz=16000,
    decode_error_policy="error",
    max_ram_gb=24,
)

# video
loader = mx8.video("s3://bucket/videos/", clip_len=16, stride=8, fps=8, batch_size_samples=32, max_ram_gb=24)

# mix
a = mx8.load("s3://bucket/a/", batch_size_samples=32, max_ram_gb=12)
b = mx8.load("s3://bucket/b/", batch_size_samples=32, max_ram_gb=12)
loader = mx8.mix([a, b], weights=[0.7, 0.3], seed=17, epoch=0, max_ram_gb=24)

# distributed attach
loader = mx8.load(
    "s3://bucket/data/",
    batch_size_samples=512,
    max_ram_gb=24,
    job_id="train-001",
    cluster_url="http://coordinator-host:50051",
)

# stats + checkpoint
print(mx8.stats(loader))
token = loader.checkpoint()
```

## Install

```bash
pip install mx8
```

For vision/training helpers:

```bash
pip install mx8 pillow numpy torch
```

## Documentation

| Doc | Description |
| --- | --- |
| [User Guide](docs/user_guide.md) | Day-to-day training and inference workflows |
| [Python API](docs/python_api.md) | Full API reference |
| [Deployment Guide](docs/deployment_guide.md) | Coordinator/agent deployment |
| [Zero-Manifest Loading](docs/zero_manifest.md) | Raw S3 loading path |
| [Memory Contract](docs/memory_contract.md) | Memory and RSS guarantees |
| [Autotune](docs/v1_autotune_api_contract.md) | Startup/runtime autotune contract |
| [Mix API](docs/mix_v17_contract.md) | Multi-dataset blending |
| [Production Runbook](docs/prod_runbook.md) | Incident handling |
| [Design Partner Onboarding](docs/design_partner_onboarding_checklist.md) | Partner launch readiness checklist |
| [CLI Reference](docs/cli_reference.md) | `mx8-pack-s3`, `mx8-snapshot-resolve` |

## Development

```bash
cargo build --workspace
cargo test --workspace
./scripts/prod_readiness.sh
./scripts/azure_gate.sh
```
