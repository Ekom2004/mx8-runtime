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
    batch=512,
    ram_gb=12,
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
loader = mx8.load("s3://bucket/data/", batch=512, ram_gb=24)

# image
loader = mx8.image(
    "s3://bucket/images/",
    batch=64,
    resize=(256, 256),
    augment="imagenet",
    seed=17,
    epoch=0,
    ram_gb=24,
)

# text
loader = mx8.text("s3://bucket/corpus/", tokenizer="gpt2", seq_len=2048, batch=32, ram_gb=24)

# audio
loader = mx8.audio(
    "s3://bucket/audio/",
    batch=32,
    samples=16000,
    channels=1,
    rate_hz=16000,
    on_decode_error="error",
    ram_gb=24,
)

# video
loader = mx8.video("s3://bucket/videos/", clip=16, stride=8, fps=8, batch=32, ram_gb=24)

# mix
a = mx8.load("s3://bucket/a/", batch=32, ram_gb=12)
b = mx8.load("s3://bucket/b/", batch=32, ram_gb=12)
loader = mx8.mix([a, b], weights=[0.7, 0.3], seed=17, epoch=0, ram_gb=24)

# distributed attach
loader = mx8.load(
    "s3://bucket/data/",
    batch=512,
    ram_gb=24,
    job="train-001",
    coord="http://coordinator-host:50051",
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
| [CLI Reference](docs/cli_reference.md) | `mx8-pack-s3`, `mx8-snapshot-resolve` |

## Development

```bash
cargo build --workspace
cargo test --workspace
./scripts/prod_readiness.sh
./scripts/azure_gate.sh
```
