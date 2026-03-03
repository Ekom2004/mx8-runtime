# MX8 User Guide

This guide is for day-to-day MX8 usage with the simple API surfaces:

- `mx8.load`
- `mx8.image`
- `mx8.video`
- `mx8.mix`
- `mx8.stats`

Use this for practical operation. Use `docs/python_api.md` for the full parameter reference.

## Daily Workflow: Training

### 1) Start a run

```python
import mx8

loader = mx8.load(
    "s3://bucket/train@refresh",
    batch_size_samples=512,
    max_ram_gb=24,
    profile="balanced",
)
```

### 2) Train and monitor

```python
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 100 == 0:
        print(mx8.stats(loader))
```

### 3) Save checkpoints (model + MX8 token)

```python
token = loader.checkpoint()
torch.save({"model": model.state_dict(), "mx8": token}, "ckpt.pt")
```

### 4) Resume after restart

```python
ckpt = torch.load("ckpt.pt")
model.load_state_dict(ckpt["model"])

loader = mx8.load(
    "s3://bucket/train@refresh",
    batch_size_samples=512,
    max_ram_gb=24,
    resume_from=ckpt["mx8"],
)
```

### 5) Distributed training attach

```python
loader = mx8.load(
    "s3://bucket/train@refresh",
    batch_size_samples=512,
    max_ram_gb=24,
    job_id="train-001",
    cluster_url="http://coordinator-host:50051",
)
```

Notes:

- Training is epoch-boundary elastic in v1: add/remove nodes between epochs.
- Mid-epoch rank loss still requires restart + resume.
- On distributed resume, pass the same token content to all ranks.

## Daily Workflow: Inference / ETL

### 1) Start a run

```python
import mx8

loader = mx8.load(
    "s3://bucket/corpus/",
    batch_size_samples=1024,
    max_ram_gb=24,
    profile="throughput",
)
```

### 2) Process and poll health

```python
for step, batch in enumerate(loader):
    process(batch)
    if step % 200 == 0:
        print(mx8.stats(loader))
```

### 3) Resume long jobs

```python
token = loader.checkpoint()

loader = mx8.load(
    "s3://bucket/corpus/",
    batch_size_samples=1024,
    max_ram_gb=24,
    resume_from=token,
)
```

Inference/ETL is the strongest recovery path in MX8: lease reassignment plus checkpoint/resume.

## Vision and Video

### Image

```python
loader = mx8.image(
    "s3://bucket/images/",
    batch_size_samples=64,
    resize_hw=(224, 224),
    max_ram_gb=24,
)
```

### Video

```python
loader = mx8.video(
    "s3://bucket/videos/",
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    max_ram_gb=24,
)
```

Both support `checkpoint()` and `resume_from=...`.

## Mix

```python
a = mx8.load("s3://bucket/a/", batch_size_samples=32, max_ram_gb=12)
b = mx8.load("s3://bucket/b/", batch_size_samples=32, max_ram_gb=12)

mixed = mx8.mix(
    [a, b],
    weights=[0.7, 0.3],
    seed=17,
    epoch=0,
    max_ram_gb=24,
)
```

Resume behavior:

- `mixed.checkpoint()` returns a mix token.
- On resume, if source checkpoints differ from token snapshots, mix continues in best-effort mode.
- Inspect `mixed.stats()["mix_resume_source_checkpoint_mismatch_total"]`.

## Operational Shortcuts

- Human snapshot: `mx8.stats(loader)`
- Raw counters: `mx8.stats(loader, raw=True)`
- Full incident response steps: `docs/prod_runbook.md`
- Distributed setup details: `docs/deployment_guide.md`
