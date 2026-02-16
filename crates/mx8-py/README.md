# mx8 (Python)

MX8 is a bounded-memory data runtime exposed to Python (built with PyO3 + maturin).

The v0 focus is “don’t OOM”: MX8 enforces backpressure with hard caps (so prefetch can’t runaway).

Further docs:

- Python API: `../../docs/python_api.md`
- Vision labels/layout: `../../docs/vision_labels.md`
- S3/runtime tuning: `../../docs/s3_runtime_tuning.md`
- Memory contract: `../../docs/memory_contract.md`
- AI agent guide: `../../docs/ai_agent_guide.md`
- AI agent context (JSON): `../../docs/ai_agent_context.json`
- Troubleshooting: `../../docs/troubleshooting.md`

## Install (from wheel)

Once you have a wheel (from CI or local build):

- `python -m venv .venv && . .venv/bin/activate`
- `pip install mx8-*.whl`

## Install (from PyPI)

- `python -m venv .venv && . .venv/bin/activate`
- `pip install mx8`
- Optional vision/training deps: `pip install pillow numpy torch`

## Quickstart (local, no S3)

```python
import mx8

mx8.pack_dir(
    "/path/to/imagefolder",
    out="/path/to/mx8-dataset",
    shard_mb=512,
    label_mode="imagefolder",
    require_labels=True,
)

loader = mx8.vision.ImageFolderLoader(
    "/path/to/mx8-dataset@refresh",
    batch_size_samples=64,
    max_inflight_bytes=256 * 1024 * 1024,
    resize_hw=(224, 224),  # (H,W); optional
)

print(loader.classes)  # ["cat", "dog", ...] if labels.tsv exists

for images, labels in loader:
    pass
```

## Bounded memory (v0)

Set a hard cap and periodically print high-water marks:

```python
import mx8

loader = mx8.vision.ImageFolderLoader(
    "/path/to/mx8-dataset@refresh",
    batch_size_samples=64,
    max_inflight_bytes=256 * 1024 * 1024,
    max_queue_batches=8,
    prefetch_batches=4,
)

for step, (images, labels) in enumerate(loader):
    if step % 100 == 0:
        print(loader.stats())  # includes ram_high_water_bytes
```

Avoid patterns that intentionally accumulate batches:

```python
# ❌ Don't do this (will grow RSS regardless of any loader)
all_batches = list(loader)
```

## Labels (optional)

`label_mode="imagefolder"` is designed to scale:

- Per-sample records reference a numeric `label_id` (u64), not a repeated string.
- The human-readable mapping is stored once at `out/_mx8/labels.tsv`.

If your input layout is mixed (files directly under the prefix *and* subfolders), `label_mode="auto"` may disable ImageFolder labeling. To enforce ImageFolder semantics, use:

```python
mx8.pack_dir(..., label_mode="imagefolder", require_labels=True)
```
