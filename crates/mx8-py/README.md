# mx8 (Python)

This is the Python package for MX8 (built with PyO3 + maturin).

## Install (from wheel)

Once you have a wheel (from CI or local build):

- `python -m venv .venv && . .venv/bin/activate`
- `pip install mx8-*.whl`

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
    # Bounded-memory runtime stats (high-water is capped by max_inflight_bytes).
    # stats = loader.stats()
    pass
```

## Labels (optional)

`label_mode="imagefolder"` is designed to scale:

- Per-sample records reference a numeric `label_id` (u64), not a repeated string.
- The human-readable mapping is stored once at `out/_mx8/labels.tsv`.

If your input layout is mixed (files directly under the prefix *and* subfolders), `label_mode="auto"` may disable ImageFolder labeling. To enforce ImageFolder semantics, use:

```python
mx8.pack_dir(..., label_mode="imagefolder", require_labels=True)
```
