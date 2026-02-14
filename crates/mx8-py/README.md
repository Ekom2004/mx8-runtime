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
    resize_hw=(224, 224),  # (H,W); optional
)

print(loader.classes)  # ["cat", "dog", ...] if labels.tsv exists

for images, labels in loader:
    pass
```

