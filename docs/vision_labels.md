# Vision Labels and Layouts

This page explains how MX8 handles class labels for vision datasets, why integer label IDs are used instead of strings, and how to control label behavior when packing.


## Why label IDs instead of strings

For large datasets with millions of samples, storing a label string per row would waste significant space in the manifest and memory at runtime. MX8 stores a compact integer `label_id` per sample and writes a single shared mapping file at `_mx8/labels.tsv` that maps each `label_id` to its human-readable name. This keeps manifests small and stable regardless of class count.


## How labels are created

Labels are generated during packing. Pass `label_mode="imagefolder"` to `mx8.pack` or `mx8.pack_dir` to enable ImageFolder labeling, where the class name is the name of the immediate parent directory (`prefix/<label>/<file>`).

```python
import mx8

mx8.pack_dir(
    "/data/raw_images",
    out="/data/mx8_images",
    label_mode="imagefolder",
    require_labels=True,
)
```

`label_mode="auto"` attempts to detect the layout automatically. `label_mode="none"` produces a manifest with no label information. `require_labels=True` makes the packer fail fast if it cannot produce labels for every sample, which is useful for catching mixed or ambiguous layouts early.


## Mixed layouts

If your input directory has files directly under the root alongside class subdirectories, `label_mode="auto"` may disable ImageFolder labels because the layout is ambiguous. Use `label_mode="imagefolder"` with `require_labels=True` for a strict check that fails if any sample is outside a labeled subdirectory, or use `label_mode="none"` if labels are not needed.


## Training vs inference

For classification training, labels are typically required. Use `mx8.image` to get decoded images and integer label tensors directly, or call `batch.to_torch_with_labels()` on the raw loader batch.

For inference and ETL, labels are usually optional. Use `mx8.load` and process raw bytes and sample IDs without worrying about label assignment.


## Zero-manifest label assignment

When using zero-manifest loading (pointing MX8 at a raw S3 prefix without a precomputed manifest), MX8 assigns label IDs on the fly from the directory names during the scan. These IDs are not stable across separate runs. If you need stable label IDs for checkpointing or multi-run comparisons, pack the dataset first. See `docs/zero_manifest.md` for details.
