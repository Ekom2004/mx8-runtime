# Vision Labels and Layouts (v0)

This page explains how MX8 handles labels for vision datasets and why label IDs are used.

## Why label IDs exist

For large datasets, repeating label strings per sample wastes memory and manifest size.

MX8 stores:

- per-sample `label_id` (numeric, compact)
- one shared mapping file `_mx8/labels.tsv` (`label_id<TAB>label`)

This keeps manifests small and stable even with many classes / many samples.

## How labels are created

Use `mx8.pack` or `mx8.pack_dir` with:

- `label_mode="imagefolder"` to force ImageFolder labeling (`prefix/<label>/<file>`)
- `label_mode="auto"` to detect layout
- `require_labels=True` to fail fast if labels cannot be produced

Example:

```python
import mx8

mx8.pack_dir(
    "/data/raw_images",
    out="/data/mx8_images",
    label_mode="imagefolder",
    require_labels=True,
)
```

## Mixed layouts

If your input mixes:

- files directly under the root, and
- class subfolders (`cat/...`, `dog/...`)

then `label_mode="auto"` may disable ImageFolder labels (ambiguous layout).

Use one of:

- `label_mode="imagefolder", require_labels=True` (strict; fail if invalid)
- `label_mode="none"` (explicitly unlabeled pipeline)

## Training vs inference/ETL

- Training/classification: labels typically required; use `mx8.image(...)` or `batch.to_torch_with_labels()`.
- Inference/ETL: labels optional; use `mx8.load(...)` and process raw bytes / sample IDs.

## API contract to users

Document this to users:

- Labels are optional in v0.
- If labels are enabled, they are represented as stable numeric IDs.
- Human-readable label names come from `_mx8/labels.tsv`.
- Enforce labeling only when the workload needs it (`require_labels=True`).
