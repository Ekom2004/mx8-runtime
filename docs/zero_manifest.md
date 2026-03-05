# Zero-Manifest Loading

This note documents how MX8 loads data directly from a raw S3 prefix without requiring a precomputed manifest, and the scan-and-reservoir mechanism that makes it work.

Primary implementation: `crates/mx8-py/src/lib.rs` (scan activation and pipeline wiring), `crates/mx8-runtime/src/pipeline.rs` (S3 scan and reservoir logic).


## What problem is being solved?

Most data loading pipelines require you to prepare your dataset before you can use it. You generate a file index, convert to a special format, or run a packaging step. For teams that want to explore a new dataset quickly, or run a one-off job against files that have never been packaged, this is a real barrier.

The zero-manifest path removes that requirement. You point MX8 at any raw S3 prefix and it starts delivering bounded, shuffled batches immediately. No packing, no manifest file, no prep work. The same hard memory cap and backpressure guarantees apply as in any other MX8 loader.


## How it works

When `mx8.load` receives an S3 prefix and no precomputed manifest exists at `<prefix>/_mx8/manifest.tsv`, MX8 activates the scan path instead of the normal snapshot resolver.

The scan pages through the prefix using `ListObjectsV2`, collecting every object key and byte size. As objects arrive, they are fed into a bounded shuffle reservoir so the pipeline does not wait for the full listing to finish. Batches start flowing as soon as enough objects have been seen.

If the prefix follows an ImageFolder layout — where each immediate subdirectory is a class name — MX8 detects the class names from the key paths and assigns integer label IDs on the fly. No label file is needed.

Once the scan completes, MX8 derives a synthetic snapshot identifier from the prefix so the rest of the system — autotune, stats, proof logs, lease semantics — behaves identically to the manifest-backed path.

If a precomputed manifest is found at `_mx8/manifest.tsv` during the preflight check, MX8 silently falls back to the normal path. No code change is needed. The scan path is only used when no manifest exists.


## Using zero-manifest loading

The API is identical to a normal `mx8.load` call. No special arguments are needed.

```python
import mx8

loader = mx8.load(
    "s3://your-bucket/raw-dataset/",
    batch=64,
    ram_gb=12,
    profile="balanced",
)

for batch in loader:
    process(batch.payload, batch.offsets)
```

To consume ImageFolder labels when they are available:

```python
for batch in loader:
    labels = batch.label_ids  # list of ints, one per sample, or None
```

Label ID assignment is first-seen order within a single scan run and is not stable across separate runs. If you need stable label IDs — for checkpointing, reproducibility, or multi-run comparisons — pack the dataset first with `mx8.pack`. Label IDs are then stable and baked into the manifest permanently.


## When to use zero-manifest vs. a packed manifest

Zero-manifest is the right choice for exploration, first runs, and ad-hoc jobs. There is no setup and no wait. The tradeoffs are that shuffle quality is a bounded approximation, label IDs are not stable across runs, and startup time scales with the number of objects in the prefix.

A packed manifest is the right choice for production. You run `mx8.pack` once, and from that point forward MX8 uses the precomputed manifest directly — near-zero startup time, full deterministic shuffle, stable label IDs, and a cryptographic hash you can use to pin and reproduce any run exactly.

The typical path is to start with zero-manifest during development and pack once you are ready for production.


## Configuration

`MX8_ZERO_MANIFEST_ENABLED` controls whether the scan path is active. It defaults to true. Set it to false to require a manifest and disable zero-manifest entirely.

`MX8_ZERO_MANIFEST_RESERVOIR` controls the shuffle reservoir size, which defaults to 100,000 entries. A larger reservoir gives better shuffle quality at the cost of more memory during the scan phase.

`MX8_S3_SCAN_SHUFFLE_SEED` pins the shuffle seed so you get the same scan order across runs without packing. By default it uses the current time, so each run shuffles differently.


## Proof logs

MX8 emits structured events to the `mx8_proof` log target during the scan. Enable with `MX8_LOG=mx8_proof=info`.

The `zero_manifest_scan_enabled` event fires when the scan path activates and records the dataset base, whether recursive scanning is on, and the reservoir size.

The `zero_manifest_scan_fallback_precomputed_manifest` event fires when a precomputed manifest is found and MX8 switches to the normal path.

The `s3_scan_completed` event fires when the scan finishes and records how many objects were seen, how many records were emitted, and how many were sent downstream.


## Limitations

Zero-manifest only activates for `s3://` prefixes. Filesystem paths always require a manifest — use `mx8.pack_dir` to create one for local datasets.

Pinned links using the `@sha256` syntax require a real manifest by definition. That is what the hash is pointing to.

For very large prefixes with millions of objects, LIST scan latency adds up at startup. Packing once eliminates this cost entirely for production workloads.

The reservoir shuffle is a bounded approximation. If the dataset is significantly larger than the reservoir size, shuffle quality degrades. Increase `MX8_ZERO_MANIFEST_RESERVOIR` or use a packed manifest for strict shuffle guarantees.
