# CLI Reference

This page documents the user-facing CLI tools that ship with MX8.

Primary implementations: `crates/mx8-snapshot/src/bin/`, `crates/mx8-runtime/src/bin/`.


## Building the tools

```bash
cargo build -p mx8-snapshot --features s3 --bin mx8-pack-s3
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve
cargo build -p mx8-runtime  --features s3 --bin mx8-seed-s3
```

`mx8-pack-s3` and `mx8-seed-s3` require the `s3` feature. `mx8-snapshot-resolve` requires it when working with S3 links or S3-backed manifest stores. If a binary is run without the required feature, it exits with code 2 and prints a clear message.

For MinIO or other S3-compatible endpoints in development, set these environment variables:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=1
export MX8_S3_ENDPOINT_URL='http://127.0.0.1:9000'
export MX8_S3_FORCE_PATH_STYLE=1
```


## mx8-pack-s3

Packs many small S3 objects into MX8 tar shards and writes a canonical manifest. Run this once on a raw dataset prefix to produce a packed dataset that MX8 can load efficiently without large S3 LIST operations.

```bash
cargo run -p mx8-snapshot --features s3 --bin mx8-pack-s3 -- \
  --pack-in 's3://my-bucket/raw/train/' \
  --pack-out 's3://my-bucket/mx8/train/'
```

On success, it prints a summary line to stdout:

```
samples=<n> shards=<n> manifest_key=<key> manifest_hash=<sha256>
```

| CLI argument | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--pack-in` | `MX8_PACK_IN` | required | S3 input prefix |
| `--pack-out` | `MX8_PACK_OUT` | required | S3 output prefix |
| `--shard-mb` | `MX8_PACK_SHARD_MB` | `512` | target uncompressed shard size in MiB |
| `--label-mode` | `MX8_S3_LABEL_MODE` | `auto` | `auto`, `none`, or `imagefolder` |
| `--require-labels` | `MX8_PACK_REQUIRE_LABELS` | `false` | fail if any key is not ImageFolder-compatible |

Runtime env overrides (no CLI flag):

- `MX8_PACK_PARALLEL_FETCHES` (default `128`): S3 GET parallelism used by the packer.
- `MX8_PACK_PART_MB` (default `16`): multipart upload part size in MiB.


## mx8-snapshot-resolve

Resolves a dataset link to a pinned snapshot and prints the `manifest_hash`. Useful for pre-pinning a snapshot before a job starts, or for verifying that a dataset link resolves correctly.

```bash
cargo run -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve -- \
  --dataset-link 's3://my-bucket/mx8/train/@refresh' \
  --manifest-store-root '~/.mx8/manifests'
```

On success, it prints:

```
manifest_hash: <sha256>
```

| CLI argument | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--dataset-link` | `MX8_DATASET_LINK` | required | plain path, `@refresh`, or `@sha256:<hash>` |
| `--manifest-store-root` | `MX8_MANIFEST_STORE_ROOT` | `~/.mx8/manifests` | filesystem path or S3 prefix |
| `--dev-manifest-path` | `MX8_DEV_MANIFEST_PATH` | unset | dev manifest bootstrap path |
| `--node-id` | `MX8_NODE_ID` | `resolver` | lock owner identity for proof logs |
| `--snapshot-lock-stale-ms` | `MX8_SNAPSHOT_LOCK_STALE_MS` | `60000` | how long before a stale lock is reaped |
| `--snapshot-wait-timeout-ms` | `MX8_SNAPSHOT_WAIT_TIMEOUT_MS` | `30000` | how long to wait for a competing resolver |
| `--snapshot-recursive` | `MX8_SNAPSHOT_RECURSIVE` | `true` | whether to include nested keys during prefix indexing |


## mx8-seed-s3

Uploads a local file as a single S3 object. Used by deterministic local gates to seed test data into MinIO before running a gate.

```bash
cargo run -p mx8-runtime --features s3 --bin mx8-seed-s3 -- \
  --bucket mx8-demo \
  --key data.bin \
  --file ./data.bin
```

| CLI argument | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--bucket` | `MX8_MINIO_BUCKET` | `mx8-demo` | destination bucket |
| `--key` | `MX8_MINIO_KEY` | `data.bin` | destination object key |
| `--file` | `MX8_SEED_FILE` | required | local file path to upload |

The tool emits an `s3_seed_complete` proof event with `bucket`, `key`, and `bytes` on success. Bucket creation is best-effort — a creation failure is logged as a warning and the upload still proceeds. Upload errors fail the process.


## Gate commands

To run the deterministic snapshot resolution gate against MinIO: `./scripts/minio_s3_prefix_snapshot_gate.sh`.

To test recursive versus non-recursive snapshot behavior: `./scripts/minio_s3_prefix_recursive_gate.sh`.

To run the full pack, resolve, and label verification gate: `./scripts/minio_pack_gate.sh`.
