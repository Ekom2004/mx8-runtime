# V1.6 Zero-Manifest (`mx8.load`) TODO

Goal: `mx8.load("s3://bucket/prefix/", recursive=True)` works without manual snapshot command, while keeping bounded-memory and deterministic snapshot guarantees.

## Completed

- [x] Add Python API `recursive` flag to:
  - `mx8.load(...)`
  - `mx8.resolve_manifest_hash(...)`
  - `mx8.DataLoader(...)`
  - `mx8.vision.ImageFolderLoader(...)`
- [x] Thread `recursive` into `SnapshotResolverConfig`.
- [x] Apply `recursive` in local prefix indexing.
- [x] Replace S3 full-key buffering with streaming page processing (no full key list in RAM).
- [x] Add fail-fast guard for non-monotonic S3 key ordering during index scan.
- [x] Keep single-writer snapshot lock behavior (`snapshot_indexer_elected` + wait path).
- [x] Add local recursive/non-recursive tests in `mx8-snapshot`.
- [x] Add compact index summary proof event:
  - `event="snapshot_index_summary"` with `objects_indexed`, `pages_scanned` (S3), `scan_ms`, `recursive`, `label_mode`.
- [x] Add optional S3 spill/merge mode for key ordering with bounded key RAM:
  - `MX8_SNAPSHOT_S3_EXTERNAL_SORT=1`
  - `MX8_SNAPSHOT_S3_SPILL_KEYS_PER_RUN=<n>`
- [x] Add Phase-A dual-path manifest materialization:
  - `SnapshotResolverConfig.materialize_manifest_bytes` (default `true`)
  - coordinator + `mx8-snapshot-resolve` run with `materialize_manifest_bytes=false`
  - Python loader lazily materializes by `manifest_hash` when needed
- [x] Remove mandatory in-memory canonical bytes for S3 indexing in non-materialized mode:
  - S3 indexer now writes canonical TSV incrementally to temp file + streaming SHA-256
  - resolver persists by file path (`put_manifest_file`) when bytes are non-materialized
  - proof log still includes manifest byte count for indexed snapshot

## Next

- [x] Add a MinIO gate that asserts `recursive=false` excludes nested keys for S3 prefixes.
- [x] `s3://` manifest-store true streaming upload:
  - `put_manifest_file` now uses `ByteStream::from_path(...)` + conditional put
  - collision path verifies equivalence without loading local file into memory
  - file: `crates/mx8-manifest-store/src/s3.rs:282`
- [x] Control-plane streaming serve path:
  - `GetManifestStream` now uses manifest length + range reads (no full manifest `Vec<u8>` load)
  - file: `crates/mx8-coordinator/src/main.rs:1008`
- [ ] Agent/runtime streaming manifest ingest:
  - remove full-bytes requirement in agent/runtime parse path
  - files: `crates/mx8d-agent/src/main.rs:717`, `crates/mx8-runtime/src/pipeline.rs:130`
- [ ] End-to-end streaming parser contract:
  - define and implement manifest parser on `Read/AsyncRead` stream (not `Vec<u8>`)
  - thread this contract through resolver → coordinator → agent/runtime
