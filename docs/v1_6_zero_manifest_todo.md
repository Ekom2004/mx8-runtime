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
- [x] Agent/runtime streaming manifest ingest:
  - remove full-bytes requirement in agent/runtime parse path
  - agent no longer preloads full `Vec<ManifestRecord>` at startup; per-lease path uses runtime range reader path
  - files: `crates/mx8d-agent/src/main.rs`, `crates/mx8-runtime/src/pipeline.rs`
- [ ] End-to-end streaming parser contract (partial):
  - [x] define and implement manifest parser on `Read/BufRead` stream (not `Vec<u8>`) in runtime
  - [x] thread parser contract into agent/runtime lease path (manifest path range parsing is reader-based)
  - [x] direct async stream-to-runtime parser handoff without intermediate cache file (default path in agent)

## Phase 1 — Streaming parser contract (scoped)

Goal: add a single manifest parsing contract that accepts stream readers (`Read`/`BufRead`) so agent/runtime can parse without requiring a full manifest `Vec<u8>`.

### TODO checklist

- [x] Add a reader-first parser surface in `mx8-runtime`:
  - `parse_canonical_manifest_tsv_reader<R: BufRead>(reader: R) -> Result<Vec<ManifestRecord>>`
  - Keep/route existing bytes/path helpers through the reader surface for backward compatibility.
- [x] Keep canonical invariants identical to current parser behavior:
  - `schema_version` header required and validated
  - row shape and field validation unchanged
  - `sample_id` monotonic/sequential checks unchanged
- [x] Add a bounded-memory parser mode for stream ingestion in runtime hot path:
  - no `read_to_end` / no mandatory full-manifest materialization in parser path
  - parser consumes line-by-line from reader
- [x] Add a small adapter surface for async transport boundaries:
  - explicit conversion boundary where async chunk streams are turned into a `BufRead` source
  - keep transport concerns out of parser logic
- [x] Keep public runtime call sites source-compatible for this phase:
  - existing APIs that accept `Vec<u8>` continue to work
  - new reader-based path is available for phase-2 agent wiring

### Tests (required)

- [x] Unit: `manifest_reader_accepts_valid_canonical_tsv`
  - valid canonical TSV parsed via `BufRead` path.
- [x] Unit: `manifest_reader_rejects_missing_schema_header`
  - fails when `schema_version=` header is missing.
- [x] Unit: `manifest_reader_rejects_non_sequential_sample_ids`
  - fails on gaps/regressions in `sample_id`.
- [x] Unit: `manifest_reader_matches_bytes_parser_semantics`
  - same input parsed via bytes helper and reader helper yields identical records/errors.
- [x] Unit: `manifest_reader_handles_large_input_without_full_buffer_requirement`
  - synthetic large manifest streamed through reader path; parser succeeds with bounded behavior.
- [x] Integration (runtime): `runtime_stream_reader_path_parity_with_bytes_path`
  - same manifest through reader path and bytes path yields identical delivered sample ordering.

### Success means

- [x] Functional parity:
  - reader path and bytes path return the same parsed records and the same validation failures.
- [x] Correctness invariants preserved:
  - canonical TSV contract (`schema_version`, row format, sample_id invariants) unchanged.
- [x] Memory behavior improved for phase handoff:
  - parser path does not require full-manifest materialization to parse.
- [x] Migration-ready:
  - phase-2 agent/runtime wiring can swap to reader-based ingest without changing manifest semantics.

### Verification command

- `cargo test -p mx8-runtime manifest_reader`
- `cargo test -p mx8-runtime runtime_stream_reader_path_parity_with_bytes_path`

## Phase 3 — Direct stream handoff safety cut

Goal: enable direct `GetManifestStream` -> runtime parser handoff (no intermediate cache file) with explicit fail-closed semantics and bounded buffering.

### Safety semantics (must hold)

- [x] Truncated stream fails closed:
  - if manifest stream ends early / transport breaks mid-parse, lease execution does not start.
  - no partial manifest is accepted.
- [x] Schema mismatch hard-fails:
  - if chunk schema/version differs across a stream, manifest ingest aborts immediately.
  - no fallback parsing of mixed-schema stream.
- [x] Bounded buffering + backpressure:
  - direct stream parser has explicit hard cap on in-flight partial line bytes.
  - oversized single-line payloads hard-fail instead of growing unbounded parser buffers.

### TODO checklist

- [x] Add direct stream ingest path in `mx8d-agent` as default:
  - default mode: direct stream path
  - cached-path fallback retired after burn-in.
- [x] Wire direct stream chunks into runtime parser adapter:
  - use reader/chunk contract from `mx8-runtime` parser surface.
  - do not require a full `Vec<u8>` materialization.
- [x] Make failure semantics explicit in code + logs:
  - `manifest_stream_truncated` (or equivalent) emitted on early EOF/transport failure.
  - `manifest_schema_mismatch` (or equivalent) emitted on chunk schema/version mismatch.
  - no `lease_started` event when manifest ingest failed.
- [x] Add bounded stream-buffer controls for handoff:
  - explicit parser carry cap env/config (`MX8_AGENT_MANIFEST_STREAM_MAX_LINE_BYTES`, default 8 MiB).
  - oversize chunk/line path covered by test (`manifest_stream_backpressure_bounded`).

### Tests (required)

- [x] Integration: `manifest_stream_truncated_fails_closed`
  - simulate truncated `GetManifestStream`; assert lease execution does not start.
- [x] Integration: `manifest_stream_schema_mismatch_hard_fails`
  - feed mixed schema/version chunks; assert immediate hard-fail.
- [x] Integration: `lease_manifest_stream_parity_with_cached_path`
  - same manifest via direct stream and cached-file path yields identical lease-level progress outcome.
- [x] Unit: `manifest_stream_backpressure_bounded`
  - parser carry buffer remains bounded by configured cap.

### Success means

- [x] Correctness:
  - direct stream path preserves canonical manifest semantics and deterministic range selection.
- [x] Safety:
  - truncated or schema-invalid streams fail closed with zero lease execution.
- [x] Resource bounds:
  - parser carry buffer remains bounded by explicit cap on direct stream path.
- [x] Rollout readiness:
  - direct path passes smoke gate with fallback path still available.

### Verification command

- `RUSTUP_TOOLCHAIN=stable cargo test -p mx8d-agent manifest_stream`
- `RUSTUP_TOOLCHAIN=stable cargo test -p mx8-runtime manifest_reader`
- `MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT=1 ./scripts/smoke.sh`
