# Troubleshooting

## `reqwest` rlib error during build/run

Symptom:

- `error: crate reqwest required to be available in rlib format, but was not found in this form`

Fix:

- `cargo clean -p reqwest`
- rerun your original command (`cargo run ...` or `./scripts/smoke.sh`)

## `rg: command not found` in demo scripts

Some scripts use `rg` (ripgrep). Install ripgrep, then rerun:

- macOS (Homebrew): `brew install ripgrep`

## `permission denied: ./scripts/` or `command not found` with env vars

Common shell typo is splitting the command across lines incorrectly.

Correct:

- `MX8_BENCH_WANTS=1,4 WORLD_SIZE=16 ./scripts/demo2_minio_scale.sh`

Incorrect:

- `MX8_BENCH_WANTS=1,4 WORLD_SIZE=16 ./`
- then `scripts/demo2_minio_scale.sh` on next line

## Docker/MinIO startup failure

Symptom:

- `Cannot connect to the Docker daemon ...`

Fix:

- start Docker Desktop (or Docker daemon)
- rerun the MinIO gate/script

## `TrustStore configured to enable native roots ...` during MinIO smoke

Symptom:

- panic from `aws-smithy-http-client` in local dev build path:
  - `TrustStore configured to enable native roots but no valid root certificates parsed!`

Notes:

- This is an environment-specific TLS/native-roots issue in local debug builds.
- It was observed while running MinIO gates, not as a correctness issue in MX8 manifest logic.

Workaround:

- rerun with dev debug assertions disabled for that environment:
  - `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT=1 ./scripts/smoke.sh`

## `pip install mx8==X.Y.Z` cannot find version

Symptom:

- `No matching distribution found for mx8==...`

Fix:

- verify published versions: `python -m pip index versions mx8`
- install an available version (or publish the missing tag)

## Vision decode regression debugging

If you want to compare decode backends or validate Rust decode behavior:

- baseline Rust compare: `MX8_DECODE_BACKEND=rust MX8_DECODE_THREADS=4 MX8_RUST_JPEG_CODEC=turbo MX8_RUST_RESIZE_BACKEND=fast ...`
- full decode env surface is documented in `docs/python_api.md`

Default is Python decode in v0.

## `Killed: 9` in `demo2_minio.sh` logs

In recovery gates, the script intentionally kills one agent process to verify lease expiry + requeue behavior.

This line is expected when the gate is configured to test recovery.

## `byte-aware batching requires byte_length ...`

Symptom:

- loader fails when `target_batch_bytes` or `max_batch_bytes` is set

Fix:

- ensure manifest rows include both `byte_offset` and `byte_length`
- if manifest has full-object rows (`None/None`), disable byte-aware batching or repack/index so lengths are explicit

## Process OOM protection

If you want deterministic failure before OS OOM kill, set:

- `MX8_MAX_PROCESS_RSS_BYTES=<bytes>`

When process RSS exceeds this cap, MX8 fails fast with a clear `process rss ... exceeds max_ram_bytes ...` error.
