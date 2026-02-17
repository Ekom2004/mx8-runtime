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

## `pip install mx8==X.Y.Z` cannot find version

Symptom:

- `No matching distribution found for mx8==...`

Fix:

- verify published versions: `python -m pip index versions mx8`
- install an available version (or publish the missing tag)

## Vision decode regression debugging

If you want to compare decode backends or validate Rust decode behavior:

- `MX8_DECODE_BACKEND=rust ...`

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
