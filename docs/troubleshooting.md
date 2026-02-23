# Troubleshooting

This page covers common errors and how to fix them. For production incidents and rollback procedures, see `docs/prod_runbook.md`.


## reqwest rlib error during build or run

You may see this error during `cargo run` or `./scripts/smoke.sh`:

```
error: crate reqwest required to be available in rlib format, but was not found in this form
```

This is an incremental build artifact issue, not a code problem. Fix it by cleaning the reqwest crate and rerunning:

```bash
cargo clean -p reqwest
```

Then rerun your original command.


## rg: command not found in demo scripts

Some scripts use `rg` (ripgrep) for log scanning. Install it with `brew install ripgrep` on macOS, then rerun the script.


## permission denied or command not found with environment variables

The most common cause is splitting a command with inline env vars across multiple lines incorrectly. The env vars must be on the same line as the command or exported beforehand.

Correct:

```bash
MX8_BENCH_WANTS=1,4 WORLD_SIZE=16 ./scripts/demo2_minio_scale.sh
```

Incorrect (splits the command):

```bash
MX8_BENCH_WANTS=1,4 WORLD_SIZE=16 ./
scripts/demo2_minio_scale.sh
```


## Docker or MinIO startup failure

If you see `Cannot connect to the Docker daemon`, Docker is not running. Start Docker Desktop or the Docker daemon, then rerun the MinIO gate or script.


## TrustStore native roots panic during MinIO smoke

You may see a panic from `aws-smithy-http-client` in local debug builds:

```
TrustStore configured to enable native roots but no valid root certificates parsed!
```

This is an environment-specific TLS issue in debug builds, not a correctness problem with MX8. Rerun with debug assertions disabled:

```bash
CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT=1 ./scripts/smoke.sh
```


## pip install cannot find a specific version

If `pip install mx8==X.Y.Z` fails with no matching distribution, the version may not be published yet. Check available versions with:

```bash
python -m pip index versions mx8
```

Then install an available version, or publish the missing tag if you are the maintainer.


## Vision decode regression debugging

To compare decode backends or validate Rust decode behavior, set `MX8_DECODE_BACKEND=rust` and configure the Rust path:

```bash
MX8_DECODE_BACKEND=rust \
MX8_DECODE_THREADS=4 \
MX8_RUST_JPEG_CODEC=turbo \
MX8_RUST_RESIZE_BACKEND=fast \
python your_script.py
```

The full decode environment surface is documented in `docs/python_api.md`. The default backend in v1.8 is Python/Pillow.


## Killed: 9 in demo2_minio.sh logs

This is expected. The recovery gate script intentionally kills one agent process to verify lease expiry and requeue behavior. The kill signal is part of the test, not an error.


## byte-aware batching requires byte_length

If the loader fails with this message when `target_batch_bytes` or `max_batch_bytes` is set, your manifest rows are missing `byte_length` values. Byte-aware batching requires both `byte_offset` and `byte_length` to be present in the manifest. If your manifest has full-object rows with no offsets, either disable byte-aware batching or repack with explicit byte ranges.


## Process OOM protection

MX8 loader surfaces set a default process RSS fail-fast cap. When the process RSS exceeds the cap, MX8 fails fast with a clear error message rather than waiting for the OS to OOM-kill it.

To set an explicit org-level cap:

```bash
MX8_MAX_PROCESS_RSS_BYTES=<bytes>
```

The error message will include the current RSS and the configured cap so you can adjust accordingly.


## mx8-tui headless probe failures

If the TUI exits with `lease panel remained empty`, `runtime panel remained empty`, or `manifest panel remained empty`, check the following.

For the lease panel: verify the coordinator is reachable at `--coord-url` and that `--job-id` matches the active job.

For the runtime panel: verify at least one agent is heartbeating. The runtime panel is built entirely from node heartbeat stats, so it stays empty if no agents have checked in.

For the manifest panel: verify the manifest exists in the coordinator's `manifest_store`. Use `--manifest-path` to point the TUI at a local TSV file as a fallback.
