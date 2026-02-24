# MX8 Environment Variable Reference (v1.8)

This is the canonical `MX8_*` configuration inventory.

Inventory source:

- runtime code in `crates/`
- operator/demo/gate scripts in `scripts/`

Generation command:

```bash
rg -o --no-filename "MX8_[A-Z0-9_]+" crates scripts | sort -u
```

Current inventory size: `237` variables.

## Stability Classes

- `stable`: production/operator contract; backward-compatible within v1.x.
- `experimental`: shipped but expected to evolve.
- `internal`: dev/gate/test-only; no compatibility guarantees.

## Core Operator Runtime Variables (`stable`)

| Variable | Default | Scope | Valid values | Stability |
| --- | --- | --- | --- | --- |
| `MX8_COORD_BIND_ADDR` | `0.0.0.0:50051` | coordinator | socket address | stable |
| `MX8_COORD_HA_ENABLE` | `false` | coordinator | boolish (`true/false/1/0`) | stable |
| `MX8_COORD_HA_LEASE_PATH` | `<manifest_store_root>/../ha/<manifest_hash>.leader_lease` | coordinator | filesystem path | stable |
| `MX8_COORD_HA_LEADER_ID` | `<hostname>-<pid>` | coordinator | non-empty string | stable |
| `MX8_COORD_HA_LEASE_TTL_MS` | `5000` | coordinator | integer >= 1 | stable |
| `MX8_COORD_HA_RENEW_INTERVAL_MS` | `1000` | coordinator | integer >= 1 | stable |
| `MX8_WORLD_SIZE` | `1` | coordinator | integer >= 1 | stable |
| `MX8_HEARTBEAT_INTERVAL_MS` | `1000` | coordinator | integer >= 1 | stable |
| `MX8_LEASE_TTL_MS` | `10000` | coordinator | integer >= 1 | stable |
| `MX8_DATASET_LINK` | unset | coordinator/resolver | plain path/prefix, `@refresh`, `@sha256:<hash>` | stable |
| `MX8_MANIFEST_STORE_ROOT` | `~/.mx8/manifests` | coordinator/resolver/python loaders | filesystem path or `s3://...` prefix | stable |
| `MX8_SNAPSHOT_LOCK_STALE_MS` | `60000` | coordinator/resolver | integer >= 1 | stable |
| `MX8_SNAPSHOT_WAIT_TIMEOUT_MS` | `30000` | coordinator/resolver | integer >= 1 | stable |
| `MX8_SNAPSHOT_RECURSIVE` | `true` (resolver CLI) | snapshot resolver | bool (`true/false/1/0`) | stable |
| `MX8_MANIFEST_HASH` | `dev` (legacy path) | coordinator/demo | manifest hash string | stable |
| `MX8_SHUFFLE` | `false` | coordinator | boolish (`true/false/1/0`) | stable |
| `MX8_SEED` | `0` | coordinator | `u64` | stable |
| `MX8_EPOCH` | `0` | coordinator | `u32` | stable |
| `MX8_COORD_URL` | `http://127.0.0.1:50051` | agent/tui/distributed python loader | URL | stable |
| `MX8_JOB_ID` | `local-job` (agent), `local-job` (tui) | control plane | non-empty string | stable |
| `MX8_NODE_ID` | `local-node` (agent), `resolver` (resolver CLI) | control plane | non-empty string | stable |
| `MX8_BATCH_SIZE_SAMPLES` | `512` | agent/runtime/python | integer >= 1 | stable |
| `MX8_PREFETCH_BATCHES` | `1` | agent/runtime/python | integer >= 1 | stable |
| `MX8_MAX_QUEUE_BATCHES` | `64` | agent/runtime/python | integer >= 1 | stable |
| `MX8_MAX_INFLIGHT_BYTES` | `134217728` (agent) | agent/runtime/python | integer >= 1 | stable |
| `MX8_TARGET_BATCH_BYTES` | unset | agent/python | integer >= 1 and <= max batch | stable |
| `MX8_MAX_BATCH_BYTES` | unset | agent/python | integer >= 1 and <= inflight cap | stable |
| `MX8_MAX_PROCESS_RSS_BYTES` | unset (derived defaults in Python loaders) | agent/python loaders | integer >= 1 | stable |
| `MX8_PROGRESS_INTERVAL_MS` | `500` (agent) | agent/distributed python loader | integer >= 1 | stable |
| `MX8_GRPC_MAX_MESSAGE_BYTES` | `67108864` | coordinator/agent/tui/python distributed loader | integer >= 1 | stable |
| `MX8_METRICS_SNAPSHOT_INTERVAL_MS` | `0` | coordinator/agent/runtime | integer >= 0 (0 disables) | stable |
| `MX8_TUI_POLL_MS` | `1000` | tui | integer >= 1 | stable |
| `MX8_TUI_HEADLESS_POLLS` | `0` | tui | integer >= 0 | stable |
| `MX8_TUI_MANIFEST_HASH` | unset | tui | manifest hash | stable |
| `MX8_TUI_SEARCH` | `""` | tui | string | stable |
| `MX8_TUI_ROWS_PER_PAGE` | `12` | tui | integer >= 1 | stable |
| `MX8_TUI_MANIFEST_PATH` | unset | tui | local TSV path | stable |
| `MX8_TUI_STALE_HEARTBEAT_MS` | `5000` | tui | integer >= 1 | stable |
| `MX8_TUI_LEASE_STALL_MS` | `10000` | tui | integer >= 1 | stable |
| `MX8_PACK_IN` | unset | `mx8-pack-s3` | `s3://...` prefix | stable |
| `MX8_PACK_OUT` | unset | `mx8-pack-s3` | `s3://...` prefix | stable |
| `MX8_PACK_SHARD_MB` | `512` | `mx8-pack-s3` | integer >= 1 | stable |
| `MX8_S3_LABEL_MODE` | `auto` | snapshot/packer | `auto|none|imagefolder` | stable |
| `MX8_PACK_REQUIRE_LABELS` | `false` | `mx8-pack-s3` | bool (`true/false`) | stable |
| `MX8_MINIO_BUCKET` | `mx8-demo` | `mx8-seed-s3` | bucket name | stable |
| `MX8_MINIO_KEY` | `data.bin` | `mx8-seed-s3` | object key | stable |
| `MX8_SEED_FILE` | unset | `mx8-seed-s3` | local file path | stable |
| `MX8_S3_ENDPOINT_URL` | unset | all S3 clients | URL | stable |
| `MX8_S3_FORCE_PATH_STYLE` | unset (auto-true when endpoint override exists) | all S3 clients | bool (`true/false/1/0`) | stable |
| `MX8_LOG` | fallback: `RUST_LOG`, then `info` | all Rust binaries | tracing filter string | stable |

## Feature/Performance Knobs (`experimental`)

| Variable | Default | Scope | Valid values | Stability |
| --- | --- | --- | --- | --- |
| `MX8_ZERO_MANIFEST_ENABLED` | `true` | Python DataLoader | bool (`1/true/yes/on`) | experimental |
| `MX8_ZERO_MANIFEST_RESERVOIR` | `100000` | Python DataLoader | integer >= 1 | experimental |
| `MX8_DEV_LEASE_WANT` | `0` (agent) | agent dev lease loop | integer >= 0 | experimental |
| `MX8_S3_SCAN_SHUFFLE_SEED` | current unix time | runtime S3 scanner | integer (`u64`) | experimental |
| `MX8_AGENT_MANIFEST_STREAM_MAX_LINE_BYTES` | `8388608` | agent manifest parser | integer >= 1 | experimental |
| `MX8_SNAPSHOT_S3_EXTERNAL_SORT` | `false` | snapshot indexer | bool (`true/false/1/0`) | experimental |
| `MX8_SNAPSHOT_S3_SPILL_KEYS_PER_RUN` | `100000` | snapshot indexer | integer >= 1 | experimental |
| `MX8_VIDEO_STAGE1_INDEX` | `false` | snapshot video indexing | bool (`1/true/yes/on`) | experimental |
| `MX8_VIDEO_STAGE1_DISABLE_FFPROBE` | `false` | snapshot video indexing | bool (`1/true/yes/on`) | experimental |
| `MX8_VIDEO_STAGE1_BYTES_PER_FRAME_ESTIMATE` | `51200` | snapshot video indexing | integer >= 1 | experimental |
| `MX8_VIDEO_DECODE_BACKEND` | `cli` | Python video loader | `cli|ffi` | experimental |
| `MX8_FFMPEG_BIN` | `ffmpeg` | Python video decode path | executable path | experimental |
| `MX8_VIDEO_STAGE2D_MAX_RANGES` | `8` | Python video range planner | integer >= 1 | experimental |
| `MX8_VIDEO_STAGE2D_MERGE_GAP_BYTES` | `0` | Python video range planner | integer >= 0 | experimental |
| `MX8_VIDEO_STAGE2_MAX_CLIPS_IN_MEMORY` | `2000000` | Python video stage1 index | integer >= 1 | experimental |
| `MX8_VIDEO_STAGE2_BYTES_PER_CLIP` | `4096` | Python video loader | integer >= 1 | experimental |
| `MX8_VIDEO_AUTOTUNE_PERIOD_BATCHES` | `16` | Python video runtime autotune | integer >= 1 | experimental |
| `MX8_MIX_AUTOTUNE_PERIOD_TICKS` | `32` | Python mix runtime autotune | integer >= 1 | experimental |
| `MX8_MIX_SNAPSHOT` | `false` | Python mix diagnostics | bool (`1/true/yes/on`) | experimental |
| `MX8_MIX_SNAPSHOT_PERIOD_TICKS` | `64` | Python mix diagnostics | integer >= 1 | experimental |
| `MX8_DECODE_BACKEND` | `python` | Python image loader | `python|rust` | experimental |
| `MX8_RUST_JPEG_CODEC` | `zune` | Python image loader | `zune|image|turbo` | experimental |
| `MX8_RUST_RESIZE_BACKEND` | `fast` | Python image loader | `fast|image` | experimental |
| `MX8_DECODE_THREADS` | host parallelism | Python image loader | integer >= 1 | experimental |

## Internal and Gate Variables (`internal`)

Variables used only by demos/gates/bench scripts or dev binaries are `internal`.

Policy for internal vars:

- default: script/binary-specific fallback, often unset unless script exports value
- scope: CI/local gates, demos, benchmarks, incident simulations
- valid values: defined by owning script/binary
- compatibility: no guarantee

Common internal families:

- `MX8_SMOKE_*`
- `MX8_TORCH_*`
- `MX8_IMAGE_BENCH_*`
- `MX8_VIDEO_STAGE2A_*`, `MX8_VIDEO_STAGE2B_*`, `MX8_VIDEO_STAGE3A_*`
- `MX8_MIX_GATE_*`
- `MX8_DEV_*` beyond explicitly documented operator usage

## Full Inventory (All `MX8_*` names found in repo)

```text
MX8_AGENT_MANIFEST_STREAM_MAX_LINE_BYTES
MX8_AUTOTUNE
MX8_AUTOTUNE_AB_MODE
MX8_AUTOTUNE_PROFILE
MX8_BATCH_SIZE_SAMPLES
MX8_BENCH_WANTS
MX8_BURNIN_LOG_DIR
MX8_BURNIN_RETRIES
MX8_BURNIN_RUNS
MX8_BYTES_PER_SAMPLE
MX8_CHECK_OFFLINE
MX8_COORD_BIND_ADDR
MX8_COORD_HA_ENABLE
MX8_COORD_HA_LEADER_ID
MX8_COORD_HA_LEASE_PATH
MX8_COORD_HA_LEASE_TTL_MS
MX8_COORD_HA_RENEW_INTERVAL_MS
MX8_COORD_URL
MX8_DATASET_LINK
MX8_DATASET_LINK_A
MX8_DATASET_LINK_B
MX8_DECODE_BACKEND
MX8_DECODE_BENCH_MIN_SPEEDUP
MX8_DECODE_THREADS
MX8_DEV_BLOCK_SIZE
MX8_DEV_LEASE_WANT
MX8_DEV_MANIFEST_PATH
MX8_DEV_MANIFEST_PATH_A
MX8_DEV_MANIFEST_PATH_B
MX8_DEV_TOTAL_SAMPLES
MX8_END_ID
MX8_EPOCH
MX8_FFMPEG_BIN
MX8_GRPC_MAX_MESSAGE_BYTES
MX8_HEARTBEAT_INTERVAL_MS
MX8_HTTP_BANDWIDTH_BPS
MX8_HTTP_FAIL_EVERY_N
MX8_HTTP_LATENCY_MS
MX8_IMAGE_BENCH_BACKENDS
MX8_IMAGE_BENCH_BATCH_SIZE
MX8_IMAGE_BENCH_DECODE_THREADS_LIST
MX8_IMAGE_BENCH_IMAGE_HW
MX8_IMAGE_BENCH_MODES
MX8_IMAGE_BENCH_RUST_CODECS
MX8_IMAGE_BENCH_SAMPLES
MX8_IMAGE_BENCH_STEPS
MX8_IMAGE_BENCH_TORCH_THREADS_LIST
MX8_IMAGE_BENCH_WARMUP_STEPS
MX8_JOB_ID
MX8_JOB_ID_AUTOTUNE_AB
MX8_JOB_ID_DETERMINISM
MX8_JOB_ID_NODUPES
MX8_JOB_ID_RESTART
MX8_KEEP_ARTIFACTS
MX8_KILL_AFTER_MS
MX8_KILL_NODE_INDEX
MX8_LEASE_TTL_MS
MX8_LOG
MX8_MANIFEST_CACHE_DIR
MX8_MANIFEST_HASH
MX8_MANIFEST_STORE_BUCKET
MX8_MANIFEST_STORE_PREFIX
MX8_MANIFEST_STORE_ROOT
MX8_MAX_BATCH_BYTES
MX8_MAX_INFLIGHT_BYTES
MX8_MAX_PROCESS_RSS_BYTES
MX8_MAX_QUEUE_BATCHES
MX8_METRICS_SNAPSHOT_INTERVAL_MS
MX8_MINIO_BUCKET
MX8_MINIO_CONSOLE_PORT
MX8_MINIO_IMAGE
MX8_MINIO_KEY
MX8_MINIO_OUT_PREFIX
MX8_MINIO_PORT
MX8_MINIO_PREFIX
MX8_MINIO_RAW_PREFIX
MX8_MIX_AUTOTUNE_PERIOD_TICKS
MX8_MIX_DDP_STEPS
MX8_MIX_GATE_CHECK_EXHAUSTION
MX8_MIX_GATE_EPOCH
MX8_MIX_GATE_EPOCH_DRIFT_WEIGHTS
MX8_MIX_GATE_EXPECT_EPOCH_DRIFT
MX8_MIX_GATE_MAX_INFLIGHT_BYTES
MX8_MIX_GATE_MAX_PROCESS_RSS_BYTES
MX8_MIX_GATE_RATIO_TOL
MX8_MIX_GATE_SEED
MX8_MIX_GATE_STEPS
MX8_MIX_GATE_STRICT
MX8_MIX_GATE_WEIGHTS
MX8_MIX_SNAPSHOT
MX8_MIX_SNAPSHOT_PERIOD_TICKS
MX8_NODE_ID
MX8_PACK_IN
MX8_PACK_OUT
MX8_PACK_REQUIRE_LABELS
MX8_PACK_SHARD_MB
MX8_PREFETCH_BATCHES
MX8_PREFETCH_COMPARE
MX8_PROGRESS_INTERVAL_MS
MX8_PY_SMOKE_INSTALL_TORCH
MX8_RESOLVE_PROCS
MX8_RUST_JPEG_CODEC
MX8_RUST_RESIZE_BACKEND
MX8_S3_ENDPOINT_URL
MX8_S3_FORCE_PATH_STYLE
MX8_S3_LABEL_MODE
MX8_S3_SCAN_SHUFFLE_SEED
MX8_SCALE_KEY_PAD_BYTES
MX8_SEED
MX8_SEED_FILE
MX8_SHUFFLE
MX8_SINK_SLEEP_MS
MX8_SMOKE_AUTOTUNE_PRESSURE_SIM
MX8_SMOKE_DEMO2_MINIO
MX8_SMOKE_DEMO2_MINIO_SCALE
MX8_SMOKE_MINIO
MX8_SMOKE_MINIO_MANIFEST_STORE
MX8_SMOKE_MINIO_PACK
MX8_SMOKE_MINIO_S3_PREFIX_RECURSIVE
MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT
MX8_SMOKE_MINIO_VIDEO_STAGE1
MX8_SMOKE_MIX
MX8_SMOKE_MIX_MULTIRANK
MX8_SMOKE_MIX_STRICT
MX8_SMOKE_OFFLINE
MX8_SMOKE_PROD_READINESS
MX8_SMOKE_PY_IMAGE_DECODE_BENCH
MX8_SMOKE_PY_IMAGE_PILLOW
MX8_SMOKE_PY_LOCAL_IMAGE_PILLOW
MX8_SMOKE_PY_MINIO_IMAGE
MX8_SMOKE_SOAK_DEMO2_MINIO_SCALE
MX8_SMOKE_TORCH_DDP
MX8_SMOKE_TORCH_DDP_AUTOTUNE_AB
MX8_SMOKE_TORCH_DDP_DETERMINISM
MX8_SMOKE_TORCH_DDP_NODUPES
MX8_SMOKE_TORCH_DDP_RESTART
MX8_SMOKE_TUI
MX8_SMOKE_VIDEO_GA
MX8_SMOKE_VIDEO_GA_MODE
MX8_SMOKE_VIDEO_STAGE1
MX8_SMOKE_VIDEO_STAGE2A
MX8_SMOKE_VIDEO_STAGE2B
MX8_SMOKE_VIDEO_STAGE2B_CLEAN_ENV
MX8_SMOKE_VIDEO_STAGE2B_STRESS
MX8_SMOKE_VIDEO_STAGE2C_PERF
MX8_SMOKE_VIDEO_STAGE2D_RANGE
MX8_SMOKE_VIDEO_STAGE3A
MX8_SNAPSHOT_LOCK_STALE_MS
MX8_SNAPSHOT_RECURSIVE
MX8_SNAPSHOT_S3_EXTERNAL_SORT
MX8_SNAPSHOT_S3_SPILL_KEYS_PER_RUN
MX8_SNAPSHOT_WAIT_TIMEOUT_MS
MX8_START_ID
MX8_TARGET_BATCH_BYTES
MX8_TORCH_AB_COMPUTE_MS
MX8_TORCH_AB_MAX_INFLIGHT_BYTES
MX8_TORCH_AB_MAX_PROCESS_RSS_BYTES
MX8_TORCH_AB_MAX_QUEUE_BATCHES
MX8_TORCH_AB_MIN_WAIT_IMPROVEMENT
MX8_TORCH_AB_PREFETCH_BATCHES
MX8_TORCH_AB_STEPS
MX8_TORCH_AB_WANT
MX8_TORCH_BATCH_SIZE_SAMPLES
MX8_TORCH_CRASH_AFTER_STEP
MX8_TORCH_CRASH_RANK
MX8_TORCH_DDP_AUTOTUNE_AB
MX8_TORCH_DDP_DETERMINISM
MX8_TORCH_DDP_NODUPES
MX8_TORCH_DDP_RESTART
MX8_TORCH_DIST_TIMEOUT_S
MX8_TORCH_LOAD_CKPT_PATH
MX8_TORCH_LR
MX8_TORCH_NODUPES_STEPS
MX8_TORCH_RESTART_LOAD_STEPS
MX8_TORCH_RESTART_STEPS
MX8_TORCH_SAVE_CKPT_PATH
MX8_TORCH_SAVE_CKPT_STEP
MX8_TORCH_STEPS
MX8_TOTAL_SAMPLES
MX8_TRAIN_STEPS
MX8_TUI_GATE_COORD_PORT
MX8_TUI_HEADLESS_POLLS
MX8_TUI_LEASE_STALL_MS
MX8_TUI_MANIFEST_HASH
MX8_TUI_MANIFEST_PATH
MX8_TUI_POLL_MS
MX8_TUI_ROWS_PER_PAGE
MX8_TUI_SEARCH
MX8_TUI_STALE_HEARTBEAT_MS
MX8_VIDEO_AUTOTUNE_PERIOD_BATCHES
MX8_VIDEO_DECODE_BACKEND
MX8_VIDEO_STAGE1_BYTES_PER_FRAME_ESTIMATE
MX8_VIDEO_STAGE1_DISABLE_FFPROBE
MX8_VIDEO_STAGE1_INDEX
MX8_VIDEO_STAGE2A_BATCH_SIZE
MX8_VIDEO_STAGE2A_CLIP_LEN
MX8_VIDEO_STAGE2A_EPOCH
MX8_VIDEO_STAGE2A_FPS
MX8_VIDEO_STAGE2A_MAX_BATCHES
MX8_VIDEO_STAGE2A_SEED
MX8_VIDEO_STAGE2A_STRIDE
MX8_VIDEO_STAGE2A_TMP_ROOT
MX8_VIDEO_STAGE2B_BATCH_SIZE
MX8_VIDEO_STAGE2B_CLIP_LEN
MX8_VIDEO_STAGE2B_EPOCH
MX8_VIDEO_STAGE2B_FPS
MX8_VIDEO_STAGE2B_MAX_BATCHES
MX8_VIDEO_STAGE2B_SEED
MX8_VIDEO_STAGE2B_STRESS_BATCH_SIZE
MX8_VIDEO_STAGE2B_STRESS_BYTES_PER_CLIP
MX8_VIDEO_STAGE2B_STRESS_CLIP_LEN
MX8_VIDEO_STAGE2B_STRESS_EPOCH
MX8_VIDEO_STAGE2B_STRESS_FPS
MX8_VIDEO_STAGE2B_STRESS_MAX_BATCHES
MX8_VIDEO_STAGE2B_STRESS_MAX_INFLIGHT_BYTES
MX8_VIDEO_STAGE2B_STRESS_SECONDS
MX8_VIDEO_STAGE2B_STRESS_SEED
MX8_VIDEO_STAGE2B_STRESS_STRIDE
MX8_VIDEO_STAGE2B_STRESS_TMP_ROOT
MX8_VIDEO_STAGE2B_STRESS_VIDEO_COUNT
MX8_VIDEO_STAGE2B_STRIDE
MX8_VIDEO_STAGE2B_TMP_ROOT
MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_BATCH
MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_CLIP
MX8_VIDEO_STAGE2C_MIN_SAMPLES_PER_SEC
MX8_VIDEO_STAGE2D_MAX_RANGES
MX8_VIDEO_STAGE2D_MERGE_GAP_BYTES
MX8_VIDEO_STAGE2_BYTES_PER_CLIP
MX8_VIDEO_STAGE2_MAX_CLIPS_IN_MEMORY
MX8_VIDEO_STAGE3A_BATCH_SIZE
MX8_VIDEO_STAGE3A_CLIP_LEN
MX8_VIDEO_STAGE3A_EPOCH
MX8_VIDEO_STAGE3A_FPS
MX8_VIDEO_STAGE3A_MAX_BATCHES
MX8_VIDEO_STAGE3A_SEED
MX8_VIDEO_STAGE3A_STRIDE
MX8_VIDEO_STAGE3A_TMP_ROOT
MX8_WAIT_DRAIN_TIMEOUT_MS
MX8_WAIT_REQUEUE_TIMEOUT_MS
MX8_WORLD_SIZE
MX8_ZERO_MANIFEST_ENABLED
MX8_ZERO_MANIFEST_RESERVOIR
```

## Notes

- If a variable appears in this inventory but is not in the `stable`/`experimental` tables, treat it as `internal`.
- When behavior or defaults change, this page must be updated in the same change.
