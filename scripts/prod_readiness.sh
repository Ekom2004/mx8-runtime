#!/usr/bin/env bash
set -euo pipefail

# Production readiness gate for MX8 v1.8.
#
# This script is intentionally opinionated and deterministic:
# - no-overlap/recovery correctness
# - memory safety (inflight + process RSS cap behavior)
# - jitter/autotune contract checks
# - multi-node soak under repeated failure
#
# Usage:
#   ./scripts/prod_readiness.sh
#
# This gate intentionally uses fixed thresholds. If you need to tune parameters,
# run the underlying scripts directly (not this locked readiness suite).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

readonly DEMO2_TOTAL_SAMPLES=12000
readonly DEMO2_BLOCK_SIZE=1000
readonly DEMO2_SINK_SLEEP_MS=25
readonly DEMO2_KILL_AFTER_MS=750
readonly DEMO2_WAIT_REQUEUE_TIMEOUT_MS=60000
readonly DEMO2_WAIT_DRAIN_TIMEOUT_MS=120000

readonly SOAK_WORLD_SIZE=8
readonly SOAK_TOTAL_SAMPLES=400000
readonly SOAK_KILL_COUNT=2
readonly SOAK_KILL_INTERVAL_MS=30000
readonly SOAK_WAIT_DRAIN_TIMEOUT_MS=600000
readonly SOAK_LEASE_TTL_MS=30000

readonly DDP_WORLD_SIZE=4

echo "[mx8] prod_readiness: static checks"
./scripts/check.sh

echo "[mx8] prod_readiness: jitter guardrail"
cargo test -p mx8-runtime jitter_guardrail

echo "[mx8] prod_readiness: process RSS cap fail-fast"
cargo test -p mx8-runtime process_rss_cap_fails_fast_when_too_small

echo "[mx8] prod_readiness: demo2 local lease recovery"
RUST_LOG=info \
  MX8_TOTAL_SAMPLES="${DEMO2_TOTAL_SAMPLES}" \
  MX8_DEV_BLOCK_SIZE="${DEMO2_BLOCK_SIZE}" \
  MX8_SINK_SLEEP_MS="${DEMO2_SINK_SLEEP_MS}" \
  MX8_KILL_AFTER_MS="${DEMO2_KILL_AFTER_MS}" \
  MX8_WAIT_REQUEUE_TIMEOUT_MS="${DEMO2_WAIT_REQUEUE_TIMEOUT_MS}" \
  MX8_WAIT_DRAIN_TIMEOUT_MS="${DEMO2_WAIT_DRAIN_TIMEOUT_MS}" \
  cargo run -p mx8-runtime --bin mx8-demo2

echo "[mx8] prod_readiness: deterministic autotune pressure simulation"
./scripts/autotune_memory_pressure_sim.sh

if ! command -v docker >/dev/null 2>&1; then
  echo "[mx8] prod_readiness failed: docker is required for MinIO soak gate" >&2
  exit 1
fi

echo "[mx8] prod_readiness: multi-node soak + repeated requeue"
WORLD_SIZE="${SOAK_WORLD_SIZE}" \
TOTAL_SAMPLES="${SOAK_TOTAL_SAMPLES}" \
KILL_COUNT="${SOAK_KILL_COUNT}" \
KILL_INTERVAL_MS="${SOAK_KILL_INTERVAL_MS}" \
WAIT_DRAIN_TIMEOUT_MS="${SOAK_WAIT_DRAIN_TIMEOUT_MS}" \
LEASE_TTL_MS="${SOAK_LEASE_TTL_MS}" \
./scripts/soak_demo2_minio_scale.sh

echo "[mx8] prod_readiness: distributed no-overlap gate"
WORLD_SIZE="${DDP_WORLD_SIZE}" \
MX8_TORCH_DDP_NODUPES=1 \
./scripts/torch_ddp_gate.sh

echo "[mx8] prod_readiness OK"
