#!/usr/bin/env bash
set -euo pipefail

# Repo smoke runner: quick "main is healthy" gate.
#
# Usage:
#   ./scripts/smoke.sh
#   MX8_SMOKE_OFFLINE=1 ./scripts/smoke.sh
#
# Notes:
# - Demo 2 is an internal correctness gate (kill-a-node lease recovery).
# - Demo 3 is an internal performance+correctness gate (prefetch hides latency, bounded caps).

OFFLINE="${MX8_SMOKE_OFFLINE:-0}"

if [[ "${OFFLINE}" == "1" ]]; then
  MX8_CHECK_OFFLINE=1 ./scripts/check.sh
else
  ./scripts/check.sh
fi

echo "[mx8] demo2 (leases + kill-a-node recovery)"
RUST_LOG=info \
  MX8_TOTAL_SAMPLES=8000 \
  MX8_DEV_BLOCK_SIZE=1000 \
  MX8_SINK_SLEEP_MS=25 \
  MX8_KILL_AFTER_MS=750 \
  MX8_WAIT_REQUEUE_TIMEOUT_MS=15000 \
  MX8_WAIT_DRAIN_TIMEOUT_MS=30000 \
  cargo run -p mx8-runtime --bin mx8-demo2

echo "[mx8] demo3 (prefetch + retry under injected latency)"
# Keep the demo quick while still exercising retry/backoff + prefetch.
RUST_LOG=warn \
  MX8_HTTP_LATENCY_MS=5 \
  MX8_HTTP_FAIL_EVERY_N=7 \
  MX8_TOTAL_SAMPLES=256 \
  MX8_PREFETCH_COMPARE=8 \
  cargo run -p mx8-runtime --bin mx8-demo3

if [[ "${MX8_SMOKE_MINIO:-0}" == "1" ]]; then
  echo "[mx8] minio_gate (S3-compatible fetch via docker)"
  ./scripts/minio_gate.sh
fi

echo "[mx8] smoke OK"
