#!/usr/bin/env bash
set -euo pipefail

# Accelerated burn-in runner for default direct-stream manifest ingest.
#
# What it does per cycle:
# - runs smoke with MinIO snapshot gate enabled
# - fails fast if smoke fails
# - fails fast if any direct-stream safety/fallback proof event appears:
#   - event="manifest_direct_stream_fallback"
#   - event="manifest_stream_truncated"
#   - event="manifest_schema_mismatch"
#
# Usage:
#   ./scripts/direct_stream_burnin.sh
#   MX8_BURNIN_RUNS=5 ./scripts/direct_stream_burnin.sh
#
# Optional:
#   MX8_BURNIN_LOG_DIR=/tmp/mx8-burnin ./scripts/direct_stream_burnin.sh

RUNS="${MX8_BURNIN_RUNS:-3}"
LOG_DIR="${MX8_BURNIN_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mx8-direct-stream-burnin.XXXXXX")}"
RETRIES="${MX8_BURNIN_RETRIES:-2}"

if [[ "${RUNS}" -lt 1 ]]; then
  echo "[burnin] MX8_BURNIN_RUNS must be >= 1" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[burnin] direct-stream accelerated burn-in"
echo "[burnin] runs=${RUNS}"
echo "[burnin] retries_per_run=${RETRIES}"
echo "[burnin] logs=${LOG_DIR}"

bad_pattern='event="manifest_direct_stream_fallback"|event="manifest_stream_truncated"|event="manifest_schema_mismatch"'

for i in $(seq 1 "${RUNS}"); do
  attempt=1
  while true; do
    log_path="${LOG_DIR}/run-${i}-attempt-${attempt}.log"
    echo "[burnin] run ${i}/${RUNS} attempt ${attempt} starting"

    set +e
    (
      PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
      CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false \
      MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT=1 \
      ./scripts/smoke.sh
    ) 2>&1 | tee "${log_path}"
    rc=$?
    set -e

    if [[ "${rc}" -eq 0 ]]; then
      break
    fi
    if [[ "${attempt}" -ge "${RETRIES}" ]]; then
      echo "[burnin] FAIL: run ${i} failed after ${RETRIES} attempts" >&2
      exit 1
    fi
    echo "[burnin] run ${i} attempt ${attempt} failed (rc=${rc}); retrying"
    attempt=$((attempt + 1))
  done

  if grep -nE "${bad_pattern}" "${log_path}" >/dev/null; then
    echo "[burnin] FAIL: detected direct-stream safety/fallback event in run ${i}" >&2
    grep -nE "${bad_pattern}" "${log_path}" >&2 || true
    exit 1
  fi

  if ! grep -nE "\\[mx8\\] smoke OK" "${log_path}" >/dev/null; then
    echo "[burnin] FAIL: run ${i} did not report smoke OK" >&2
    exit 1
  fi

  echo "[burnin] run ${i}/${RUNS} passed (attempt ${attempt})"
done

echo "[burnin] PASS: ${RUNS}/${RUNS} runs clean"
echo "[burnin] direct-stream safety checks stayed clean across burn-in"
