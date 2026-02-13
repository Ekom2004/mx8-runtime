#!/usr/bin/env bash
set -euo pipefail

# Internal gating demo: exercises multi-node leases + kill-a-node recovery on localhost.
#
# This is NOT a launch demo. It exists to prove:
# - leases drive real work
# - cursor advances only after deliver
# - lease expiry requeues remainder [cursor,end)
# - no overlapping live leases
#
# Usage:
#   ./scripts/demo2_local.sh
#   WORLD_SIZE=4 ./scripts/demo2_local.sh
#
# Logs + artifacts are written to a temp directory printed at the end.

if ! command -v python3 >/dev/null 2>&1; then
  echo "[demo2] python3 not found" >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-2}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-80000}"
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-10000}"

BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES:-512}"
SINK_SLEEP_MS="${SINK_SLEEP_MS:-25}"
MAX_INFLIGHT_BYTES="${MAX_INFLIGHT_BYTES:-33554432}" # 32 MiB
MAX_QUEUE_BATCHES="${MAX_QUEUE_BATCHES:-64}"

LEASE_TTL_MS="${LEASE_TTL_MS:-2000}"
HEARTBEAT_INTERVAL_MS="${HEARTBEAT_INTERVAL_MS:-200}"
PROGRESS_INTERVAL_MS="${PROGRESS_INTERVAL_MS:-200}"

KILL_NODE_INDEX="${KILL_NODE_INDEX:-1}"       # 1-based index
KILL_AFTER_MS="${KILL_AFTER_MS:-750}"         # kill during first leases
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS:-15000}"
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS:-60000}"

JOB_ID="${JOB_ID:-demo2}"
DATASET_LINK="${DATASET_LINK:-demo://demo2/}"
COORD_PORT="${COORD_PORT:-}"

ts_ms() {
  python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

sleep_ms() {
  python3 - "$1" <<'PY'
import sys, time
time.sleep(int(sys.argv[1]) / 1000.0)
PY
}

log_has_line_with_all() {
  # Usage: log_has_line_with_all <path> <substr1> [<substr2> ...]
  python3 - "$@" <<'PY'
import sys

path = sys.argv[1]
need = sys.argv[2:]

try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if all(s in line for s in need):
                sys.exit(0)
except FileNotFoundError:
    sys.exit(1)

sys.exit(1)
PY
}

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-demo2-local.XXXXXX")"
STORE_ROOT="${TMP_ROOT}/store"
CACHE_DIR="${TMP_ROOT}/cache"
DATA_FILE="${TMP_ROOT}/data.bin"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"
COORD_LOG="${TMP_ROOT}/coordinator.log"

cleanup() {
  set +e
  if [[ -n "${AGENT_PIDS:-}" ]]; then
    for pid in ${AGENT_PIDS}; do
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
  fi
  if [[ -n "${COORD_PID:-}" ]]; then
    kill -TERM "${COORD_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[demo2] tmp_root=${TMP_ROOT}"

mkdir -p "${STORE_ROOT}" "${CACHE_DIR}"

echo "[demo2] creating data file (${TOTAL_SAMPLES} samples, ${BYTES_PER_SAMPLE} bytes/sample)"
total_bytes="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
python3 - "${DATA_FILE}" "${total_bytes}" <<'PY'
import sys
path = sys.argv[1]
size = int(sys.argv[2])
with open(path, "wb") as f:
    f.truncate(size)
PY

echo "[demo2] writing dev manifest (${DEV_MANIFEST})"
{
  awk -v n="${TOTAL_SAMPLES}" -v p="${DATA_FILE}" -v b="${BYTES_PER_SAMPLE}" 'BEGIN{
    for (i=0;i<n;i++) {
      printf "%d\t%s\t%d\t%d\n", i, p, i*b, b
    }
  }'
} > "${DEV_MANIFEST}"

echo "[demo2] building binaries"
cargo build -p mx8-coordinator -p mx8d-agent >/dev/null

if [[ -z "${COORD_PORT}" ]]; then
  COORD_PORT="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi

echo "[demo2] starting coordinator"
MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
MX8_WORLD_SIZE="${WORLD_SIZE}" \
MX8_HEARTBEAT_INTERVAL_MS="${HEARTBEAT_INTERVAL_MS}" \
MX8_LEASE_TTL_MS="${LEASE_TTL_MS}" \
MX8_DATASET_LINK="${DATASET_LINK}@refresh" \
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
MX8_DEV_BLOCK_SIZE="${BLOCK_SIZE}" \
MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
target/debug/mx8-coordinator --world-size "${WORLD_SIZE}" >"${COORD_LOG}" 2>&1 &
COORD_PID="$!"

sleep 1

AGENT_PIDS=""
for i in $(seq 1 "${WORLD_SIZE}"); do
  node_id="node${i}"
  log="${TMP_ROOT}/agent_${node_id}.log"
  echo "[demo2] starting agent ${node_id}"
  MX8_COORD_URL="http://127.0.0.1:${COORD_PORT}" \
  MX8_JOB_ID="${JOB_ID}" \
  MX8_NODE_ID="${node_id}" \
  MX8_MANIFEST_CACHE_DIR="${CACHE_DIR}" \
  MX8_DEV_LEASE_WANT=1 \
  MX8_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
  MX8_MAX_QUEUE_BATCHES="${MAX_QUEUE_BATCHES}" \
  MX8_MAX_INFLIGHT_BYTES="${MAX_INFLIGHT_BYTES}" \
  MX8_SINK_SLEEP_MS="${SINK_SLEEP_MS}" \
  MX8_PROGRESS_INTERVAL_MS="${PROGRESS_INTERVAL_MS}" \
  MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
  target/debug/mx8d-agent >"${log}" 2>&1 &
  AGENT_PIDS="${AGENT_PIDS} $!"
done

sleep_ms "${KILL_AFTER_MS}"

kill_pid="$(echo "${AGENT_PIDS}" | awk -v idx="${KILL_NODE_INDEX}" '{print $idx}')"
echo "[demo2] killing node index=${KILL_NODE_INDEX} pid=${kill_pid}"
kill -KILL "${kill_pid}" >/dev/null 2>&1 || true

echo "[demo2] waiting for coordinator to emit range_requeued"
start_ms="$(ts_ms)"
while true; do
  if log_has_line_with_all "${COORD_LOG}" 'event="range_requeued"'; then
    echo "[demo2] saw range_requeued"
    break
  fi
  now_ms="$(ts_ms)"
  if (( now_ms - start_ms > WAIT_REQUEUE_TIMEOUT_MS )); then
    echo "[demo2] timeout waiting for range_requeued; see ${COORD_LOG}"
    exit 1
  fi
  sleep_ms 200
done

echo "[demo2] waiting for coordinator to emit job_drained"
start_ms="$(ts_ms)"
while true; do
  if log_has_line_with_all "${COORD_LOG}" 'event="job_drained"'; then
    echo "[demo2] saw job_drained"
    break
  fi
  now_ms="$(ts_ms)"
  if (( now_ms - start_ms > WAIT_DRAIN_TIMEOUT_MS )); then
    echo "[demo2] timeout waiting for job_drained; see ${COORD_LOG}"
    exit 1
  fi
  sleep_ms 500
done

echo "[demo2] done"
echo "[demo2] artifacts: ${TMP_ROOT}"
echo "[demo2] coordinator log: ${COORD_LOG}"
