#!/usr/bin/env bash
set -euo pipefail

# Internal gating demo: exercises multi-node leases + kill-a-node recovery while agents fetch
# real bytes from an S3-compatible store (MinIO).
#
# Usage:
#   ./scripts/demo2_minio.sh
#   WORLD_SIZE=4 ./scripts/demo2_minio.sh
#
# Prereqs:
# - docker (daemon running)
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "[demo2_minio] docker not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[demo2_minio] docker daemon not running" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[demo2_minio] curl not found" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[demo2_minio] python3 not found" >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-2}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8000}"
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-1000}"

BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES:-512}"
SINK_SLEEP_MS="${SINK_SLEEP_MS:-25}"
MAX_INFLIGHT_BYTES="${MAX_INFLIGHT_BYTES:-33554432}" # 32 MiB
MAX_QUEUE_BATCHES="${MAX_QUEUE_BATCHES:-64}"
DEV_LEASE_WANT="${MX8_DEV_LEASE_WANT:-1}"

LEASE_TTL_MS="${LEASE_TTL_MS:-2000}"
HEARTBEAT_INTERVAL_MS="${HEARTBEAT_INTERVAL_MS:-200}"
PROGRESS_INTERVAL_MS="${PROGRESS_INTERVAL_MS:-200}"

KILL_NODE_INDEX="${KILL_NODE_INDEX:-1}"       # 1-based index; set to 0 to auto-pick first lease holder
KILL_AFTER_MS="${KILL_AFTER_MS:-750}"         # kill during first leases
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS:-15000}"
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS:-15000}"
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS:-60000}"
WAIT_COORD_READY_TIMEOUT_MS="${WAIT_COORD_READY_TIMEOUT_MS:-30000}"

JOB_ID="${JOB_ID:-demo2-minio}"
DATASET_LINK="${DATASET_LINK:-s3://mx8-demo/data.bin}"
COORD_PORT="${COORD_PORT:-}"

MINIO_IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
MINIO_PORT="${MX8_MINIO_PORT:-9000}"
MINIO_CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
MINIO_NAME="mx8-demo2-minio-$$"

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
import re

path = sys.argv[1]
need = sys.argv[2:]
ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = ansi.sub("", line)
            if all(s in line for s in need):
                sys.exit(0)
except FileNotFoundError:
    sys.exit(1)

sys.exit(1)
PY
}

log_find_first_lease_granted_node_id() {
  # Usage: log_find_first_lease_granted_node_id <path>
  # Prints the first observed node_id for event="lease_granted", or nothing if none found yet.
  python3 - "$@" <<'PY'
import sys
import re

path = sys.argv[1]
ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ansi.sub("", raw)
            if 'event="lease_granted"' not in line:
                continue
            m = re.search(r"\bnode_id=([^\s]+)\b", line)
            if m:
                sys.stdout.write(m.group(1))
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

log_has_lease_granted_for_node_exact() {
  # Usage: log_has_lease_granted_for_node_exact <path> <node_id>
  python3 - "$@" <<'PY'
import sys
import re

path = sys.argv[1]
node_id = sys.argv[2]
ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
needle = re.compile(rf'event="lease_granted".*\\bnode_id={re.escape(node_id)}\\b')

try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ansi.sub("", raw)
            if needle.search(line):
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

wait_for_tcp_listen() {
  # Usage: wait_for_tcp_listen <host> <port> <timeout_ms>
  python3 - "$@" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_ms = int(sys.argv[3])

deadline = time.time() + (timeout_ms / 1000.0)
last = None
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        s.close()
        sys.exit(0)
    except Exception as e:
        last = e
    finally:
        try:
            s.close()
        except Exception:
            pass
    time.sleep(0.1)

raise SystemExit(f"timeout waiting for {host}:{port} to accept TCP connections (last_error={last})")
PY
}

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-demo2-minio.XXXXXX")"
STORE_ROOT="${TMP_ROOT}/store"
CACHE_DIR="${TMP_ROOT}/cache"
DATA_FILE="${TMP_ROOT}/data.bin"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"
COORD_LOG="${TMP_ROOT}/coordinator.log"

cleanup() {
  set +e
  docker rm -f "${MINIO_NAME}" >/dev/null 2>&1 || true
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

echo "[demo2_minio] tmp_root=${TMP_ROOT}"
mkdir -p "${STORE_ROOT}" "${CACHE_DIR}"

echo "[demo2_minio] starting minio container (${MINIO_IMAGE})"
docker run -d --name "${MINIO_NAME}" \
  -p "127.0.0.1:${MINIO_PORT}:9000" \
  -p "127.0.0.1:${MINIO_CONSOLE_PORT}:9001" \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  "${MINIO_IMAGE}" server /data --address ":9000" --console-address ":9001" >/dev/null

READY_URL="http://127.0.0.1:${MINIO_PORT}/minio/health/ready"
echo "[demo2_minio] waiting for minio ready (${READY_URL})"
for _ in $(seq 1 50); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "${READY_URL}" >/dev/null 2>&1 || (echo "[demo2_minio] minio not ready" >&2 && exit 1)

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=1

export MX8_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_PORT}"
export MX8_S3_FORCE_PATH_STYLE=1

BUCKET="${MX8_MINIO_BUCKET:-mx8-demo}"
KEY="${MX8_MINIO_KEY:-data.bin}"

MANIFEST_STORE_BUCKET="${MX8_MANIFEST_STORE_BUCKET:-mx8-manifests}"
MANIFEST_STORE_PREFIX_DEFAULT="demo2-${MINIO_NAME}"
MANIFEST_STORE_PREFIX="${MX8_MANIFEST_STORE_PREFIX:-${MANIFEST_STORE_PREFIX_DEFAULT}}"
MANIFEST_STORE_ROOT_DEFAULT="s3://${MANIFEST_STORE_BUCKET}/${MANIFEST_STORE_PREFIX}"
MANIFEST_STORE_ROOT="${MX8_MANIFEST_STORE_ROOT:-${MANIFEST_STORE_ROOT_DEFAULT}}"

echo "[demo2_minio] creating local data file (${TOTAL_SAMPLES} samples, ${BYTES_PER_SAMPLE} bytes/sample)"
total_bytes="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
python3 - "${DATA_FILE}" "${total_bytes}" <<'PY'
import sys
path = sys.argv[1]
size = int(sys.argv[2])
with open(path, "wb") as f:
    f.truncate(size)
PY

echo "[demo2_minio] seeding s3://bucket/key via mx8-seed-s3"
MX8_MINIO_BUCKET="${BUCKET}" \
MX8_MINIO_KEY="${KEY}" \
MX8_SEED_FILE="${DATA_FILE}" \
cargo run -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null

echo "[demo2_minio] writing dev manifest (${DEV_MANIFEST})"
{
  awk -v n="${TOTAL_SAMPLES}" -v b="${BUCKET}" -v k="${KEY}" -v bytes="${BYTES_PER_SAMPLE}" 'BEGIN{
    for (i=0;i<n;i++) {
      printf "%d\ts3://%s/%s\t%d\t%d\n", i, b, k, i*bytes, bytes
    }
  }'
} > "${DEV_MANIFEST}"

echo "[demo2_minio] building binaries (coordinator+agent with s3 feature)"
cargo build -p mx8-coordinator --features s3 >/dev/null
cargo build -p mx8d-agent --features s3 >/dev/null

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

echo "[demo2_minio] starting coordinator"
MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
MX8_WORLD_SIZE="${WORLD_SIZE}" \
MX8_HEARTBEAT_INTERVAL_MS="${HEARTBEAT_INTERVAL_MS}" \
MX8_LEASE_TTL_MS="${LEASE_TTL_MS}" \
MX8_DATASET_LINK="s3://${BUCKET}/${KEY}@refresh" \
MX8_MANIFEST_STORE_ROOT="${MANIFEST_STORE_ROOT}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
MX8_DEV_BLOCK_SIZE="${BLOCK_SIZE}" \
MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
target/debug/mx8-coordinator --world-size "${WORLD_SIZE}" >"${COORD_LOG}" 2>&1 &
COORD_PID="$!"

wait_for_tcp_listen 127.0.0.1 "${COORD_PORT}" "${WAIT_COORD_READY_TIMEOUT_MS}"

AGENT_PIDS=""
for i in $(seq 1 "${WORLD_SIZE}"); do
  node_id="node${i}"
  log="${TMP_ROOT}/agent_${node_id}.log"
  echo "[demo2_minio] starting agent ${node_id}"
	  MX8_COORD_URL="http://127.0.0.1:${COORD_PORT}" \
	  MX8_JOB_ID="${JOB_ID}" \
	  MX8_NODE_ID="${node_id}" \
	  MX8_MANIFEST_CACHE_DIR="${CACHE_DIR}" \
	  MX8_DEV_LEASE_WANT="${DEV_LEASE_WANT}" \
	  MX8_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
	  MX8_MAX_QUEUE_BATCHES="${MAX_QUEUE_BATCHES}" \
	  MX8_MAX_INFLIGHT_BYTES="${MAX_INFLIGHT_BYTES}" \
  MX8_SINK_SLEEP_MS="${SINK_SLEEP_MS}" \
  MX8_PROGRESS_INTERVAL_MS="${PROGRESS_INTERVAL_MS}" \
  MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
  target/debug/mx8d-agent >"${log}" 2>&1 &
  AGENT_PIDS="${AGENT_PIDS} $!"
done

desired_kill_node_id=""
if [[ "${KILL_NODE_INDEX}" != "0" ]]; then
  desired_kill_node_id="node${KILL_NODE_INDEX}"
fi
kill_node_id="${desired_kill_node_id}"
if [[ -n "${desired_kill_node_id}" ]]; then
  echo "[demo2_minio] waiting for first lease_granted to ${desired_kill_node_id}"
else
  echo "[demo2_minio] waiting for first lease_granted (auto kill target)"
fi
start_ms="$(ts_ms)"
while true; do
  if [[ -n "${desired_kill_node_id}" ]]; then
    if log_has_lease_granted_for_node_exact "${COORD_LOG}" "${desired_kill_node_id}"; then
      echo "[demo2_minio] saw first lease_granted for ${desired_kill_node_id}"
      break
    fi
  else
    if kill_node_id="$(log_find_first_lease_granted_node_id "${COORD_LOG}")"; then
      echo "[demo2_minio] selected kill_node_id=${kill_node_id}"
      KILL_NODE_INDEX="$(echo "${kill_node_id}" | sed -E 's/^node//')"
      if [[ -z "${KILL_NODE_INDEX}" ]]; then
        echo "[demo2_minio] could not parse kill index from ${kill_node_id}" >&2
        exit 1
      fi
      break
    fi
  fi
  now_ms="$(ts_ms)"
  if (( now_ms - start_ms > WAIT_FIRST_LEASE_TIMEOUT_MS )); then
    if [[ -n "${desired_kill_node_id}" ]]; then
      echo "[demo2_minio] timeout waiting for first lease_granted for ${desired_kill_node_id}; falling back to first lease_granted node; see ${COORD_LOG}" >&2
      if kill_node_id="$(log_find_first_lease_granted_node_id "${COORD_LOG}")"; then
        echo "[demo2_minio] fallback kill_node_id=${kill_node_id}" >&2
        KILL_NODE_INDEX="$(echo "${kill_node_id}" | sed -E 's/^node//')"
        if [[ -z "${KILL_NODE_INDEX}" ]]; then
          echo "[demo2_minio] could not parse kill index from ${kill_node_id}" >&2
          exit 1
        fi
        break
      fi
    fi
    echo "[demo2_minio] could not find any lease_granted in coordinator log; aborting" >&2
    exit 1
  fi
  sleep_ms 100
done

sleep_ms "${KILL_AFTER_MS}"

kill_pid="$(echo "${AGENT_PIDS}" | awk -v idx="${KILL_NODE_INDEX}" '{print $idx}')"
echo "[demo2_minio] killing node_id=${kill_node_id} index=${KILL_NODE_INDEX} pid=${kill_pid}"
kill_time_ms="$(ts_ms)"
kill -KILL "${kill_pid}" >/dev/null 2>&1 || true

echo "[demo2_minio] waiting for coordinator to emit range_requeued"
start_ms="$(ts_ms)"
while true; do
  if log_has_line_with_all "${COORD_LOG}" 'event="range_requeued"'; then
    echo "[demo2_minio] saw range_requeued"
    break
  fi
  now_ms="$(ts_ms)"
  if (( now_ms - start_ms > WAIT_REQUEUE_TIMEOUT_MS )); then
    echo "[demo2_minio] timeout waiting for range_requeued; see ${COORD_LOG}"
    exit 1
  fi
  sleep_ms 200
done

echo "[demo2_minio] waiting for coordinator to emit job_drained"
start_ms="$(ts_ms)"
while true; do
  if log_has_line_with_all "${COORD_LOG}" 'event="job_drained"'; then
    echo "[demo2_minio] saw job_drained"
    break
  fi
  now_ms="$(ts_ms)"
  if (( now_ms - start_ms > WAIT_DRAIN_TIMEOUT_MS )); then
    echo "[demo2_minio] timeout waiting for job_drained; see ${COORD_LOG}"
    exit 1
  fi
  sleep_ms 500
done

print_coordinator_summary() {
  # Usage: print_coordinator_summary <coord_log_path> <kill_node_id> <kill_time_ms>
  python3 - "$@" <<'PY'
import sys
import re
from datetime import datetime, timezone
from typing import Optional

coord_log = sys.argv[1]
kill_node_id = sys.argv[2]
kill_time_ms = int(sys.argv[3])

ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

def parse_line_time_ms(line: str) -> Optional[int]:
    # Example:
    # 2026-02-13T12:05:09.564516Z  INFO ... event="lease_granted" ...
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+", line)
    if not m:
        return None
    ts = m.group(1)
    # Python's fromisoformat doesn't accept trailing 'Z'.
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)

events = {
    "lease_granted": 0,
    "lease_completed": 0,
    "lease_expired": 0,
    "range_requeued": 0,
    "progress": 0,
}

first_grant_ms: Optional[int] = None
drained_ms: Optional[int] = None
expired_ms: Optional[int] = None
requeued_ms: Optional[int] = None
kill_node_re = re.compile(rf"\bnode_id={re.escape(kill_node_id)}\b")

with open(coord_log, "r", encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = ansi.sub("", raw)
        t = parse_line_time_ms(line)

        if 'event="lease_granted"' in line:
            events["lease_granted"] += 1
            if first_grant_ms is None and t is not None:
                first_grant_ms = t

        if 'event="lease_completed"' in line:
            events["lease_completed"] += 1

        if 'event="lease_expired"' in line:
            events["lease_expired"] += 1
            if expired_ms is None and kill_node_re.search(line) and t is not None:
                expired_ms = t

        if 'event="range_requeued"' in line:
            events["range_requeued"] += 1
            if requeued_ms is None and kill_node_re.search(line) and t is not None:
                requeued_ms = t

        # Coordinator proof logs use event="progress" for accepted progress reports.
        if 'event="progress"' in line:
            events["progress"] += 1

        if 'event="job_drained"' in line and t is not None:
            drained_ms = t

def fmt(ms: Optional[int]) -> str:
    return "na" if ms is None else str(ms)

grant_to_drained = None if (first_grant_ms is None or drained_ms is None) else (drained_ms - first_grant_ms)
kill_to_expired = None if expired_ms is None else (expired_ms - kill_time_ms)
expired_to_requeue = None if (expired_ms is None or requeued_ms is None) else (requeued_ms - expired_ms)
requeue_to_drained = None if (requeued_ms is None or drained_ms is None) else (drained_ms - requeued_ms)

print(
    "[demo2_minio] coordinator_summary "
    f"kill_node_id={kill_node_id} "
    f"counts=lease_granted:{events['lease_granted']},"
    f"lease_completed:{events['lease_completed']},"
    f"lease_expired:{events['lease_expired']},"
    f"range_requeued:{events['range_requeued']},"
    f"progress:{events['progress']} "
    f"durations_ms=first_grant_to_drained:{fmt(grant_to_drained)},"
    f"kill_to_expired:{fmt(kill_to_expired)},"
    f"expired_to_requeue:{fmt(expired_to_requeue)},"
    f"requeue_to_drained:{fmt(requeue_to_drained)}"
)
PY
}

echo "[demo2_minio] done"
echo "[demo2_minio] artifacts: ${TMP_ROOT}"
echo "[demo2_minio] coordinator log: ${COORD_LOG}"
print_coordinator_summary "${COORD_LOG}" "${kill_node_id}" "${kill_time_ms}"
