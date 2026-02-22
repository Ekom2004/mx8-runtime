#!/usr/bin/env bash
set -euo pipefail

# Soak gate: run the distributed MinIO demo with multiple node kills.
#
# Goal: prove multi-node lease recovery stays correct under repeated failures, and the
# coordinator emits proof logs that we can summarize/replay.
#
# Usage:
#   ./scripts/soak_demo2_minio_scale.sh
#   WORLD_SIZE=16 TOTAL_SAMPLES=1000000 KILL_COUNT=3 KILL_INTERVAL_MS=60000 ./scripts/soak_demo2_minio_scale.sh
#
# Prereqs:
# - docker (daemon running)
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WORLD_SIZE="${WORLD_SIZE:-16}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000000}"
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-10000}"

# Make the run "long enough" and failure-heavy by default. Tune for your machine.
KILL_COUNT="${KILL_COUNT:-3}"
KILL_AFTER_MS="${KILL_AFTER_MS:-0}"
KILL_INTERVAL_MS="${KILL_INTERVAL_MS:-30000}"

SINK_SLEEP_MS="${SINK_SLEEP_MS:-0}"
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS:-60000}"
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS:-60000}"
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS:-600000}" # 10 minutes by default

extract_artifacts() {
  python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "artifacts:" in line:
            print(line.split("artifacts:", 1)[1].strip())
            raise SystemExit(0)
raise SystemExit(1)
PY
}

validate_artifacts() {
  python3 - "$1" "$2" <<'PY'
import os
import re
import sys

root = sys.argv[1]
world_size = int(sys.argv[2])

ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return ansi.sub("", f.read())

def must_contain(path: str, needle: str) -> None:
    txt = read_text(path)
    if needle not in txt:
        raise SystemExit(f"{path} missing required substring: {needle}")

def must_not_contain(path: str, needle: str) -> None:
    txt = read_text(path)
    if needle in txt:
        raise SystemExit(f"{path} contained unexpected substring: {needle}")

coord_log = os.path.join(root, "coordinator.log")
must_contain(coord_log, 'event="job_drained"')
must_contain(coord_log, 'event="no_overlap_ok"')
must_contain(coord_log, 'event="lease_completed"')
must_contain(coord_log, 'event="progress"')
must_not_contain(coord_log, "message length too large")
must_not_contain(coord_log, "OutOfRange")

saw_direct_stream = False
for i in range(1, world_size + 1):
    agent_log = os.path.join(root, f"agent_node{i}.log")
    must_not_contain(agent_log, "message length too large")
    must_not_contain(agent_log, "OutOfRange")
    txt = read_text(agent_log)
    if 'event="lease_started"' in txt and 'source="direct_stream"' in txt:
        saw_direct_stream = True

if not saw_direct_stream:
    raise SystemExit("expected at least one agent to execute a direct-stream lease")

print("[soak] validate OK")
PY
}

tmp_log="$(mktemp "${TMPDIR:-/tmp}/mx8-demo2-minio-soak.XXXXXX")"
echo "[soak] log=${tmp_log}" >&2

WORLD_SIZE="${WORLD_SIZE}" \
TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE}" \
BLOCK_SIZE="${BLOCK_SIZE}" \
SINK_SLEEP_MS="${SINK_SLEEP_MS}" \
KILL_COUNT="${KILL_COUNT}" \
KILL_NODE_INDEX=0 \
KILL_AFTER_MS="${KILL_AFTER_MS}" \
KILL_INTERVAL_MS="${KILL_INTERVAL_MS}" \
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS}" \
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS}" \
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS}" \
./scripts/demo2_minio.sh | tee "${tmp_log}" >&2

artifacts="$(extract_artifacts "${tmp_log}")"
echo "[soak] artifacts=${artifacts}" >&2

validate_artifacts "${artifacts}" "${WORLD_SIZE}" >&2
echo "[soak] OK: WORLD_SIZE=${WORLD_SIZE} TOTAL_SAMPLES=${TOTAL_SAMPLES} KILL_COUNT=${KILL_COUNT}"
