#!/usr/bin/env bash
set -euo pipefail

# Internal gating demo: run the distributed MinIO demo at a larger scale to prove:
# - manifest proxying works for >4MiB manifests (chunked stream)
# - lease recovery still works
#
# Usage:
#   ./scripts/demo2_minio_scale.sh
#
# Prereqs:
# - docker (daemon running)
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WORLD_SIZE="${WORLD_SIZE:-4}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-200000}"
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-10000}"

SINK_SLEEP_MS="${SINK_SLEEP_MS:-0}"
KILL_AFTER_MS="${KILL_AFTER_MS:-750}"
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS:-30000}"
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS:-30000}"
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS:-240000}"

tmp_log_base="$(mktemp "${TMPDIR:-/tmp}/mx8-demo2-minio-scale.XXXXXX")"
tmp_log="${tmp_log_base}.log"
mv "${tmp_log_base}" "${tmp_log}"
echo "[demo2_minio_scale] running WORLD_SIZE=${WORLD_SIZE} TOTAL_SAMPLES=${TOTAL_SAMPLES} BLOCK_SIZE=${BLOCK_SIZE}"

WORLD_SIZE="${WORLD_SIZE}" \
TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE}" \
BLOCK_SIZE="${BLOCK_SIZE}" \
SINK_SLEEP_MS="${SINK_SLEEP_MS}" \
KILL_AFTER_MS="${KILL_AFTER_MS}" \
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS}" \
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS}" \
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS}" \
./scripts/demo2_minio.sh | tee "${tmp_log}"

artifacts="$(
  python3 - "${tmp_log}" <<'PY'
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "artifacts:" in line:
            print(line.split("artifacts:", 1)[1].strip())
            raise SystemExit(0)
raise SystemExit(1)
PY
)"

echo "[demo2_minio_scale] artifacts=${artifacts}"

python3 - "${artifacts}" "${WORLD_SIZE}" <<'PY'
import os
import re
import sys

root = sys.argv[1]
world_size = int(sys.argv[2])

ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return ansi.sub("", f.read())

def must_not_contain(path: str, needle: str) -> None:
    txt = read_text(path)
    if needle in txt:
        raise SystemExit(f"{path} contained unexpected substring: {needle}")

def find_manifest_cached_bytes(path: str) -> int:
    txt = read_text(path)
    for line in txt.splitlines():
        if 'event="manifest_cached"' in line:
            m = re.search(r"\bmanifest_bytes=(\d+)\b", line)
            if m:
                return int(m.group(1))
    raise SystemExit(f"{path} missing manifest_cached with manifest_bytes=...")

min_expected = 4 * 1024 * 1024 + 1

coord_log = os.path.join(root, "coordinator.log")
must_not_contain(coord_log, "message length too large")
must_not_contain(coord_log, "OutOfRange")

any_ok = False
for i in range(1, world_size + 1):
    agent_log = os.path.join(root, f"agent_node{i}.log")
    must_not_contain(agent_log, "message length too large")
    must_not_contain(agent_log, "OutOfRange")

    b = find_manifest_cached_bytes(agent_log)
    if b >= min_expected:
        any_ok = True

if not any_ok:
    raise SystemExit(
        f"expected at least one agent to cache a >4MiB manifest (>= {min_expected} bytes)"
    )

print("[demo2_minio_scale] OK: large manifest cached without gRPC limit errors")
PY

echo "[demo2_minio_scale] done"
