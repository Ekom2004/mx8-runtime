#!/usr/bin/env bash
set -euo pipefail

# Internal gating demo: run the distributed MinIO demo at a larger scale to prove:
# - manifest proxying works for >4MiB manifests (chunked stream)
# - lease recovery still works
#
# Usage:
#   ./scripts/demo2_minio_scale.sh
#   MX8_DEV_LEASE_WANT=4 ./scripts/demo2_minio_scale.sh
#   MX8_BENCH_WANTS=1,4 ./scripts/demo2_minio_scale.sh
#
# Prereqs:
# - docker (daemon running)
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WORLD_SIZE="${WORLD_SIZE:-4}"
# This gate's goal is to validate *large manifest handling* (>4MiB) and recovery, not to
# benchmark S3 throughput on localhost MinIO. Keep the data plane small and inflate the
# manifest size by using a long object key.
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8000}"
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-1000}"
DEV_LEASE_WANT="${MX8_DEV_LEASE_WANT:-1}"
BENCH_WANTS="${MX8_BENCH_WANTS:-}"

SINK_SLEEP_MS="${SINK_SLEEP_MS:-0}"
# Scale gate wants deterministic recovery; kill whichever node gets the first lease to
# avoid waiting for a specific node ID (e.g., node1 vs node11 substring pitfalls).
KILL_NODE_INDEX="${KILL_NODE_INDEX:-0}"
# For scale runs, kill as soon as we observe the first lease_granted so we deterministically
# exercise the "lease expires → remainder requeued" path (avoid racing with fast completion).
KILL_AFTER_MS="${KILL_AFTER_MS:-0}"
WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS:-30000}"
WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS:-30000}"
WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS:-240000}"

# Inflate the manifest size deterministically by default (without increasing sample count).
# Override by setting `MX8_MINIO_KEY` directly if desired.
if [[ -z "${MX8_MINIO_KEY:-}" ]]; then
  KEY_PAD_BYTES="${MX8_SCALE_KEY_PAD_BYTES:-900}"
  # MinIO uses a filesystem backend; individual path segments are typically limited to 255 bytes.
  # Build a deep key with multiple <=240B segments so it's valid, while still inflating the
  # manifest TSV beyond 4MiB.
  export MX8_MINIO_KEY
  MX8_MINIO_KEY="$(python3 - "${KEY_PAD_BYTES}" <<'PY'
import sys

n = int(sys.argv[1])
chunk = 240
segs = []
while n > 0:
    k = min(chunk, n)
    segs.append("x" * k)
    n -= k
print("pad/" + "/".join(segs) + "/data.bin")
PY
)"
fi

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

validate_and_elapsed_ms() {
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

def find_elapsed_ms(path: str) -> int:
    txt = read_text(path)
    lines = txt.splitlines()

    start = None
    for line in lines:
        if 'event="progress"' in line:
            m = re.search(r"\bunix_time_ms=(\d+)\b", line)
            if m:
                start = int(m.group(1))
                break

    end = None
    for line in reversed(lines):
        if 'event="job_drained"' in line:
            m = re.search(r"\bunix_time_ms=(\d+)\b", line)
            if m:
                end = int(m.group(1))
                break

    if start is None:
        raise SystemExit(f"{path} missing progress unix_time_ms")
    if end is None:
        raise SystemExit(f"{path} missing job_drained unix_time_ms")
    if end < start:
        raise SystemExit(f"{path} end < start (end={end}, start={start})")
    return end - start

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

elapsed_ms = find_elapsed_ms(coord_log)
print(elapsed_ms)
PY
}

run_once() {
  local want="$1"
  local tmp_parent="${TMPDIR:-/tmp}"
  tmp_parent="${tmp_parent%/}"
  # macOS/BSD `mktemp` ultimately relies on `mkstemp(3)`, which requires the template
  # to end in `XXXXXX` (no suffix after the Xs). Keep the log extension out of the
  # template for portability.
  local tmp_log
  tmp_log="$(mktemp "${tmp_parent}/mx8-demo2-minio-scale.XXXXXX")"

  echo "[demo2_minio_scale] running want=${want} WORLD_SIZE=${WORLD_SIZE} TOTAL_SAMPLES=${TOTAL_SAMPLES} BLOCK_SIZE=${BLOCK_SIZE}" >&2

  WORLD_SIZE="${WORLD_SIZE}" \
  TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
  BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE}" \
  BLOCK_SIZE="${BLOCK_SIZE}" \
  SINK_SLEEP_MS="${SINK_SLEEP_MS}" \
  KILL_NODE_INDEX="${KILL_NODE_INDEX}" \
  KILL_AFTER_MS="${KILL_AFTER_MS}" \
  WAIT_FIRST_LEASE_TIMEOUT_MS="${WAIT_FIRST_LEASE_TIMEOUT_MS}" \
  WAIT_REQUEUE_TIMEOUT_MS="${WAIT_REQUEUE_TIMEOUT_MS}" \
  WAIT_DRAIN_TIMEOUT_MS="${WAIT_DRAIN_TIMEOUT_MS}" \
  MX8_DEV_LEASE_WANT="${want}" \
  ./scripts/demo2_minio.sh | tee "${tmp_log}" >&2

  local artifacts
  artifacts="$(extract_artifacts "${tmp_log}")"
  echo "[demo2_minio_scale] artifacts=${artifacts}" >&2

  local elapsed_ms
  elapsed_ms="$(validate_and_elapsed_ms "${artifacts}" "${WORLD_SIZE}")"
  echo "[demo2_minio_scale] OK: large manifest cached without gRPC limit errors elapsed_ms=${elapsed_ms} (want=${want})" >&2
  echo "${elapsed_ms}"
}

if [[ -n "${BENCH_WANTS}" ]]; then
  echo "[demo2_minio_scale] bench wants=${BENCH_WANTS}"
  python3 - "${BENCH_WANTS}" <<'PY' >/dev/null
import sys
wants = [w.strip() for w in sys.argv[1].split(",") if w.strip()]
for w in wants:
    int(w)  # validate
PY

  IFS=',' read -r -a wants <<<"${BENCH_WANTS}"
  elapsed_by_want=()
  for w in "${wants[@]}"; do
    w="$(echo "${w}" | tr -d '[:space:]')"
    if [[ -z "${w}" ]]; then
      continue
    fi
    elapsed_by_want+=("${w}:$(run_once "${w}")")
  done

  echo "[demo2_minio_scale] bench results:"
  for entry in "${elapsed_by_want[@]}"; do
    echo "  ${entry}"
  done
else
  run_once "${DEV_LEASE_WANT}" >/dev/null
fi

echo "[demo2_minio_scale] done"
