#!/usr/bin/env bash
set -euo pipefail

# MinIO gate for S3-backed manifest_store.
#
# Proves:
# - concurrent @refresh snapshot resolution elects a single indexer via S3 lock
# - all resolvers observe the same deterministic manifest_hash
#
# Usage:
#   ./scripts/minio_manifest_store_gate.sh
#
# Prereqs:
# - docker (daemon running)
# - curl
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[mx8] docker not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[mx8] docker daemon not running (cannot connect to Docker socket)" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[mx8] curl not found (needed for MinIO health check)" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found" >&2
  exit 1
fi

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-manifest-store-gate-$$"

RESOLVE_PROCS="${MX8_RESOLVE_PROCS:-8}"
TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-200000}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-256}"

MANIFEST_STORE_BUCKET="${MX8_MANIFEST_STORE_BUCKET:-mx8-manifests}"
MANIFEST_STORE_PREFIX="${MX8_MANIFEST_STORE_PREFIX:-manifests}"

DATASET_LINK="${MX8_DATASET_LINK:-s3://mx8-demo/data@refresh}"
LOCK_STALE_MS="${MX8_SNAPSHOT_LOCK_STALE_MS:-60000}"
WAIT_TIMEOUT_MS="${MX8_SNAPSHOT_WAIT_TIMEOUT_MS:-60000}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-minio-manifest-store-gate.XXXXXX")"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[mx8] starting minio container (${IMAGE})"
docker run -d --name "${NAME}" \
  -p "127.0.0.1:${PORT}:9000" \
  -p "127.0.0.1:${CONSOLE_PORT}:9001" \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  "${IMAGE}" server /data --address ":9000" --console-address ":9001" >/dev/null

READY_URL="http://127.0.0.1:${PORT}/minio/health/ready"
echo "[mx8] waiting for minio ready (${READY_URL})"
for _ in $(seq 1 50); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "[mx8] minio did not become ready" >&2
  docker logs "${NAME}" | tail -n 200 >&2 || true
  exit 1
fi

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=1

export MX8_S3_ENDPOINT_URL="http://127.0.0.1:${PORT}"
export MX8_S3_FORCE_PATH_STYLE=1

export MX8_MANIFEST_STORE_ROOT="s3://${MANIFEST_STORE_BUCKET}/${MANIFEST_STORE_PREFIX}"

echo "[mx8] generating dev manifest (${TOTAL_SAMPLES} samples)"
python3 - "${DEV_MANIFEST}" "${TOTAL_SAMPLES}" "${BYTES_PER_SAMPLE}" <<'PY'
import sys

path = sys.argv[1]
n = int(sys.argv[2])
bytes_per = int(sys.argv[3])

with open(path, "w", encoding="utf-8") as f:
    for i in range(n):
        off = i * bytes_per
        f.write(f"{i}\ts3://mx8-demo/data\t{off}\t{bytes_per}\n")
PY

echo "[mx8] building mx8-snapshot-resolve (s3 feature enabled)"
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve >/dev/null

echo "[mx8] running ${RESOLVE_PROCS} concurrent resolvers (DATASET_LINK=${DATASET_LINK})"
pids=""
for i in $(seq 1 "${RESOLVE_PROCS}"); do
  log="${TMP_ROOT}/resolver_${i}.log"
  MX8_DATASET_LINK="${DATASET_LINK}" \
    MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
    MX8_NODE_ID="resolver${i}" \
    MX8_SNAPSHOT_LOCK_STALE_MS="${LOCK_STALE_MS}" \
    MX8_SNAPSHOT_WAIT_TIMEOUT_MS="${WAIT_TIMEOUT_MS}" \
    target/debug/mx8-snapshot-resolve >"${log}" 2>&1 &
  pids="${pids} $!"
done

for pid in ${pids}; do
  wait "${pid}"
done

python3 - "${TMP_ROOT}" "${RESOLVE_PROCS}" <<'PY'
import os
import re
import sys

root = sys.argv[1]
n = int(sys.argv[2])

manifest_hashes = []
indexer = 0

for i in range(1, n + 1):
    path = os.path.join(root, f"resolver_{i}.log")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    indexer += txt.count('event="snapshot_indexer_elected"')

    mh = None
    for line in txt.splitlines():
        if line.startswith("manifest_hash:"):
            mh = line.split("manifest_hash:", 1)[1].strip()
            break
    if not mh:
        raise SystemExit(f"{path} missing manifest_hash output")
    manifest_hashes.append(mh)

if len(set(manifest_hashes)) != 1:
    raise SystemExit(f"expected identical manifest_hash for all resolvers, got: {sorted(set(manifest_hashes))}")

if indexer != 1:
    raise SystemExit(f"expected exactly one snapshot_indexer_elected, got {indexer}")

print("[mx8] minio_manifest_store_gate OK")
print("[mx8] manifest_hash:", manifest_hashes[0])
PY

