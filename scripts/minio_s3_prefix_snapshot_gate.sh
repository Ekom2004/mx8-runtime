#!/usr/bin/env bash
set -euo pipefail

# MinIO gate for S3-prefix snapshot indexing.
#
# Proves:
# - `mx8-snapshot-resolve` can create a pinned snapshot from an S3 prefix link
#   without `MX8_DEV_MANIFEST_PATH` (LIST -> canonical manifest -> manifest_hash).
# - hashing is deterministic when the object set is unchanged.
#
# Usage:
#   ./scripts/minio_s3_prefix_snapshot_gate.sh
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
NAME="mx8-minio-s3-prefix-snapshot-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-vision}"
PREFIX="${MX8_MINIO_PREFIX:-vision/}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-minio-s3-prefix-snapshot-gate.XXXXXX")"
STORE_ROOT="${TMP_ROOT}/store"
SEED_FILE="${TMP_ROOT}/seed.bin"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${STORE_ROOT}"
python3 - "${SEED_FILE}" <<'PY'
import os, sys
path = sys.argv[1]
with open(path, "wb") as f:
    f.write(os.urandom(1024))
PY

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

echo "[mx8] seeding objects under s3://${BUCKET}/${PREFIX}"
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null
for i in 1 2 3 4 5; do
  key="${PREFIX}class${i}/img${i}.jpg"
  MX8_MINIO_BUCKET="${BUCKET}" \
    MX8_MINIO_KEY="${key}" \
    MX8_SEED_FILE="${SEED_FILE}" \
    target/debug/mx8-seed-s3 >/dev/null
done

echo "[mx8] building mx8-snapshot-resolve (s3 feature enabled)"
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve >/dev/null

export MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}"
export MX8_DATASET_LINK="s3://${BUCKET}/${PREFIX}@refresh"
export MX8_NODE_ID="resolver"

echo "[mx8] resolving snapshot from S3 prefix (run 1)"
h1="$(target/debug/mx8-snapshot-resolve | tee "${TMP_ROOT}/resolve1.log" | sed -n 's/^manifest_hash: //p')"
if [[ -z "${h1}" ]]; then
  echo "[mx8] missing manifest_hash output (run 1)" >&2
  cat "${TMP_ROOT}/resolve1.log" >&2 || true
  exit 1
fi

echo "[mx8] resolving snapshot from S3 prefix (run 2)"
h2="$(target/debug/mx8-snapshot-resolve | tee "${TMP_ROOT}/resolve2.log" | sed -n 's/^manifest_hash: //p')"
if [[ -z "${h2}" ]]; then
  echo "[mx8] missing manifest_hash output (run 2)" >&2
  cat "${TMP_ROOT}/resolve2.log" >&2 || true
  exit 1
fi

if [[ "${h1}" != "${h2}" ]]; then
  echo "[mx8] expected deterministic manifest_hash (unchanged prefix), got h1=${h1} h2=${h2}" >&2
  exit 1
fi

manifest_path="${STORE_ROOT}/by-hash/${h1}"
if [[ ! -f "${manifest_path}" ]]; then
  echo "[mx8] expected manifest bytes at ${manifest_path}" >&2
  exit 1
fi

python3 - "${manifest_path}" "${BUCKET}" "${PREFIX}" <<'PY'
import sys

path = sys.argv[1]
bucket = sys.argv[2]
prefix = sys.argv[3]

txt = open(path, "r", encoding="utf-8", errors="replace").read()
need = [
    f"s3://{bucket}/{prefix}class1/img1.jpg",
    f"s3://{bucket}/{prefix}class5/img5.jpg",
]
for s in need:
    if s not in txt:
        raise SystemExit(f"manifest missing expected location: {s}")
print("[mx8] minio_s3_prefix_snapshot_gate OK")
PY

echo "[mx8] manifest_hash: ${h1}"

