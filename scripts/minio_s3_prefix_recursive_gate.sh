#!/usr/bin/env bash
set -euo pipefail

# MinIO gate for recursive vs non-recursive zero-manifest indexing.
#
# Proves:
# - `MX8_SNAPSHOT_RECURSIVE=0` indexes only top-level objects under the prefix.
# - `MX8_SNAPSHOT_RECURSIVE=1` indexes nested objects recursively.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[mx8] docker not found" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[mx8] docker daemon not running" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[mx8] curl not found" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found" >&2
  exit 1
fi

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-s3-prefix-recursive-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-vision}"
PREFIX="${MX8_MINIO_PREFIX:-vision/}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-minio-s3-prefix-recursive-gate.XXXXXX")"
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
curl -fsS "${READY_URL}" >/dev/null

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=1
export MX8_S3_ENDPOINT_URL="http://127.0.0.1:${PORT}"
export MX8_S3_FORCE_PATH_STYLE=1

echo "[mx8] seeding top-level + nested objects under s3://${BUCKET}/${PREFIX}"
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null
for key in \
  "${PREFIX}a.jpg" \
  "${PREFIX}nested/deeper/b.jpg" \
  "${PREFIX}nested/deeper/c.jpg"; do
  MX8_MINIO_BUCKET="${BUCKET}" \
    MX8_MINIO_KEY="${key}" \
    MX8_SEED_FILE="${SEED_FILE}" \
    target/debug/mx8-seed-s3 >/dev/null
done

echo "[mx8] building mx8-snapshot-resolve (s3 feature enabled)"
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve >/dev/null

export MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}"
export MX8_NODE_ID="resolver"
export MX8_DATASET_LINK="s3://${BUCKET}/${PREFIX}@refresh"

echo "[mx8] resolving snapshot (recursive=false)"
h_nonrec="$(
  MX8_SNAPSHOT_RECURSIVE=false target/debug/mx8-snapshot-resolve \
    | tee "${TMP_ROOT}/resolve-nonrec.log" \
    | sed -n 's/^manifest_hash: //p'
)"

echo "[mx8] resolving snapshot (recursive=true)"
h_rec="$(
  MX8_SNAPSHOT_RECURSIVE=true target/debug/mx8-snapshot-resolve \
    | tee "${TMP_ROOT}/resolve-rec.log" \
    | sed -n 's/^manifest_hash: //p'
)"

if [[ -z "${h_nonrec}" || -z "${h_rec}" ]]; then
  echo "[mx8] missing manifest hash output" >&2
  exit 1
fi

if [[ "${h_nonrec}" == "${h_rec}" ]]; then
  echo "[mx8] expected different manifest_hash for recursive=false vs true" >&2
  exit 1
fi

nonrec_path="${STORE_ROOT}/by-hash/${h_nonrec}"
rec_path="${STORE_ROOT}/by-hash/${h_rec}"
[[ -f "${nonrec_path}" ]] || { echo "[mx8] missing ${nonrec_path}" >&2; exit 1; }
[[ -f "${rec_path}" ]] || { echo "[mx8] missing ${rec_path}" >&2; exit 1; }

python3 - "${nonrec_path}" "${rec_path}" "${BUCKET}" "${PREFIX}" <<'PY'
import sys

nonrec_path, rec_path, bucket, prefix = sys.argv[1:5]

def rows(path):
    out = []
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("schema_version="):
            continue
        out.append(line.split("\t")[1])
    return out

nonrec = rows(nonrec_path)
rec = rows(rec_path)

want_top = f"s3://{bucket}/{prefix}a.jpg"
want_nested_b = f"s3://{bucket}/{prefix}nested/deeper/b.jpg"
want_nested_c = f"s3://{bucket}/{prefix}nested/deeper/c.jpg"

if nonrec != [want_top]:
    raise SystemExit(f"non-recursive manifest mismatch: got={nonrec!r}, want={[want_top]!r}")

want_rec = sorted([want_top, want_nested_b, want_nested_c])
if sorted(rec) != want_rec:
    raise SystemExit(f"recursive manifest mismatch: got={sorted(rec)!r}, want={want_rec!r}")

print("[mx8] minio_s3_prefix_recursive_gate OK")
PY

echo "[mx8] manifest_hash_nonrecursive: ${h_nonrec}"
echo "[mx8] manifest_hash_recursive: ${h_rec}"
