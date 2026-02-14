#!/usr/bin/env bash
set -euo pipefail

# Python+MinIO gate for "vision v0" ergonomics:
#
# Proves:
# - `mx8.DataLoader` can resolve `s3://bucket/prefix@refresh` via LIST (MinIO).
# - S3 prefix indexing attaches ImageFolder-style `label_id`s (stable mapping).
# - Runtime delivers `label_ids` alongside `(payload, offsets, sample_ids)`.
#
# Usage:
#   ./scripts/py_minio_imagefolder_gate.sh
#
# Prereqs:
# - docker (daemon running)
# - python3
# - curl

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

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

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-minio-imagefolder-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"

echo "[mx8] install deps (maturin)"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] maturin develop (mx8, with s3 enabled)"
(
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-py-imagefolder-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-vision}"
PREFIX="${MX8_MINIO_PREFIX:-vision/}"

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
    f.write(os.urandom(32))
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
for _ in $(seq 1 80); do
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

# Force ImageFolder labeling for this gate.
export MX8_S3_LABEL_MODE=imagefolder

echo "[mx8] seeding s3://${BUCKET}/${PREFIX}(cat|dog)/..."
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null
for key in \
  "${PREFIX}cat/000.bin" \
  "${PREFIX}cat/001.bin" \
  "${PREFIX}dog/000.bin" \
  "${PREFIX}dog/001.bin"
do
  MX8_MINIO_BUCKET="${BUCKET}" \
    MX8_MINIO_KEY="${key}" \
    MX8_SEED_FILE="${SEED_FILE}" \
    target/debug/mx8-seed-s3 >/dev/null
done

echo "[mx8] python: iterate loader + assert label_ids"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="s3://${BUCKET}/${PREFIX}@refresh" \
"${PYTHON_BIN}" - <<'PY'
import os
import mx8

root = os.environ["MX8_MANIFEST_STORE_ROOT"]
link = os.environ["MX8_DATASET_LINK"]

loader = mx8.DataLoader(
    link,
    manifest_store_root=root,
    batch_size_samples=2,
    max_inflight_bytes=8 * 1024 * 1024,
    max_queue_batches=8,
    prefetch_batches=2,
    node_id="py_gate",
)

seen = []
for batch in loader:
    sids = list(batch.sample_ids)
    labs = batch.label_ids
    if labs is None:
        raise SystemExit("expected batch.label_ids to be present (got None)")
    labs = list(labs)
    if len(sids) != len(labs):
        raise SystemExit(f"label_ids length mismatch: sample_ids={len(sids)} labels={len(labs)}")
    for sid, lab in zip(sids, labs):
        seen.append((int(sid), int(lab)))
    if len(seen) >= 4:
        break

seen.sort()
want = [(0, 0), (1, 0), (2, 1), (3, 1)]  # cat=0, dog=1 (lexicographic label order)
if seen != want:
    raise SystemExit(f"unexpected (sample_id,label_id) pairs: got={seen} want={want}")

print("[mx8] py_minio_imagefolder_gate OK")
PY

