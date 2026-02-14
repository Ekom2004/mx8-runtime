#!/usr/bin/env bash
set -euo pipefail

# Pillow-based vision training gate:
#
# Proves the end-to-end "vision v0" user story:
# - Seed ImageFolder-style JPEGs into MinIO (S3-compatible).
# - Pack raw prefix -> tar shards + `_mx8/manifest.tsv`.
# - `mx8.DataLoader` reads byte ranges from the packed prefix.
# - Pillow decodes JPEG bytes in Python, and a tiny Torch loop trains for N steps.
#
# Usage:
#   ./scripts/py_minio_vision_pillow_gate.sh
#
# Prereqs:
# - docker (daemon running)
# - curl
# - python3

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

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-pillow-vision-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-vision}"
RAW_PREFIX="${MX8_MINIO_RAW_PREFIX:-raw/vision/train/}"
OUT_PREFIX="${MX8_MINIO_OUT_PREFIX:-mx8/vision/train/}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-minio-vision-pillow-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
VENV_DIR="${TMP_ROOT}/venv"
IMG_DIR="${TMP_ROOT}/imgs"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${STORE_ROOT}" "${IMG_DIR}"

echo "[mx8] venv + deps (numpy + pillow + maturin)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy pillow >/dev/null

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
export MX8_S3_LABEL_MODE=imagefolder

echo "[mx8] generating tiny JPEGs (cat/dog)"
"${PYTHON_BIN}" - "${IMG_DIR}" <<'PY'
import os, sys
from PIL import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def write(name: str, rgb: tuple[int,int,int]) -> None:
    img = Image.new("RGB", (16, 16), rgb)
    img.save(os.path.join(out, name), format="JPEG", quality=90)

write("cat0.jpg", (255, 0, 0))
write("cat1.jpg", (200, 0, 0))
write("dog0.jpg", (0, 255, 0))
write("dog1.jpg", (0, 200, 0))
PY

echo "[mx8] seeding raw JPEG objects into MinIO"
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null

MX8_MINIO_BUCKET="${BUCKET}" MX8_MINIO_KEY="${RAW_PREFIX}cat/cat0.jpg" MX8_SEED_FILE="${IMG_DIR}/cat0.jpg" target/debug/mx8-seed-s3 >/dev/null
MX8_MINIO_BUCKET="${BUCKET}" MX8_MINIO_KEY="${RAW_PREFIX}cat/cat1.jpg" MX8_SEED_FILE="${IMG_DIR}/cat1.jpg" target/debug/mx8-seed-s3 >/dev/null
MX8_MINIO_BUCKET="${BUCKET}" MX8_MINIO_KEY="${RAW_PREFIX}dog/dog0.jpg" MX8_SEED_FILE="${IMG_DIR}/dog0.jpg" target/debug/mx8-seed-s3 >/dev/null
MX8_MINIO_BUCKET="${BUCKET}" MX8_MINIO_KEY="${RAW_PREFIX}dog/dog1.jpg" MX8_SEED_FILE="${IMG_DIR}/dog1.jpg" target/debug/mx8-seed-s3 >/dev/null

echo "[mx8] packing raw prefix -> tar shards"
cargo build -p mx8-snapshot --features s3 --bin mx8-pack-s3 >/dev/null
MX8_PACK_IN="s3://${BUCKET}/${RAW_PREFIX}" \
MX8_PACK_OUT="s3://${BUCKET}/${OUT_PREFIX}" \
MX8_PACK_SHARD_MB=1 \
MX8_PACK_REQUIRE_LABELS=true \
target/debug/mx8-pack-s3 >/dev/null

echo "[mx8] install torch"
"${PYTHON_BIN}" -m pip install -U torch >/dev/null

echo "[mx8] maturin develop"
(
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

echo "[mx8] running minimal Pillow vision train"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="s3://${BUCKET}/${OUT_PREFIX}@refresh" \
MX8_TRAIN_STEPS="${MX8_TRAIN_STEPS:-8}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_vision_pillow_train_minimal.py" >/dev/null

echo "[mx8] py_minio_vision_pillow_gate OK"
