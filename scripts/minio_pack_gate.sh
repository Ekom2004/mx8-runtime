#!/usr/bin/env bash
set -euo pipefail

# MinIO gate for `mx8-pack-s3`:
#
# Proves:
# - Pack: S3 prefix (many small objects) -> tar shards + `_mx8/manifest.tsv`
# - Snapshot resolve: uses precomputed manifest (no LIST storm required)
# - Python DataLoader: can read byte ranges from tar shards and delivers optional `label_ids`
#
# Usage:
#   ./scripts/minio_pack_gate.sh
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
NAME="mx8-minio-pack-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-vision}"
RAW_PREFIX="${MX8_MINIO_RAW_PREFIX:-raw/train/}"
OUT_PREFIX="${MX8_MINIO_OUT_PREFIX:-mx8/train/}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-minio-pack-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
SEED_FILE="${TMP_ROOT}/seed.bin"
VENV_DIR="${TMP_ROOT}/venv"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${STORE_ROOT}"

python3 - "${SEED_FILE}" <<'PY'
import os, sys
path = sys.argv[1]
with open(path, "wb") as f:
    f.write(os.urandom(64))
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

echo "[mx8] seeding raw objects under s3://${BUCKET}/${RAW_PREFIX}(cat|dog)/..."
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null
for key in \
  "${RAW_PREFIX}cat/000.bin" \
  "${RAW_PREFIX}cat/001.bin" \
  "${RAW_PREFIX}dog/000.bin" \
  "${RAW_PREFIX}dog/001.bin"
do
  MX8_MINIO_BUCKET="${BUCKET}" \
    MX8_MINIO_KEY="${key}" \
    MX8_SEED_FILE="${SEED_FILE}" \
    target/debug/mx8-seed-s3 >/dev/null
done

echo "[mx8] building mx8-pack-s3 + mx8-snapshot-resolve (s3 feature enabled)"
cargo build -p mx8-snapshot --features s3 --bin mx8-pack-s3 >/dev/null
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve >/dev/null

export MX8_S3_LABEL_MODE=imagefolder

IN_LINK="s3://${BUCKET}/${RAW_PREFIX}"
OUT_LINK="s3://${BUCKET}/${OUT_PREFIX}"

echo "[mx8] packing ${IN_LINK} -> ${OUT_LINK}"
MX8_PACK_IN="${IN_LINK}" \
MX8_PACK_OUT="${OUT_LINK}" \
MX8_PACK_SHARD_MB=1 \
MX8_PACK_REQUIRE_LABELS=true \
target/debug/mx8-pack-s3 | tee "${TMP_ROOT}/pack.log" >/dev/null

echo "[mx8] resolving snapshot from packed prefix (should use _mx8/manifest.tsv)"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_LINK}@refresh" \
MX8_NODE_ID="resolver" \
RUST_LOG=info \
target/debug/mx8-snapshot-resolve | tee "${TMP_ROOT}/resolve.log" >/dev/null

python3 - "${TMP_ROOT}/resolve.log" <<'PY'
import sys
txt = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
if "snapshot_index_precomputed_manifest" not in txt:
    raise SystemExit("expected resolver to index from precomputed manifest (missing snapshot_index_precomputed_manifest log)")
print("[mx8] precomputed manifest path used")
PY

echo "[mx8] venv + maturin develop"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
(
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

echo "[mx8] python: load packed prefix + assert label_ids"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_LINK}@refresh" \
"${PYTHON_BIN}" - <<'PY'
import os
import mx8

root = os.environ["MX8_MANIFEST_STORE_ROOT"]
link = os.environ["MX8_DATASET_LINK"]

loader = mx8.DataLoader(
    link,
    manifest_store=root,
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
    for sid, lab in zip(sids, labs):
        seen.append((int(sid), int(lab)))
    if len(seen) >= 4:
        break
seen.sort()
want = [(0, 0), (1, 0), (2, 1), (3, 1)]  # cat=0, dog=1
if seen != want:
    raise SystemExit(f"unexpected (sample_id,label_id) pairs: got={seen} want={want}")
print("[mx8] minio_pack_gate OK")
PY
