#!/usr/bin/env bash
set -euo pipefail

# Local decode-backend perf compare gate:
# - builds/install mx8 Python extension
# - creates a local ImageFolder dataset
# - packs via mx8.pack_dir
# - benchmarks rust vs python decode backend across modes/threads
#
# Usage:
#   ./scripts/py_image_decode_backend_bench.sh
#
# Optional env:
#   MX8_IMAGE_BENCH_SAMPLES=8192
#   MX8_IMAGE_BENCH_IMAGE_HW=24
#   MX8_IMAGE_BENCH_BATCH_SIZE=64
#   MX8_IMAGE_BENCH_STEPS=128
#   MX8_IMAGE_BENCH_WARMUP_STEPS=8
#   MX8_IMAGE_BENCH_BACKENDS=rust,python
#   MX8_IMAGE_BENCH_RUST_CODECS=zune,image,turbo
#   MX8_IMAGE_BENCH_MODES=decode_only,train_step
#   MX8_IMAGE_BENCH_TORCH_THREADS_LIST=1,4,8
#   MX8_IMAGE_BENCH_DECODE_THREADS_LIST=1,4,8
#   MX8_RUST_RESIZE_BACKEND=fast|image
#   MX8_DECODE_BENCH_MIN_SPEEDUP=0.95   # fail if rust speedup is lower

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-image-decode-bench.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"
TOTAL_SAMPLES="${MX8_IMAGE_BENCH_SAMPLES:-8192}"
IMAGE_HW="${MX8_IMAGE_BENCH_IMAGE_HW:-24}"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}/cat" "${RAW_DIR}/dog" "${OUT_DIR}"

echo "[mx8] venv + deps (numpy + pillow + torch + maturin)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy pillow >/dev/null
"${PYTHON_BIN}" -m pip install -U torch >/dev/null

echo "[mx8] generating ${TOTAL_SAMPLES} JPEGs ${IMAGE_HW}x${IMAGE_HW} (cat/dog)"
"${PYTHON_BIN}" - "${RAW_DIR}" "${TOTAL_SAMPLES}" "${IMAGE_HW}" <<'PY'
import os
import sys
from PIL import Image

raw = sys.argv[1]
total = int(sys.argv[2])
hw = int(sys.argv[3])
if total < 2:
    raise SystemExit("TOTAL_SAMPLES must be >= 2")
if hw < 8:
    raise SystemExit("IMAGE_HW must be >= 8")

for i in range(total):
    cls = "cat" if i % 2 == 0 else "dog"
    sub = os.path.join(raw, cls)
    os.makedirs(sub, exist_ok=True)
    r = (i * 17) % 255
    g = (i * 31) % 255
    b = (i * 47) % 255
    img = Image.new("RGB", (hw, hw), (r, g, b))
    img.save(os.path.join(sub, f"{cls}_{i:06d}.jpg"), format="JPEG", quality=90)
PY

echo "[mx8] maturin develop"
(
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

echo "[mx8] python: mx8.pack_dir"
"${PYTHON_BIN}" - "${RAW_DIR}" "${OUT_DIR}" <<'PY' >/dev/null
import sys
import mx8

raw_dir = sys.argv[1]
out_dir = sys.argv[2]
res = mx8.pack_dir(
    raw_dir,
    out=out_dir,
    shard_mb=8,
    label_mode="imagefolder",
    require_labels=True,
)
if int(res["samples"]) <= 0:
    raise SystemExit(f"unexpected pack_dir result: {res}")
PY

echo "[mx8] running decode backend benchmark (rust vs python)"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}@refresh" \
MX8_IMAGE_BENCH_BATCH_SIZE="${MX8_IMAGE_BENCH_BATCH_SIZE:-64}" \
MX8_IMAGE_BENCH_STEPS="${MX8_IMAGE_BENCH_STEPS:-128}" \
MX8_IMAGE_BENCH_WARMUP_STEPS="${MX8_IMAGE_BENCH_WARMUP_STEPS:-8}" \
MX8_IMAGE_BENCH_BACKENDS="${MX8_IMAGE_BENCH_BACKENDS:-rust,python}" \
MX8_IMAGE_BENCH_RUST_CODECS="${MX8_IMAGE_BENCH_RUST_CODECS:-zune,image}" \
MX8_IMAGE_BENCH_MODES="${MX8_IMAGE_BENCH_MODES:-decode_only,train_step}" \
MX8_IMAGE_BENCH_TORCH_THREADS_LIST="${MX8_IMAGE_BENCH_TORCH_THREADS_LIST:-1}" \
MX8_IMAGE_BENCH_DECODE_THREADS_LIST="${MX8_IMAGE_BENCH_DECODE_THREADS_LIST:-1}" \
MX8_DECODE_BENCH_MIN_SPEEDUP="${MX8_DECODE_BENCH_MIN_SPEEDUP:-0}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_image_decode_backend_bench.py"

echo "[mx8] py_image_decode_backend_bench OK"
