#!/usr/bin/env bash
set -euo pipefail

# Local decode-backend perf compare gate:
# - builds/install mx8 Python extension
# - creates a local ImageFolder dataset
# - packs via mx8.pack_dir
# - benchmarks rust (default) vs python decode backend
#
# Usage:
#   ./scripts/py_vision_decode_backend_bench.sh
#
# Optional env:
#   MX8_VISION_BENCH_SAMPLES=8192
#   MX8_VISION_BENCH_BATCH_SIZE=64
#   MX8_VISION_BENCH_STEPS=128
#   MX8_VISION_BENCH_WARMUP_STEPS=8
#   MX8_DECODE_BENCH_MIN_SPEEDUP=0.95   # fail if rust speedup is lower

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-vision-decode-bench.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"
TOTAL_SAMPLES="${MX8_VISION_BENCH_SAMPLES:-8192}"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}/cat" "${RAW_DIR}/dog" "${OUT_DIR}"

echo "[mx8] venv + deps (numpy + pillow + torch + maturin)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy pillow >/dev/null
"${PYTHON_BIN}" -m pip install -U torch >/dev/null

echo "[mx8] generating ${TOTAL_SAMPLES} tiny JPEGs (cat/dog)"
"${PYTHON_BIN}" - "${RAW_DIR}" "${TOTAL_SAMPLES}" <<'PY'
import os
import sys
from PIL import Image

raw = sys.argv[1]
total = int(sys.argv[2])
if total < 2:
    raise SystemExit("TOTAL_SAMPLES must be >= 2")

for i in range(total):
    cls = "cat" if i % 2 == 0 else "dog"
    sub = os.path.join(raw, cls)
    os.makedirs(sub, exist_ok=True)
    r = (i * 17) % 255
    g = (i * 31) % 255
    b = (i * 47) % 255
    img = Image.new("RGB", (24, 24), (r, g, b))
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
MX8_VISION_BENCH_BATCH_SIZE="${MX8_VISION_BENCH_BATCH_SIZE:-64}" \
MX8_VISION_BENCH_STEPS="${MX8_VISION_BENCH_STEPS:-128}" \
MX8_VISION_BENCH_WARMUP_STEPS="${MX8_VISION_BENCH_WARMUP_STEPS:-8}" \
MX8_DECODE_BENCH_MIN_SPEEDUP="${MX8_DECODE_BENCH_MIN_SPEEDUP:-0}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_vision_decode_backend_bench.py"

echo "[mx8] py_vision_decode_backend_bench OK"
