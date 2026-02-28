#!/usr/bin/env bash
set -euo pipefail

# Image augmentation gate for mx8.image.
#
# Proves:
# - `augment="imagenet"` preset wiring.
# - explicit augmentation knobs (crop/flip/jitter/normalize).
# - deterministic replay for fixed (manifest_hash, seed, epoch, sample_id).
#
# Usage:
#   ./scripts/image_aug_gate.sh
#
# Prereqs:
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-image-aug-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}/cat" "${RAW_DIR}/dog" "${OUT_DIR}"

echo "[mx8] venv + deps (numpy + pillow + torch + maturin)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy pillow >/dev/null
"${PYTHON_BIN}" -m pip install -U torch >/dev/null

echo "[mx8] generating gradient JPEG dataset (cat/dog)"
"${PYTHON_BIN}" - "${RAW_DIR}" <<'PY'
import os
import sys
import numpy as np
from PIL import Image

root = sys.argv[1]
h, w = 320, 320
yy, xx = np.meshgrid(np.arange(h, dtype=np.uint16), np.arange(w, dtype=np.uint16), indexing="ij")

def write(label: str, name: str, phase: int) -> None:
    r = ((xx + phase * 13) % 256).astype(np.uint8)
    g = ((yy + phase * 29) % 256).astype(np.uint8)
    b = (((xx + yy) // 2 + phase * 7) % 256).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    out = os.path.join(root, label, name)
    Image.fromarray(rgb).save(out, format="JPEG", quality=95)

write("cat", "cat0.jpg", 1)
write("cat", "cat1.jpg", 2)
write("dog", "dog0.jpg", 3)
write("dog", "dog1.jpg", 4)
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
    shard_mb=1,
    label_mode="imagefolder",
    require_labels=True,
)
if int(res["samples"]) != 4:
    raise SystemExit(f"unexpected pack_dir result: {res}")
PY

echo "[mx8] running image augmentation contract gate"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}@refresh" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_image_aug_gate.py" >/dev/null

echo "[mx8] image_aug_gate OK"
