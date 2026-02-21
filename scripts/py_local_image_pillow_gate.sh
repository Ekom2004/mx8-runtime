#!/usr/bin/env bash
set -euo pipefail

# Local (no S3/MinIO) image training gate:
#
# Proves the frictionless user story without infra:
# - Create a tiny ImageFolder-style directory on disk.
# - `mx8.pack_dir(...)` packs it into tar shards + `_mx8/manifest.tsv` + `_mx8/labels.tsv`.
# - `mx8.image(...)` yields `(images, labels)` torch tensors.
# - A tiny Torch loop trains for N steps.
#
# Usage:
#   ./scripts/py_local_image_pillow_gate.sh
#
# Prereqs:
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-local-image-gate.XXXXXX")"
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

echo "[mx8] generating tiny JPEGs (cat/dog)"
"${PYTHON_BIN}" - "${RAW_DIR}" <<'PY'
import os, sys
from PIL import Image

root = sys.argv[1]

def write(subdir: str, name: str, rgb: tuple[int,int,int]) -> None:
    out = os.path.join(root, subdir, name)
    img = Image.new("RGB", (16, 16), rgb)
    img.save(out, format="JPEG", quality=90)

write("cat", "cat0.jpg", (255, 0, 0))
write("cat", "cat1.jpg", (200, 0, 0))
write("dog", "dog0.jpg", (0, 255, 0))
write("dog", "dog1.jpg", (0, 200, 0))
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

echo "[mx8] python: mx8.image classes"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}@refresh" \
"${PYTHON_BIN}" - <<'PY' >/dev/null
import os
import mx8

loader = mx8.image(
    os.environ["MX8_DATASET_LINK"],
    manifest_store=os.environ["MX8_MANIFEST_STORE_ROOT"],
    batch_size_samples=2,
    prefetch_batches=2,
)
classes = loader.classes
if classes is None:
    raise SystemExit("expected loader.classes to be present (got None)")
classes = list(classes)
if classes != ["cat", "dog"]:
    raise SystemExit(f"unexpected classes: {classes}")
PY

echo "[mx8] running minimal Pillow image train"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}@refresh" \
MX8_TRAIN_STEPS="${MX8_TRAIN_STEPS:-8}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_image_pillow_train_minimal.py" >/dev/null

echo "[mx8] py_local_image_pillow_gate OK"
