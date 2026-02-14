#!/usr/bin/env bash
set -euo pipefail

# Smoke-test a locally built wheel via pip install (no maturin develop).
#
# Usage:
#   ./scripts/build_wheel.sh
#   ./scripts/pip_wheel_smoke.sh
#
# Notes:
# - Installs torch + pillow + numpy to exercise `mx8.vision.ImageFolderLoader`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WHEEL="$(ls -1 dist/mx8-*.whl 2>/dev/null | head -n 1 || true)"
if [[ -z "${WHEEL}" ]]; then
  echo "[mx8] no wheel found (expected dist/mx8-*.whl); run ./scripts/build_wheel.sh first" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-pip-wheel-smoke.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] pip install wheel + deps"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy pillow torch >/dev/null
"${PYTHON_BIN}" -m pip install "${WHEEL}" >/dev/null

RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
STORE_ROOT="${TMP_ROOT}/store"
mkdir -p "${RAW_DIR}/cat" "${RAW_DIR}/dog" "${OUT_DIR}" "${STORE_ROOT}"

echo "[mx8] python: pack_dir + load one batch"
"${PYTHON_BIN}" - "${RAW_DIR}" "${OUT_DIR}" "${STORE_ROOT}" <<'PY'
import os, sys
from PIL import Image
import mx8

raw_dir, out_dir, store_root = sys.argv[1:]

def write(subdir: str, name: str, rgb):
    p = os.path.join(raw_dir, subdir, name)
    img = Image.new("RGB", (16, 16), rgb)
    img.save(p, format="JPEG", quality=90)

write("cat", "0.jpg", (255, 0, 0))
write("cat", "1.jpg", (200, 0, 0))
write("dog", "0.jpg", (0, 255, 0))
write("dog", "1.jpg", (0, 200, 0))

mx8.pack_dir(raw_dir, out=out_dir, shard_mb=1, label_mode="imagefolder", require_labels=True)

loader = mx8.vision.ImageFolderLoader(
    f"{out_dir}@refresh",
    manifest_store_root=store_root,
    batch_size_samples=2,
    prefetch_batches=2,
    to_float=True,
)

classes = loader.classes
assert classes == ["cat", "dog"], classes

images, labels = next(iter(loader))
assert images.shape[0] == 2
assert labels.shape[0] == 2
print("[mx8] pip_wheel_smoke OK")
PY

