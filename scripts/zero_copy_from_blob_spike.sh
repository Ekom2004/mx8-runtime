#!/usr/bin/env bash
set -euo pipefail

# Zero-copy spike gate:
# - installs torch in an isolated venv
# - compiles a tiny C++ extension that creates CUDA tensors via torch::from_blob + custom deleter
# - stress-tests deleter timing across multiple CUDA streams
#
# Usage:
#   ./scripts/zero_copy_from_blob_spike.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[mx8] nvidia-smi not found on PATH (GPU host required)" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-zero-copy-from-blob-spike-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install torch + build deps"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U torch ninja setuptools wheel >/dev/null

echo "[mx8] run zero-copy from_blob spike"
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_zero_copy_from_blob_spike.py"

echo "[mx8] zero_copy_from_blob_spike OK"
