#!/usr/bin/env bash
set -euo pipefail

# Python smoke gate for M5 (PyO3 veneer).
#
# Usage:
#   ./scripts/py_smoke.sh
#
# Prereqs:
# - python3 on PATH
#
# Notes:
# - This script creates a temporary venv and installs `maturin` into it, so the
#   Python interpreter running the example matches the one that built/installed
#   the extension module.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-py-smoke-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"

echo "[mx8] install maturin"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

STORE_ROOT="${TMP_ROOT}/store"
DATA_FILE="${TMP_ROOT}/data.bin"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"

mkdir -p "${STORE_ROOT}"

TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-256}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-256}"

TOTAL_BYTES="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
truncate -s "${TOTAL_BYTES}" "${DATA_FILE}"

rm -f "${DEV_MANIFEST}"
for ((i=0; i<"${TOTAL_SAMPLES}"; i++)); do
  off="$(( i * BYTES_PER_SAMPLE ))"
  printf "%s\t%s\t%s\t%s\n" "${i}" "${DATA_FILE}" "${off}" "${BYTES_PER_SAMPLE}" >> "${DEV_MANIFEST}"
done

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  # Ensure `maturin develop` targets *this* venv even if the caller has another venv activated.
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

echo "[mx8] python example"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
  MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
  MX8_DATASET_LINK="${TMP_ROOT}/dev@refresh" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_minimal.py"

echo "[mx8] python torch example (optional)"
if "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
    MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
    MX8_DATASET_LINK="${TMP_ROOT}/dev@refresh" \
    "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_minimal.py"
else
  echo "[mx8] torch not installed in venv; skipping"
fi

echo "[mx8] py_smoke OK"
