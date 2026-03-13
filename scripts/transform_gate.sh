#!/usr/bin/env bash
set -euo pipefail

# Transform loader gate for @mx8.transform + mx8.load(transform=...).
#
# Proves:
# - end-to-end Python install path via maturin.
# - decorator contract + startup validation.
# - deterministic transformed output on repeated runs.
# - cap violation fails in bounded way.
#
# Usage:
#   ./scripts/transform_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-transform-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"
DEV_MANIFEST_PATH="${TMP_ROOT}/dev_manifest.tsv"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}" "${OUT_DIR}"

echo "[mx8] venv + deps (maturin)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] generating raw samples"
"${PYTHON_BIN}" - "${RAW_DIR}" <<'PY'
import os
import sys

raw_dir = sys.argv[1]
samples = {
    "000.txt": b"hello world\n",
    "001.txt": b"  mx8 transform   ",
    "002.txt": b"deterministic payload",
}

for name, payload in samples.items():
    with open(os.path.join(raw_dir, name), "wb") as f:
        f.write(payload)
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
    label_mode="none",
)
if int(res["samples"]) != 3:
    raise SystemExit(f"unexpected pack_dir result: {res}")
PY

echo "[mx8] deriving dev manifest for local gate"
tail -n +2 "${OUT_DIR}/_mx8/manifest.tsv" > "${DEV_MANIFEST_PATH}"

echo "[mx8] running transform gate"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST_PATH}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m9_transform_gate.py" >/dev/null

echo "[mx8] transform_gate OK"
