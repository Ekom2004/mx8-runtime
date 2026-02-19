#!/usr/bin/env bash
set -euo pipefail

# v1.8 stage-1 gate:
# - deterministic clip indexing contract
# - failure taxonomy visibility
# - memory bound enforcement
# - schema version stability
#
# Usage:
#   ./scripts/video_stage1_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

echo "[mx8] video stage1 gate: cargo test -p mx8-snapshot video_stage1"
CARGO_INCREMENTAL=0 RUSTFLAGS="-C debuginfo=0" cargo test -p mx8-snapshot video_stage1

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-video-stage1-gate-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install maturin"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

STORE_ROOT="${TMP_ROOT}/store"
DEV_MANIFEST="${TMP_ROOT}/video_stage1_dev_manifest.tsv"
mkdir -p "${STORE_ROOT}"

echo "[mx8] python video stage1 gate"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
MX8_DATASET_LINK="${TMP_ROOT}/video_stage1@refresh" \
MX8_VIDEO_STAGE1_INDEX=1 \
MX8_VIDEO_STAGE1_DISABLE_FFPROBE=1 \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_video_stage1_gate.py"

echo "[mx8] video_stage1_gate OK"
