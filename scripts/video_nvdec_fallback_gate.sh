#!/usr/bin/env bash
set -euo pipefail

# NVDEC fallback gate:
# - request `MX8_VIDEO_DECODE_BACKEND=nvdec`
# - build without `mx8_video_nvdec` cfg
# - verify loader still delivers batches via fail-open fallback
#
# Usage:
#   ./scripts/video_nvdec_fallback_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi
if ! command -v "${MX8_FFMPEG_BIN:-ffmpeg}" >/dev/null 2>&1; then
  echo "[mx8] ffmpeg not found on PATH (MX8_FFMPEG_BIN=${MX8_FFMPEG_BIN:-ffmpeg})" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-video-nvdec-fallback-gate-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install maturin"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] maturin develop (without mx8_video_nvdec cfg)"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  unset RUSTFLAGS
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

echo "[mx8] python nvdec fallback gate"
MX8_VIDEO_NVDEC_GATE_TMP_ROOT="${TMP_ROOT}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_video_nvdec_fallback_gate.py"

echo "[mx8] video_nvdec_fallback_gate OK"
