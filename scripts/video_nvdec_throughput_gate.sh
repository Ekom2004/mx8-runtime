#!/usr/bin/env bash
set -euo pipefail

# NVDEC throughput gate:
# - build with `mx8_video_nvdec` cfg
# - compare `nvdec` vs `cli` on mixed 1080p/4K fixtures
# - enforce speedup only when NVDEC hardware path is actually active
#
# Usage:
#   ./scripts/video_nvdec_throughput_gate.sh

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

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-video-nvdec-throughput-gate-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install maturin"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] maturin develop (with mx8_video_nvdec cfg)"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  export RUSTFLAGS="--cfg mx8_video_nvdec"
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

echo "[mx8] python nvdec throughput gate"
MX8_VIDEO_NVDEC_THROUGHPUT_TMP_ROOT="${TMP_ROOT}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_video_nvdec_throughput_gate.py"

echo "[mx8] video_nvdec_throughput_gate OK"
