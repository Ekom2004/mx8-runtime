#!/usr/bin/env bash
set -euo pipefail

# Video direct-write native-op gate:
# - builds and loads mx8_video direct-write custom op
# - runs video device-output gate with direct-write enabled
#
# Usage:
#   ./scripts/video_direct_write_native_op_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi
if ! command -v rustc >/dev/null 2>&1; then
  if [[ -x "${HOME}/.cargo/bin/rustc" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi
if ! command -v rustc >/dev/null 2>&1; then
  echo "[mx8] rustc not found on PATH (install Rust via rustup or add ~/.cargo/bin to PATH)" >&2
  exit 1
fi
if ! command -v "${MX8_FFMPEG_BIN:-ffmpeg}" >/dev/null 2>&1; then
  echo "[mx8] ffmpeg not found on PATH (MX8_FFMPEG_BIN=${MX8_FFMPEG_BIN:-ffmpeg})" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-video-direct-write-op-gate-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
export PATH="${VENV_DIR}/bin:${PATH}"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install maturin + torch + ninja"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U torch >/dev/null
"${PYTHON_BIN}" -m pip install -U ninja >/dev/null

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

echo "[mx8] build direct-write native op library"
OP_LIB_PATH="$("${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_build_video_direct_write_op.py" --out-dir "${TMP_ROOT}/op-build")"
echo "[mx8] op library=${OP_LIB_PATH}"

echo "[mx8] python video device-output gate (native-op enabled)"
MX8_VIDEO_DEVICE_OUTPUT_TMP_ROOT="${TMP_ROOT}" \
MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY="${OP_LIB_PATH}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_video_device_output_gate.py"

echo "[mx8] video_direct_write_native_op_gate OK"
