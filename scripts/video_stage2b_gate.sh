#!/usr/bin/env bash
set -euo pipefail

# v1.8 stage-2b scaffold gate:
# - replay determinism
# - bounded init guard
# - runtime IO decode failure classification
# - stage1 failure taxonomy visibility carry-over
#
# Usage:
#   ./scripts/video_stage2b_gate.sh

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

echo "[mx8] video stage2b gate: cargo test -p mx8-snapshot video_stage1"
CARGO_INCREMENTAL=0 RUSTFLAGS="-C debuginfo=0" cargo test -p mx8-snapshot video_stage1

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-video-stage2b-gate-XXXXXX")"
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

echo "[mx8] python video stage2b gate"
MX8_VIDEO_STAGE2B_TMP_ROOT="${TMP_ROOT}" \
MX8_VIDEO_STAGE1_INDEX=1 \
MX8_VIDEO_STAGE2_BYTES_PER_CLIP="${MX8_VIDEO_STAGE2_BYTES_PER_CLIP:-2048}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m8_video_stage2b_gate.py"

echo "[mx8] video_stage2b_gate OK"
