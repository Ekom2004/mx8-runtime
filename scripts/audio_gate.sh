#!/usr/bin/env bash
set -euo pipefail

# Audio loader gate for mx8.audio.
#
# Proves:
# - end-to-end Python install path via maturin.
# - WAV + FLAC decode paths in Rust.
# - fixed-shape float32 samples + int64 metadata output contract.
# - decode_error_policy behavior (`skip` succeeds, `error` fails).
#
# Usage:
#   ./scripts/audio_gate.sh
#
# Prereqs:
# - python3
# - ffmpeg (or set MX8_FFMPEG_BIN)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi
FFMPEG_BIN="${MX8_FFMPEG_BIN:-ffmpeg}"
if ! command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
  echo "[mx8] ffmpeg not found on PATH (MX8_FFMPEG_BIN=${FFMPEG_BIN})" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-audio-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}" "${OUT_DIR}"

echo "[mx8] venv + deps (maturin + numpy + torch)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy torch >/dev/null

echo "[mx8] generating WAV fixtures (+ invalid sample)"
"${PYTHON_BIN}" - "${RAW_DIR}" <<'PY'
import math
import os
import struct
import sys
import wave

raw_dir = sys.argv[1]
sample_rate = 16000

# 3000 frames: exercises zero-pad to sample_count=4096.
out_wav_short = os.path.join(raw_dir, "000.wav")
with wave.open(out_wav_short, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    for i in range(3000):
        v = int(0.45 * 32767.0 * math.sin(2.0 * math.pi * 220.0 * (i / sample_rate)))
        w.writeframesraw(struct.pack("<h", v))

# 5000 frames: convert to FLAC; exercises truncate to sample_count=4096.
out_wav_long = os.path.join(raw_dir, "001.wav")
with wave.open(out_wav_long, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    for i in range(5000):
        v = int(0.35 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
        w.writeframesraw(struct.pack("<h", v))

with open(os.path.join(raw_dir, "999.bin"), "wb") as f:
    f.write(b"not-audio")
PY

echo "[mx8] encoding FLAC fixture"
"${FFMPEG_BIN}" -hide_banner -loglevel error -nostdin -y \
  -i "${RAW_DIR}/001.wav" \
  -c:a flac \
  "${RAW_DIR}/001.flac"
rm -f "${RAW_DIR}/001.wav"

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

echo "[mx8] running audio loader contract gate"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_audio_gate.py" >/dev/null

echo "[mx8] audio_gate OK"
