#!/usr/bin/env bash
set -euo pipefail

# v1.7 mix gate:
# - scheduler unit/integration tests
# - public Python API deterministic replay + ratio + memory-cap gate
#
# Usage:
#   ./scripts/mix_gate.sh
#
# Optional tuning:
#   MX8_MIX_GATE_STEPS=400 MX8_MIX_GATE_RATIO_TOL=0.02 ./scripts/mix_gate.sh
#   MX8_MIX_GATE_WEIGHTS=0.8,0.2 ./scripts/mix_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

echo "[mx8] mix gate: cargo test -p mx8-py mix_"
CARGO_INCREMENTAL=0 RUSTFLAGS="-C debuginfo=0" cargo test -p mx8-py mix_

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-mix-gate-XXXXXX")"
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
DATA_A="${TMP_ROOT}/data_a.bin"
DATA_B="${TMP_ROOT}/data_b.bin"
MANIFEST_A="${TMP_ROOT}/dev_manifest_a.tsv"
MANIFEST_B="${TMP_ROOT}/dev_manifest_b.tsv"
mkdir -p "${STORE_ROOT}"

TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-256}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-64}"

TMP_ROOT="${TMP_ROOT}" \
TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

tmp = Path(os.environ["TMP_ROOT"])
total = int(os.environ["TOTAL_SAMPLES"])
sample_bytes = int(os.environ["BYTES_PER_SAMPLE"])

def write_dataset(tag: bytes, data_path: Path, manifest_path: Path):
    with data_path.open("wb") as f:
        for _ in range(total):
            f.write(tag * sample_bytes)
    with manifest_path.open("w", encoding="utf-8") as m:
        for i in range(total):
            off = i * sample_bytes
            m.write(f"{i}\t{data_path}\t{off}\t{sample_bytes}\n")

write_dataset(b"A", tmp / "data_a.bin", tmp / "dev_manifest_a.tsv")
write_dataset(b"B", tmp / "data_b.bin", tmp / "dev_manifest_b.tsv")
PY

echo "[mx8] python mix gate"
TMP_ROOT="${TMP_ROOT}" \
TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
BYTES_PER_SAMPLE="${BYTES_PER_SAMPLE}" \
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DEV_MANIFEST_PATH_A="${MANIFEST_A}" \
MX8_DEV_MANIFEST_PATH_B="${MANIFEST_B}" \
MX8_DATASET_LINK_A="${TMP_ROOT}/ds_a@refresh" \
MX8_DATASET_LINK_B="${TMP_ROOT}/ds_b@refresh" \
MX8_MIX_SNAPSHOT=1 \
MX8_MIX_SNAPSHOT_PERIOD_TICKS="${MX8_MIX_SNAPSHOT_PERIOD_TICKS:-32}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m7_mix_gate.py"

echo "[mx8] mix_gate OK"
