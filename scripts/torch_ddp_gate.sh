#!/usr/bin/env bash
set -euo pipefail

# Multi-process (DDP-like) gate: prove mx8 can feed PyTorch distributed training
# (static world size, restartable; v0 non-elastic).
#
# This runs everything locally with `torchrun --standalone` but uses the real
# mx8-coordinator gRPC API and `mx8.DistributedDataLoader` client.
#
# Usage:
#   ./scripts/torch_ddp_gate.sh
#
# Optional:
#   WORLD_SIZE=4 ./scripts/torch_ddp_gate.sh
#
# Prereqs:
# - python3 on PATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-torch-ddp-gate-XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"

echo "[mx8] install deps (torch + numpy + maturin)"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U torch numpy >/dev/null

STORE_ROOT="${TMP_ROOT}/store"
DATA_FILE="${TMP_ROOT}/data.bin"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"
mkdir -p "${STORE_ROOT}"

TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-20000}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-256}"
TOTAL_BYTES="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
truncate -s "${TOTAL_BYTES}" "${DATA_FILE}"

rm -f "${DEV_MANIFEST}"
for ((i=0; i<"${TOTAL_SAMPLES}"; i++)); do
  off="$(( i * BYTES_PER_SAMPLE ))"
  printf "%s\t%s\t%s\t%s\n" "${i}" "${DATA_FILE}" "${off}" "${BYTES_PER_SAMPLE}" >> "${DEV_MANIFEST}"
done

echo "[mx8] build coordinator"
cargo build -p mx8-coordinator >/dev/null

COORD_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

COORD_URL="http://127.0.0.1:${COORD_PORT}"
WORLD_SIZE="${WORLD_SIZE:-2}"
JOB_ID="${MX8_JOB_ID:-m5-ddp-demo}"

COORD_LOG="${TMP_ROOT}/coordinator.log"
echo "[mx8] starting coordinator (world_size=${WORLD_SIZE}, url=${COORD_URL})"
MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
MX8_WORLD_SIZE="${WORLD_SIZE}" \
MX8_DATASET_LINK="${TMP_ROOT}/dev@refresh" \
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
MX8_DEV_BLOCK_SIZE="${MX8_DEV_BLOCK_SIZE:-1000}" \
MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
target/debug/mx8-coordinator --world-size "${WORLD_SIZE}" >"${COORD_LOG}" 2>&1 &
COORD_PID="$!"
trap 'kill -TERM "${COORD_PID}" >/dev/null 2>&1 || true' EXIT

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

echo "[mx8] running ddp demo (local spawn)"
MX8_COORD_URL="${COORD_URL}" \
MX8_JOB_ID="${JOB_ID}" \
WORLD_SIZE="${WORLD_SIZE}" \
MX8_TORCH_STEPS="${MX8_TORCH_STEPS:-8}" \
MX8_TORCH_LR="${MX8_TORCH_LR:-0.01}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_demo.py"

echo "[mx8] torch_ddp_gate OK"
