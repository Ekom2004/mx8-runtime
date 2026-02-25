#!/usr/bin/env bash
set -euo pipefail

# Epoch-boundary training elasticity gate (v1 contract).
#
# Proves that training data delivery can run across epoch boundaries with
# membership changes (remove/add nodes between epochs), while each epoch still
# preserves no-overlap under fixed-step distributed consumption.
#
# Contract under test:
# - membership may change only between epochs (new coordinator run)
# - each epoch is deterministic for its fixed membership
# - no overlap across ranks is preserved
#
# Usage:
#   ./scripts/training_epoch_boundary_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-epoch-boundary-gate-XXXXXX")"
VENV_DIR="${TMP_ROOT}/venv"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "[mx8] venv python=${PYTHON_BIN}"
echo "[mx8] install deps (maturin)"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

echo "[mx8] build coordinator"
cargo build -p mx8-coordinator >/dev/null

STORE_ROOT="${TMP_ROOT}/store"
DATA_FILE="${TMP_ROOT}/data.bin"
DEV_MANIFEST="${TMP_ROOT}/dev_manifest.tsv"
mkdir -p "${STORE_ROOT}"

TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-2048}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-64}"
BATCH_SIZE_SAMPLES="${MX8_BATCH_SIZE_SAMPLES:-64}"
BLOCK_SIZE="${MX8_DEV_BLOCK_SIZE:-256}"
EPOCH_GATE_STEPS="${MX8_EPOCH_GATE_STEPS:-8}"
TOTAL_BYTES="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
JOB_ID_BASE="${MX8_JOB_ID_BASE:-m5-epoch-boundary}"
KEEP_TMP="${MX8_KEEP_TMP:-0}"

truncate -s "${TOTAL_BYTES}" "${DATA_FILE}"
rm -f "${DEV_MANIFEST}"
for ((i=0; i<"${TOTAL_SAMPLES}"; i++)); do
  off="$(( i * BYTES_PER_SAMPLE ))"
  printf "%s\t%s\t%s\t%s\n" "${i}" "${DATA_FILE}" "${off}" "${BYTES_PER_SAMPLE}" >> "${DEV_MANIFEST}"
done

COORD_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
COORD_URL="http://127.0.0.1:${COORD_PORT}"

start_coordinator() {
  local world_size="$1"
  local epoch="$2"
  local coord_log="$3"
  MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
  MX8_WORLD_SIZE="${world_size}" \
  MX8_MIN_WORLD_SIZE="${world_size}" \
  MX8_SHUFFLE=true \
  MX8_SEED=0 \
  MX8_EPOCH="${epoch}" \
  MX8_DATASET_LINK="${TMP_ROOT}/epoch-boundary@refresh" \
  MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
  MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
  MX8_DEV_BLOCK_SIZE="${BLOCK_SIZE}" \
  MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
  target/debug/mx8-coordinator --world-size "${world_size}" --min-world-size "${world_size}" >"${coord_log}" 2>&1 &
  echo "$!"
}

wait_coordinator_ready() {
  local pid="$1"
  local coord_log="$2"
  local tries=200
  for ((i=0; i<tries; i++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[mx8] coordinator exited early (pid=${pid})" >&2
      echo "[mx8] coordinator log (${coord_log}):" >&2
      tail -n 200 "${coord_log}" >&2 || true
      return 1
    fi
    if (echo >/dev/tcp/127.0.0.1/"${COORD_PORT}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "[mx8] coordinator not ready on 127.0.0.1:${COORD_PORT} (pid=${pid})" >&2
  echo "[mx8] coordinator log (${coord_log}):" >&2
  tail -n 200 "${coord_log}" >&2 || true
  return 1
}

stop_coordinator() {
  local pid="$1"
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}

COORD_PID=""
cleanup() {
  if [[ -n "${COORD_PID}" ]]; then
    stop_coordinator "${COORD_PID}" || true
    COORD_PID=""
  fi
  if [[ "${KEEP_TMP}" == "1" ]]; then
    echo "[mx8] keeping temp dir: ${TMP_ROOT}"
  else
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT

run_phase() {
  local phase="$1"
  local world_size="$2"
  local epoch="$3"
  local job_id="${JOB_ID_BASE}-p${phase}"
  local coord_log="${TMP_ROOT}/coordinator_phase${phase}.log"

  echo "[mx8] phase ${phase}: epoch=${epoch} world_size=${world_size}"
  COORD_PID="$(start_coordinator "${world_size}" "${epoch}" "${coord_log}")"
  wait_coordinator_ready "${COORD_PID}" "${coord_log}"

  MX8_COORD_URL="${COORD_URL}" \
  MX8_JOB_ID="${job_id}" \
  WORLD_SIZE="${world_size}" \
  MX8_TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
  MX8_EPOCH_GATE_STEPS="${EPOCH_GATE_STEPS}" \
  MX8_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
  MX8_DEV_LEASE_WANT=1 \
  MX8_PROGRESS_INTERVAL_MS=100 \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_epoch_boundary_membership_gate.py"

  stop_coordinator "${COORD_PID}"
  COORD_PID=""
}

# Phase 1: baseline membership.
run_phase 1 2 0
# Phase 2: remove node at epoch boundary.
run_phase 2 1 1
# Phase 3: add node back at next epoch boundary.
run_phase 3 2 2

echo "[mx8] training_epoch_boundary_gate OK"
