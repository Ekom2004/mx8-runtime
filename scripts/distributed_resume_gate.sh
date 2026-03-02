#!/usr/bin/env bash
set -euo pipefail

# Distributed checkpoint/resume gate.
# Proves that DistributedDataLoader.checkpoint() + resume_from can restart
# from a coordinator crash without sample overlap and still drain full coverage.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-distributed-resume-gate-XXXXXX")"

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
TOKEN_PATH="${TMP_ROOT}/resume.token"
SEEN_PATH="${TMP_ROOT}/seen_ids.json"
mkdir -p "${STORE_ROOT}"

TOTAL_SAMPLES="${MX8_TOTAL_SAMPLES:-4096}"
BYTES_PER_SAMPLE="${MX8_BYTES_PER_SAMPLE:-64}"
BATCH_SIZE_SAMPLES="${MX8_BATCH_SIZE_SAMPLES:-128}"
BLOCK_SIZE="${MX8_DEV_BLOCK_SIZE:-${BATCH_SIZE_SAMPLES}}"
PHASE1_BATCHES="${MX8_PHASE1_BATCHES:-8}"
WORLD_SIZE=1
JOB_ID="${MX8_JOB_ID:-m5-distributed-resume-gate}"
COORD_HA_ENABLE="${MX8_COORD_HA_ENABLE:-false}"
COORD_STATE_STORE_ENABLE="${MX8_COORD_STATE_STORE_ENABLE:-false}"
COORD_HA_LEASE_PATH="${TMP_ROOT}/coordinator_ha.leader_lease"
COORD_STATE_STORE_PATH_BASE="${TMP_ROOT}/coordinator_state"
COORD_START_SEQ=0
COORD_LEADER_ID=""

TOTAL_BYTES="$(( TOTAL_SAMPLES * BYTES_PER_SAMPLE ))"
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
  local coord_log="$1"
  local state_store_path=""
  COORD_START_SEQ=$((COORD_START_SEQ + 1))
  COORD_LEADER_ID="mx8-distributed-resume-gate-${COORD_START_SEQ}"
  state_store_path="${COORD_STATE_STORE_PATH_BASE}-${COORD_START_SEQ}.json"
  MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
  MX8_WORLD_SIZE="${WORLD_SIZE}" \
  MX8_MIN_WORLD_SIZE="${WORLD_SIZE}" \
  MX8_SHUFFLE=false \
  MX8_SEED=0 \
  MX8_EPOCH=0 \
  MX8_DATASET_LINK="${TMP_ROOT}/resume@refresh" \
  MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
  MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
  MX8_DEV_BLOCK_SIZE="${BLOCK_SIZE}" \
  MX8_LEASE_LOG_PATH=none \
  MX8_COORD_HA_ENABLE="${COORD_HA_ENABLE}" \
  MX8_COORD_HA_LEASE_PATH="${COORD_HA_LEASE_PATH}" \
  MX8_COORD_HA_LEADER_ID="${COORD_LEADER_ID}" \
  MX8_COORD_STATE_STORE_ENABLE="${COORD_STATE_STORE_ENABLE}" \
  MX8_COORD_STATE_STORE_PATH="${state_store_path}" \
  MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
  target/debug/mx8-coordinator --world-size "${WORLD_SIZE}" --min-world-size "${WORLD_SIZE}" >"${coord_log}" 2>&1 &
  COORD_PID="$!"
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

wait_coordinator_leader() {
  local pid="$1"
  local coord_log="$2"
  local expected_leader_id="$3"
  local tries=300
  for ((i=0; i<tries; i++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[mx8] coordinator exited before write-leader acquisition (pid=${pid})" >&2
      echo "[mx8] coordinator log (${coord_log}):" >&2
      tail -n 200 "${coord_log}" >&2 || true
      return 1
    fi

    local observed_leader_id=""
    observed_leader_id="$(awk -F= '/^leader_id=/{print $2; exit}' "${COORD_HA_LEASE_PATH}" 2>/dev/null || true)"
    if [[ "${observed_leader_id}" == "${expected_leader_id}" ]]; then
      return 0
    fi
    sleep 0.1
  done
  echo "[mx8] coordinator did not become write leader in time (pid=${pid}, expected_leader_id=${expected_leader_id})" >&2
  if [[ -f "${COORD_HA_LEASE_PATH}" ]]; then
    echo "[mx8] current lease file (${COORD_HA_LEASE_PATH}):" >&2
    tail -n 20 "${COORD_HA_LEASE_PATH}" >&2 || true
  fi
  echo "[mx8] coordinator log (${coord_log}):" >&2
  tail -n 200 "${coord_log}" >&2 || true
  return 1
}

stop_coordinator() {
  local pid="$1"
  local sig="${2:-TERM}"
  kill -"${sig}" "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}

COORD_PID=""
trap '[[ -n "${COORD_PID}" ]] && stop_coordinator "${COORD_PID}" TERM || true; rm -rf "${TMP_ROOT}"' EXIT

COORD_LOG1="${TMP_ROOT}/coordinator_phase1.log"
echo "[mx8] starting coordinator phase1 (url=${COORD_URL})"
start_coordinator "${COORD_LOG1}"
wait_coordinator_ready "${COORD_PID}" "${COORD_LOG1}"
if [[ "${COORD_HA_ENABLE}" == "true" ]]; then
  wait_coordinator_leader "${COORD_PID}" "${COORD_LOG1}" "${COORD_LEADER_ID}"
fi

echo "[mx8] capture checkpoint token"
MX8_COORD_URL="${COORD_URL}" \
MX8_JOB_ID="${JOB_ID}" \
MX8_NODE_ID="rank0-phase1" \
MX8_TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
MX8_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
MX8_PHASE1_BATCHES="${PHASE1_BATCHES}" \
MX8_DEV_LEASE_WANT=1 \
MX8_PROGRESS_INTERVAL_MS=100 \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_distributed_resume_gate.py" \
  --mode capture \
  --token-path "${TOKEN_PATH}" \
  --seen-path "${SEEN_PATH}"

echo "[mx8] simulate coordinator crash"
stop_coordinator "${COORD_PID}" KILL
COORD_PID=""

COORD_LOG2="${TMP_ROOT}/coordinator_phase2.log"
echo "[mx8] restart coordinator phase2"
start_coordinator "${COORD_LOG2}"
wait_coordinator_ready "${COORD_PID}" "${COORD_LOG2}"
if [[ "${COORD_HA_ENABLE}" == "true" ]]; then
  wait_coordinator_leader "${COORD_PID}" "${COORD_LOG2}" "${COORD_LEADER_ID}"
fi

echo "[mx8] resume from checkpoint token"
MX8_COORD_URL="${COORD_URL}" \
MX8_JOB_ID="${JOB_ID}" \
MX8_NODE_ID="rank0-phase2" \
MX8_TOTAL_SAMPLES="${TOTAL_SAMPLES}" \
MX8_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
MX8_DEV_LEASE_WANT=1 \
MX8_PROGRESS_INTERVAL_MS=100 \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_distributed_resume_gate.py" \
  --mode resume \
  --token-path "${TOKEN_PATH}" \
  --seen-path "${SEEN_PATH}"

stop_coordinator "${COORD_PID}" TERM
COORD_PID=""

echo "[mx8] distributed_resume_gate OK"
