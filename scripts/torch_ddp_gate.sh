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
#   MX8_TORCH_DDP_AUTOTUNE_AB=1 ./scripts/torch_ddp_gate.sh
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

start_coordinator() {
  local coord_log="$1"
  MX8_COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}" \
  MX8_WORLD_SIZE="${WORLD_SIZE}" \
  MX8_SHUFFLE="${MX8_SHUFFLE:-true}" \
  MX8_SEED="${MX8_SEED:-0}" \
  MX8_EPOCH="${MX8_EPOCH:-0}" \
  MX8_DATASET_LINK="${TMP_ROOT}/dev@refresh" \
  MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
  MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST}" \
  MX8_DEV_BLOCK_SIZE="${MX8_DEV_BLOCK_SIZE:-1000}" \
  MX8_METRICS_SNAPSHOT_INTERVAL_MS=0 \
  target/debug/mx8-coordinator --world-size "${WORLD_SIZE}" >"${coord_log}" 2>&1 &
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
trap '[[ -n "${COORD_PID}" ]] && stop_coordinator "${COORD_PID}"' EXIT

echo "[mx8] maturin develop"
(
  cd "${ROOT}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml
)

COORD_LOG="${TMP_ROOT}/coordinator.log"
echo "[mx8] starting coordinator (world_size=${WORLD_SIZE}, url=${COORD_URL})"
COORD_PID="$(start_coordinator "${COORD_LOG}")"
wait_coordinator_ready "${COORD_PID}" "${COORD_LOG}"

echo "[mx8] running ddp demo (local spawn)"
MX8_COORD_URL="${COORD_URL}" \
MX8_JOB_ID="${JOB_ID}" \
WORLD_SIZE="${WORLD_SIZE}" \
MX8_TORCH_STEPS="${MX8_TORCH_STEPS:-8}" \
MX8_TORCH_LR="${MX8_TORCH_LR:-0.01}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_demo.py"

stop_coordinator "${COORD_PID}"
COORD_PID=""

if [[ "${MX8_TORCH_DDP_NODUPES:-0}" == "1" ]]; then
  COORD_LOG="${TMP_ROOT}/coordinator_nodupes.log"
  echo "[mx8] starting coordinator (nodupes, world_size=${WORLD_SIZE}, url=${COORD_URL})"
  COORD_PID="$(start_coordinator "${COORD_LOG}")"
  wait_coordinator_ready "${COORD_PID}" "${COORD_LOG}"
  echo "[mx8] running ddp nodupes gate (local spawn)"
  MX8_COORD_URL="${COORD_URL}" \
  MX8_JOB_ID="${MX8_JOB_ID_NODUPES:-m5-ddp-nodupes}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  MX8_TORCH_NODUPES_STEPS="${MX8_TORCH_NODUPES_STEPS:-64}" \
  MX8_DEV_LEASE_WANT="${MX8_DEV_LEASE_WANT:-1}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_nodupes_gate.py"
  stop_coordinator "${COORD_PID}"
  COORD_PID=""
fi

if [[ "${MX8_TORCH_DDP_AUTOTUNE_AB:-0}" == "1" ]]; then
  echo "[mx8] running ddp autotune AB gate (baseline vs autotune)"

  AB_STEPS="${MX8_TORCH_AB_STEPS:-96}"
  AB_WANT="${MX8_TORCH_AB_WANT:-1}"
  AB_PREFETCH="${MX8_TORCH_AB_PREFETCH_BATCHES:-1}"
  AB_MAX_QUEUE="${MX8_TORCH_AB_MAX_QUEUE_BATCHES:-1}"
  AB_MAX_INFLIGHT="${MX8_TORCH_AB_MAX_INFLIGHT_BYTES:-67108864}"
  AB_MAX_PROCESS_RSS="${MX8_TORCH_AB_MAX_PROCESS_RSS_BYTES:-1073741824}"
  AB_COMPUTE_MS="${MX8_TORCH_AB_COMPUTE_MS:-5}"
  AB_MIN_IMPROVEMENT="${MX8_TORCH_AB_MIN_WAIT_IMPROVEMENT:-0.0}"

  run_ab_mode() {
    local mode="$1"
    local autotune_flag="$2"
    local profile="$3"
    local coord_log="${TMP_ROOT}/coordinator_autotune_ab_${mode}.log"
    local out_log="${TMP_ROOT}/ddp_autotune_ab_${mode}.log"

    echo "[mx8] starting coordinator (autotune_ab mode=${mode})" >&2
    COORD_PID="$(start_coordinator "${coord_log}")"
    wait_coordinator_ready "${COORD_PID}" "${coord_log}"

    MX8_COORD_URL="${COORD_URL}" \
    MX8_JOB_ID="${MX8_JOB_ID_AUTOTUNE_AB:-m5-ddp-autotune-ab}-${mode}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    MX8_TORCH_AB_STEPS="${AB_STEPS}" \
    MX8_DEV_LEASE_WANT="${AB_WANT}" \
    MX8_PREFETCH_BATCHES="${AB_PREFETCH}" \
    MX8_MAX_QUEUE_BATCHES="${AB_MAX_QUEUE}" \
    MX8_MAX_INFLIGHT_BYTES="${AB_MAX_INFLIGHT}" \
    MX8_MAX_PROCESS_RSS_BYTES="${AB_MAX_PROCESS_RSS}" \
    MX8_TORCH_AB_COMPUTE_MS="${AB_COMPUTE_MS}" \
    MX8_AUTOTUNE="${autotune_flag}" \
    MX8_AUTOTUNE_PROFILE="${profile}" \
    MX8_AUTOTUNE_AB_MODE="${mode}" \
    "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_autotune_ab_gate.py" | tee "${out_log}" >&2

    stop_coordinator "${COORD_PID}"
    COORD_PID=""

    local wait_ratio
    wait_ratio="$(grep -E '^wait_ratio:' "${out_log}" | head -n 1 | sed -e 's/^wait_ratio: //')"
    if [[ -z "${wait_ratio}" ]]; then
      echo "[mx8] autotune AB gate: failed to parse wait_ratio from ${out_log}" >&2
      exit 1
    fi
    echo "${wait_ratio}"
  }

  baseline_wait="$(run_ab_mode baseline 0 safe)"
  autotune_wait="$(run_ab_mode autotune 1 throughput)"

  python3 - "${baseline_wait}" "${autotune_wait}" "${AB_MIN_IMPROVEMENT}" <<'PY'
import sys

baseline = float(sys.argv[1])
autotune = float(sys.argv[2])
min_improvement = float(sys.argv[3])

if baseline <= 0.0:
    if autotune > 0.01:
        raise SystemExit(
            f"[mx8] autotune AB gate FAILED: baseline wait_ratio={baseline:.6f}, autotune={autotune:.6f}"
        )
else:
    target = baseline * (1.0 - min_improvement)
    if autotune > target:
        raise SystemExit(
            "[mx8] autotune AB gate FAILED: "
            f"baseline={baseline:.6f}, autotune={autotune:.6f}, "
            f"required<={target:.6f} (min_improvement={min_improvement:.3f})"
        )

print(
    "[mx8] autotune AB gate OK: "
    f"baseline_wait_ratio={baseline:.6f} autotune_wait_ratio={autotune:.6f}"
)
PY
fi

if [[ "${MX8_TORCH_DDP_DETERMINISM:-0}" == "1" ]]; then
  echo "[mx8] running ddp determinism gate (same seed/epoch -> same digests; different epoch -> different digests)"

  BATCH_SIZE_SAMPLES="${MX8_TORCH_BATCH_SIZE_SAMPLES:-32}"
  DET_STEPS="${MX8_TORCH_NODUPES_STEPS:-$(( (TOTAL_SAMPLES / WORLD_SIZE) / BATCH_SIZE_SAMPLES ))}"
  if [[ "${DET_STEPS}" -le 0 ]]; then
    DET_STEPS=1
  fi

  run_nodupes_with_epoch() {
    local epoch="$1"
    local tag="$2"
    local coord_log="${TMP_ROOT}/coordinator_determinism_${tag}.log"
    local out_log="${TMP_ROOT}/ddp_nodupes_${tag}.log"

    echo "[mx8] starting coordinator (determinism tag=${tag}, epoch=${epoch})" >&2
    MX8_EPOCH="${epoch}" COORD_PID="$(start_coordinator "${coord_log}")"
    wait_coordinator_ready "${COORD_PID}" "${coord_log}"
    MX8_COORD_URL="${COORD_URL}" \
    MX8_JOB_ID="${MX8_JOB_ID_DETERMINISM:-m5-ddp-determinism}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    MX8_TORCH_NODUPES_STEPS="${DET_STEPS}" \
    MX8_TORCH_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
    MX8_DEV_LEASE_WANT=1 \
    "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_nodupes_gate.py" | tee "${out_log}" >&2

    stop_coordinator "${COORD_PID}"
    COORD_PID=""

    local digests
    digests="$(grep -E '^digests:' "${out_log}" | head -n 1 | sed -e 's/^digests: //')"
    if [[ -z "${digests}" ]]; then
      echo "[mx8] determinism gate: failed to parse digests from ${out_log}" >&2
      exit 1
    fi
    echo "${digests}"
  }

  SEED="${MX8_SEED:-0}"
  echo "[mx8] determinism config: seed=${SEED} (MX8_SEED), shuffle=${MX8_SHUFFLE:-true} (MX8_SHUFFLE)"

  base_a="$(run_nodupes_with_epoch 0 a)"
  base_b="$(run_nodupes_with_epoch 0 b)"

  if [[ "${base_a}" != "${base_b}" ]]; then
    echo "[mx8] determinism gate FAILED: same seed/epoch produced different digests" >&2
    echo "[mx8] a=${base_a}" >&2
    echo "[mx8] b=${base_b}" >&2
    exit 1
  fi

  epoch1="$(run_nodupes_with_epoch 1 e1)"
  if [[ "${epoch1}" == "${base_a}" ]]; then
    epoch2="$(run_nodupes_with_epoch 2 e2)"
    if [[ "${epoch2}" == "${base_a}" ]]; then
      echo "[mx8] determinism gate FAILED: epoch change did not change digests" >&2
      echo "[mx8] base=${base_a}" >&2
      echo "[mx8] e1=${epoch1}" >&2
      echo "[mx8] e2=${epoch2}" >&2
      exit 1
    fi
  fi

  echo "[mx8] determinism gate OK"
fi

if [[ "${MX8_TORCH_DDP_RESTART:-0}" == "1" ]]; then
  echo "[mx8] running ddp restartability gate (v0 epoch-level restart)"

  BATCH_SIZE_SAMPLES="${MX8_TORCH_BATCH_SIZE_SAMPLES:-32}"
  DET_STEPS="${MX8_TORCH_NODUPES_STEPS:-$(( (TOTAL_SAMPLES / WORLD_SIZE) / BATCH_SIZE_SAMPLES ))}"
  if [[ "${DET_STEPS}" -le 0 ]]; then
    DET_STEPS=1
  fi

  run_nodupes_epoch0() {
    local tag="$1"
    local coord_log="${TMP_ROOT}/coordinator_restart_${tag}.log"
    local out_log="${TMP_ROOT}/ddp_restart_nodupes_${tag}.log"

    echo "[mx8] starting coordinator (restart tag=${tag}, epoch=0)" >&2
    MX8_EPOCH=0 COORD_PID="$(start_coordinator "${coord_log}")"
    wait_coordinator_ready "${COORD_PID}" "${coord_log}"

    MX8_COORD_URL="${COORD_URL}" \
    MX8_JOB_ID="${MX8_JOB_ID_RESTART:-m5-ddp-restart}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    MX8_TORCH_NODUPES_STEPS="${DET_STEPS}" \
    MX8_TORCH_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
    MX8_DEV_LEASE_WANT=1 \
    "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_nodupes_gate.py" | tee "${out_log}" >&2

    stop_coordinator "${COORD_PID}"
    COORD_PID=""

    local digests
    digests="$(grep -E '^digests:' "${out_log}" | head -n 1 | sed -e 's/^digests: //')"
    if [[ -z "${digests}" ]]; then
      echo "[mx8] restart gate: failed to parse digests from ${out_log}" >&2
      exit 1
    fi
    echo "${digests}"
  }

  base="$(run_nodupes_epoch0 base)"

  CKPT_PATH="${TMP_ROOT}/ckpt.pt"
  COORD_LOG="${TMP_ROOT}/coordinator_restart_crash.log"
  echo "[mx8] starting coordinator (restart crash run, epoch=0)" >&2
  MX8_EPOCH=0 COORD_PID="$(start_coordinator "${COORD_LOG}")"
  wait_coordinator_ready "${COORD_PID}" "${COORD_LOG}"

  echo "[mx8] running crash train (expect DDP failure; checkpoint should exist)" >&2
  set +e
  MX8_COORD_URL="${COORD_URL}" \
  MX8_JOB_ID="${MX8_JOB_ID_RESTART:-m5-ddp-restart}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  MX8_TORCH_STEPS="${MX8_TORCH_RESTART_STEPS:-8}" \
  MX8_TORCH_LR="${MX8_TORCH_LR:-0.01}" \
  MX8_TORCH_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
  MX8_TORCH_SAVE_CKPT_PATH="${CKPT_PATH}" \
  MX8_TORCH_SAVE_CKPT_STEP="${MX8_TORCH_SAVE_CKPT_STEP:-2}" \
  MX8_TORCH_CRASH_RANK="${MX8_TORCH_CRASH_RANK:-1}" \
  MX8_TORCH_CRASH_AFTER_STEP="${MX8_TORCH_CRASH_AFTER_STEP:-3}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_ckpt_crash_demo.py"
  crash_rc="$?"
  set -e

  stop_coordinator "${COORD_PID}"
  COORD_PID=""

  if [[ "${crash_rc}" -eq 0 ]]; then
    echo "[mx8] restart gate FAILED: crash run unexpectedly succeeded" >&2
    exit 1
  fi
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[mx8] restart gate FAILED: expected checkpoint at ${CKPT_PATH}" >&2
    exit 1
  fi

  replay="$(run_nodupes_epoch0 replay)"
  if [[ "${base}" != "${replay}" ]]; then
    echo "[mx8] restart gate FAILED: epoch replay digests mismatch" >&2
    echo "[mx8] base=${base}" >&2
    echo "[mx8] replay=${replay}" >&2
    exit 1
  fi

  COORD_LOG="${TMP_ROOT}/coordinator_restart_load.log"
  echo "[mx8] starting coordinator (restart load run, epoch=0)" >&2
  MX8_EPOCH=0 COORD_PID="$(start_coordinator "${COORD_LOG}")"
  wait_coordinator_ready "${COORD_PID}" "${COORD_LOG}"

  echo "[mx8] running load-from-checkpoint train (should succeed)" >&2
  MX8_COORD_URL="${COORD_URL}" \
  MX8_JOB_ID="${MX8_JOB_ID_RESTART:-m5-ddp-restart}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  MX8_TORCH_STEPS="${MX8_TORCH_RESTART_LOAD_STEPS:-4}" \
  MX8_TORCH_LR="${MX8_TORCH_LR:-0.01}" \
  MX8_TORCH_BATCH_SIZE_SAMPLES="${BATCH_SIZE_SAMPLES}" \
  MX8_TORCH_LOAD_CKPT_PATH="${CKPT_PATH}" \
  "${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m5_torch_ddp_ckpt_crash_demo.py" | tee "${TMP_ROOT}/ddp_restart_load.log" >&2

  stop_coordinator "${COORD_PID}"
  COORD_PID=""

  if ! grep -qE '^loaded_checkpoint: True$' "${TMP_ROOT}/ddp_restart_load.log"; then
    echo "[mx8] restart gate FAILED: did not confirm checkpoint load" >&2
    exit 1
  fi

  echo "[mx8] restart gate OK"
fi

echo "[mx8] torch_ddp_gate OK"
