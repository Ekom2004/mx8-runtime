#!/usr/bin/env bash
set -euo pipefail

# Unified NVDEC validation report:
# - runs existing NVDEC/video gates
# - captures per-gate logs and durations
# - emits one machine-readable + human-readable summary
#
# Profiles:
#   full     : fallback + parity + pressure + throughput
#   gpu      : parity + pressure + throughput (hardware required)
#   fallback : fallback-only checks
#
# Usage:
#   ./scripts/video_nvdec_validation_report.sh
#   MX8_NVDEC_VALIDATION_PROFILE=gpu ./scripts/video_nvdec_validation_report.sh
#   MX8_NVDEC_VALIDATION_PROFILE=fallback ./scripts/video_nvdec_validation_report.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PROFILE="${MX8_NVDEC_VALIDATION_PROFILE:-full}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${MX8_NVDEC_REPORT_DIR:-${ROOT}/.artifacts/nvdec-validation/${TS}}"

mkdir -p "${REPORT_DIR}"

RESULTS_CSV="${REPORT_DIR}/results.csv"
SUMMARY_MD="${REPORT_DIR}/summary.md"
SUMMARY_TXT="${REPORT_DIR}/summary.txt"
ENV_TXT="${REPORT_DIR}/environment.txt"

declare -a GATES=()

add_gate() {
  local gate_name="$1"
  local gate_script="$2"
  if [[ ! -x "${gate_script}" ]]; then
    echo "[mx8][nvdec-validation] gate script missing or not executable: ${gate_script}" >&2
    exit 1
  fi
  GATES+=("${gate_name}|${gate_script}")
}

case "${PROFILE}" in
  full)
    add_gate "nvdec_fallback" "${ROOT}/scripts/video_nvdec_fallback_gate.sh"
    add_gate "nvdec_compiled_fallback" "${ROOT}/scripts/video_nvdec_compiled_fallback_gate.sh"
    add_gate "video_backend_parity" "${ROOT}/scripts/video_stage3a_backend_gate.sh"
    add_gate "nvdec_pressure" "${ROOT}/scripts/video_nvdec_pressure_gate.sh"
    add_gate "nvdec_throughput" "${ROOT}/scripts/video_nvdec_throughput_gate.sh"
    ;;
  gpu)
    add_gate "video_backend_parity" "${ROOT}/scripts/video_stage3a_backend_gate.sh"
    add_gate "nvdec_pressure" "${ROOT}/scripts/video_nvdec_pressure_gate.sh"
    add_gate "nvdec_throughput" "${ROOT}/scripts/video_nvdec_throughput_gate.sh"
    : "${MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW:=1}"
    : "${MX8_VIDEO_NVDEC_MIN_SPEEDUP:=1.05}"
    export MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW
    export MX8_VIDEO_NVDEC_MIN_SPEEDUP
    ;;
  fallback)
    add_gate "nvdec_fallback" "${ROOT}/scripts/video_nvdec_fallback_gate.sh"
    add_gate "nvdec_compiled_fallback" "${ROOT}/scripts/video_nvdec_compiled_fallback_gate.sh"
    ;;
  *)
    echo "[mx8][nvdec-validation] invalid MX8_NVDEC_VALIDATION_PROFILE=${PROFILE} (expected: full|gpu|fallback)" >&2
    exit 1
    ;;
esac

if [[ "${MX8_NVDEC_INCLUDE_DIRECT_DECODE_STRESS:-0}" == "1" ]]; then
  add_gate "video_direct_decode_stress" "${ROOT}/scripts/video_direct_decode_stress_gate.sh"
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "profile=${PROFILE}"
  echo "report_dir=${REPORT_DIR}"
  echo "host=$(hostname || true)"
  echo "uname=$(uname -a || true)"
  echo "rustc=$(rustc --version 2>/dev/null || echo missing)"
  echo "python3=$(python3 --version 2>/dev/null || echo missing)"
  echo "ffmpeg=${MX8_FFMPEG_BIN:-ffmpeg}"
  echo "nvidia_smi=${MX8_NVIDIA_SMI_BIN:-nvidia-smi}"
  if command -v "${MX8_NVIDIA_SMI_BIN:-nvidia-smi}" >/dev/null 2>&1; then
    "${MX8_NVIDIA_SMI_BIN:-nvidia-smi}" --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi unavailable"
  fi
} > "${ENV_TXT}"

echo "gate,status,seconds,log" > "${RESULTS_CSV}"

fail_count=0
declare -a failed_gates=()

for entry in "${GATES[@]}"; do
  IFS='|' read -r gate_name gate_script <<< "${entry}"
  log_file="${REPORT_DIR}/${gate_name}.log"
  start_s="$(date +%s)"

  echo "[mx8][nvdec-validation] START ${gate_name}"
  if "${gate_script}" 2>&1 | tee "${log_file}"; then
    gate_status="PASS"
  else
    gate_status="FAIL"
    fail_count=$((fail_count + 1))
    failed_gates+=("${gate_name}")
  fi
  end_s="$(date +%s)"
  duration_s=$((end_s - start_s))
  printf "%s,%s,%s,%s\n" "${gate_name}" "${gate_status}" "${duration_s}" "${gate_name}.log" >> "${RESULTS_CSV}"
  echo "[mx8][nvdec-validation] ${gate_status} ${gate_name} (${duration_s}s)"
done

{
  echo "# MX8 NVDEC Validation Report"
  echo
  echo "- Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Profile: ${PROFILE}"
  echo "- Report dir: \`${REPORT_DIR}\`"
  if [[ "${PROFILE}" == "gpu" ]]; then
    echo "- Throughput hardware required: ${MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW}"
    echo "- Throughput min speedup: ${MX8_VIDEO_NVDEC_MIN_SPEEDUP}"
  fi
  echo
  echo "## Results"
  echo
  echo "| Gate | Status | Seconds | Log |"
  echo "|---|---|---:|---|"
  tail -n +2 "${RESULTS_CSV}" | while IFS=',' read -r gate status seconds log; do
    echo "| ${gate} | ${status} | ${seconds} | ${log} |"
  done
  echo
  echo "## Outcome"
  echo
  if [[ "${fail_count}" -eq 0 ]]; then
    echo "PASS"
  else
    echo "FAIL (${fail_count} gate(s))"
    echo
    echo "Failed gates:"
    for gate in "${failed_gates[@]}"; do
      echo "- ${gate}"
    done
  fi
} > "${SUMMARY_MD}"

{
  echo "mx8_nvdec_validation_outcome=$([[ "${fail_count}" -eq 0 ]] && echo PASS || echo FAIL)"
  echo "mx8_nvdec_validation_profile=${PROFILE}"
  echo "mx8_nvdec_validation_report_dir=${REPORT_DIR}"
  echo "mx8_nvdec_validation_summary=${SUMMARY_MD}"
  echo "mx8_nvdec_validation_results=${RESULTS_CSV}"
} > "${SUMMARY_TXT}"

echo "[mx8][nvdec-validation] report_dir=${REPORT_DIR}"
echo "[mx8][nvdec-validation] summary=${SUMMARY_MD}"
echo "[mx8][nvdec-validation] results=${RESULTS_CSV}"

if [[ "${fail_count}" -ne 0 ]]; then
  exit 1
fi

