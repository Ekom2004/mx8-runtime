#!/usr/bin/env bash
set -euo pipefail

# Video GA gate runner.
#
# Usage:
#   ./scripts/video_ga_gate.sh --quick
#   ./scripts/video_ga_gate.sh --full
#
# --quick: fast developer loop before full run.
# --full: required go/no-go checklist for video GA.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:---full}"
case "${MODE}" in
  --quick|--full) ;;
  *)
    echo "Usage: ./scripts/video_ga_gate.sh [--quick|--full]" >&2
    exit 1
    ;;
esac

run_gate() {
  local name="$1"
  local cmd="$2"
  echo "[mx8][video-ga] ${name}"
  bash -lc "${cmd}"
}

if [[ "${MODE}" == "--quick" ]]; then
  run_gate "stage2b reliability gate" "./scripts/video_stage2b_gate.sh"
  run_gate "stage2c perf floor gate" "./scripts/video_stage2c_perf_gate.sh"
  run_gate "stage2d range planner gate" "./scripts/video_stage2d_range_gate.sh"
  run_gate "stage3a backend parity gate" "./scripts/video_stage3a_backend_gate.sh"
  echo "[mx8] video_ga_gate OK (mode=quick)"
  exit 0
fi

run_gate "stage1 indexing contract gate" "./scripts/video_stage1_gate.sh"
run_gate "stage2a public api gate" "./scripts/video_stage2a_gate.sh"
run_gate "stage2b clean environment gate" "./scripts/video_stage2b_clean_env_gate.sh"
run_gate "stage2c perf floor gate" "./scripts/video_stage2c_perf_gate.sh"
run_gate "stage2d range planner gate" "./scripts/video_stage2d_range_gate.sh"
run_gate "stage3a backend parity gate" "./scripts/video_stage3a_backend_gate.sh"

echo "[mx8] video_ga_gate OK (mode=full)"
