#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MIN_SPS="${MX8_VIDEO_STAGE2C_MIN_SAMPLES_PER_SEC:-8}"
MAX_MS_PER_BATCH="${MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_BATCH:-2000}"
MAX_MS_PER_CLIP="${MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_CLIP:-120}"

echo "[mx8] stage2c perf gate thresholds"
echo "  min_samples_per_sec=${MIN_SPS}"
echo "  max_decode_ms_per_batch=${MAX_MS_PER_BATCH}"
echo "  max_decode_ms_per_clip=${MAX_MS_PER_CLIP}"

MX8_VIDEO_STAGE2C_MIN_SAMPLES_PER_SEC="${MIN_SPS}" \
MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_BATCH="${MAX_MS_PER_BATCH}" \
MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_CLIP="${MAX_MS_PER_CLIP}" \
  ./scripts/video_stage2b_stress_gate.sh

echo "[mx8] video_stage2c_perf_gate OK"
