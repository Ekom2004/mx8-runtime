#!/usr/bin/env bash
set -euo pipefail

# v1.8 stage-2b clean-environment reproducibility gate:
# - validates host prerequisites
# - runs stage2b reliability gate
# - runs stage2b decode-heavy stress gate
#
# Usage:
#   ./scripts/video_stage2b_clean_env_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[mx8] cargo not found on PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi
if ! command -v "${MX8_FFMPEG_BIN:-ffmpeg}" >/dev/null 2>&1; then
  echo "[mx8] ffmpeg not found on PATH (MX8_FFMPEG_BIN=${MX8_FFMPEG_BIN:-ffmpeg})" >&2
  exit 1
fi

echo "[mx8] stage2b clean-env prerequisites"
echo "  cargo: $(cargo --version)"
echo "  python: $(python3 --version)"
echo "  ffmpeg: $("${MX8_FFMPEG_BIN:-ffmpeg}" -version | head -n 1)"

echo "[mx8] run stage2b reliability gate"
MX8_VIDEO_STAGE2_BYTES_PER_CLIP="${MX8_VIDEO_STAGE2_BYTES_PER_CLIP:-2048}" \
  ./scripts/video_stage2b_gate.sh

echo "[mx8] run stage2b stress gate"
MX8_VIDEO_STAGE2_BYTES_PER_CLIP="${MX8_VIDEO_STAGE2B_STRESS_BYTES_PER_CLIP:-49152}" \
MX8_VIDEO_STAGE2B_STRESS_VIDEO_COUNT="${MX8_VIDEO_STAGE2B_STRESS_VIDEO_COUNT:-20}" \
MX8_VIDEO_STAGE2B_STRESS_FPS="${MX8_VIDEO_STAGE2B_STRESS_FPS:-12}" \
MX8_VIDEO_STAGE2B_STRESS_SECONDS="${MX8_VIDEO_STAGE2B_STRESS_SECONDS:-3}" \
MX8_VIDEO_STAGE2B_STRESS_CLIP_LEN="${MX8_VIDEO_STAGE2B_STRESS_CLIP_LEN:-8}" \
MX8_VIDEO_STAGE2B_STRESS_BATCH_SIZE="${MX8_VIDEO_STAGE2B_STRESS_BATCH_SIZE:-16}" \
MX8_VIDEO_STAGE2B_STRESS_MAX_BATCHES="${MX8_VIDEO_STAGE2B_STRESS_MAX_BATCHES:-64}" \
  ./scripts/video_stage2b_stress_gate.sh

echo "[mx8] video_stage2b_clean_env_gate OK"
