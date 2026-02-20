#!/usr/bin/env bash
set -euo pipefail

# Repo smoke runner: quick "main is healthy" gate.
#
# Usage:
#   ./scripts/smoke.sh
#   MX8_SMOKE_OFFLINE=1 ./scripts/smoke.sh
#
# Notes:
# - Demo 2 is an internal correctness gate (kill-a-node lease recovery).
# - Demo 3 is an internal performance+correctness gate (prefetch hides latency, bounded caps).

OFFLINE="${MX8_SMOKE_OFFLINE:-0}"

if [[ "${OFFLINE}" == "1" ]]; then
  MX8_CHECK_OFFLINE=1 ./scripts/check.sh
else
  ./scripts/check.sh
fi

echo "[mx8] demo2 (leases + kill-a-node recovery)"
RUST_LOG=info \
  MX8_TOTAL_SAMPLES=8000 \
  MX8_DEV_BLOCK_SIZE=1000 \
  MX8_SINK_SLEEP_MS=25 \
  MX8_KILL_AFTER_MS=25 \
  MX8_WAIT_REQUEUE_TIMEOUT_MS=15000 \
  MX8_WAIT_DRAIN_TIMEOUT_MS=30000 \
  cargo run -p mx8-runtime --bin mx8-demo2

echo "[mx8] demo3 (prefetch + retry under injected latency)"
# Keep the demo quick while still exercising retry/backoff + prefetch.
RUST_LOG=warn \
  MX8_HTTP_LATENCY_MS=5 \
  MX8_HTTP_FAIL_EVERY_N=7 \
  MX8_TOTAL_SAMPLES=256 \
  MX8_PREFETCH_COMPARE=8 \
  cargo run -p mx8-runtime --bin mx8-demo3

if [[ "${MX8_SMOKE_MINIO:-0}" == "1" ]]; then
  echo "[mx8] minio_gate (S3-compatible fetch via docker)"
  ./scripts/minio_gate.sh
fi

if [[ "${MX8_SMOKE_DEMO2_MINIO:-0}" == "1" ]]; then
  echo "[mx8] demo2_minio (distributed lease recovery + MinIO bytes)"
  ./scripts/demo2_minio.sh
fi

if [[ "${MX8_SMOKE_DEMO2_MINIO_SCALE:-0}" == "1" ]]; then
  echo "[mx8] demo2_minio_scale (scale gate: >4MiB manifest + MinIO + recovery)"
  ./scripts/demo2_minio_scale.sh
fi

if [[ "${MX8_SMOKE_MINIO_MANIFEST_STORE:-0}" == "1" ]]; then
  echo "[mx8] minio_manifest_store_gate (S3 manifest_store locks + deterministic hash)"
  ./scripts/minio_manifest_store_gate.sh
fi

if [[ "${MX8_SMOKE_MINIO_S3_PREFIX_SNAPSHOT:-0}" == "1" ]]; then
  echo "[mx8] minio_s3_prefix_snapshot_gate (LIST S3 prefix -> pinned snapshot)"
  ./scripts/minio_s3_prefix_snapshot_gate.sh
fi

if [[ "${MX8_SMOKE_MINIO_S3_PREFIX_RECURSIVE:-0}" == "1" ]]; then
  echo "[mx8] minio_s3_prefix_recursive_gate (recursive vs top-level indexing)"
  ./scripts/minio_s3_prefix_recursive_gate.sh
fi

if [[ "${MX8_SMOKE_MINIO_VIDEO_STAGE1:-0}" == "1" ]]; then
  echo "[mx8] minio_video_stage1_gate (S3 stage1 video metadata extraction)"
  ./scripts/minio_video_stage1_gate.sh
fi

if [[ "${MX8_SMOKE_MINIO_PACK:-0}" == "1" ]]; then
  echo "[mx8] minio_pack_gate (pack S3 prefix -> tar shards + manifest)"
  ./scripts/minio_pack_gate.sh
fi

if [[ "${MX8_SMOKE_PY_MINIO_IMAGEFOLDER:-0}" == "1" ]]; then
  echo "[mx8] py_minio_imagefolder_gate (Python DataLoader + MinIO labels)"
  ./scripts/py_minio_imagefolder_gate.sh
fi

if [[ "${MX8_SMOKE_PY_VISION_PILLOW:-0}" == "1" ]]; then
  echo "[mx8] py_minio_vision_pillow_gate (Pillow decode + torch train)"
  ./scripts/py_minio_vision_pillow_gate.sh
fi

if [[ "${MX8_SMOKE_PY_LOCAL_VISION_PILLOW:-0}" == "1" ]]; then
  echo "[mx8] py_local_vision_pillow_gate (no S3; pack_dir + torch train)"
  ./scripts/py_local_vision_pillow_gate.sh
fi

if [[ "${MX8_SMOKE_PY_VISION_DECODE_BENCH:-0}" == "1" ]]; then
  echo "[mx8] py_vision_decode_backend_bench (rust vs python decode perf compare)"
  ./scripts/py_vision_decode_backend_bench.sh
fi

if [[ "${MX8_SMOKE_SOAK_DEMO2_MINIO_SCALE:-0}" == "1" ]]; then
  echo "[mx8] soak_demo2_minio_scale (soak gate: repeated node failures)"
  ./scripts/soak_demo2_minio_scale.sh
fi

if [[ "${MX8_SMOKE_TORCH_DDP_AUTOTUNE_AB:-0}" == "1" ]]; then
  echo "[mx8] torch_ddp_gate (multi-process PyTorch autotune AB gate)"
  MX8_TORCH_DDP_AUTOTUNE_AB=1 ./scripts/torch_ddp_gate.sh
elif [[ "${MX8_SMOKE_TORCH_DDP_DETERMINISM:-0}" == "1" ]]; then
  echo "[mx8] torch_ddp_gate (multi-process PyTorch determinism gate)"
  MX8_TORCH_DDP_DETERMINISM=1 ./scripts/torch_ddp_gate.sh
elif [[ "${MX8_SMOKE_TORCH_DDP_RESTART:-0}" == "1" ]]; then
  echo "[mx8] torch_ddp_gate (multi-process PyTorch restartability gate)"
  MX8_TORCH_DDP_RESTART=1 ./scripts/torch_ddp_gate.sh
elif [[ "${MX8_SMOKE_TORCH_DDP_NODUPES:-0}" == "1" ]]; then
  echo "[mx8] torch_ddp_gate (multi-process PyTorch no-overlap gate)"
  MX8_TORCH_DDP_NODUPES=1 ./scripts/torch_ddp_gate.sh
elif [[ "${MX8_SMOKE_TORCH_DDP:-0}" == "1" ]]; then
  echo "[mx8] torch_ddp_gate (multi-process PyTorch training gate)"
  ./scripts/torch_ddp_gate.sh
fi

if [[ "${MX8_SMOKE_AUTOTUNE_PRESSURE_SIM:-0}" == "1" ]]; then
  echo "[mx8] autotune_memory_pressure_sim (simulated RSS pressure reaction gate)"
  ./scripts/autotune_memory_pressure_sim.sh
fi

if [[ "${MX8_SMOKE_MIX:-0}" == "1" ]]; then
  echo "[mx8] mix_gate (v1.7 weighted round-robin scheduler gate)"
  ./scripts/mix_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE1:-0}" == "1" ]]; then
  echo "[mx8] video_stage1_gate (v1.8 stage-1 clip-index contract gate)"
  ./scripts/video_stage1_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2A:-0}" == "1" ]]; then
  echo "[mx8] video_stage2a_gate (v1.8 stage-2a cpu video API gate)"
  ./scripts/video_stage2a_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2B:-0}" == "1" ]]; then
  echo "[mx8] video_stage2b_gate (v1.8 stage-2b cpu decode reliability scaffold gate)"
  ./scripts/video_stage2b_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2B_STRESS:-0}" == "1" ]]; then
  echo "[mx8] video_stage2b_stress_gate (v1.8 stage-2b decode-heavy bounded stress gate)"
  ./scripts/video_stage2b_stress_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2B_CLEAN_ENV:-0}" == "1" ]]; then
  echo "[mx8] video_stage2b_clean_env_gate (v1.8 stage-2b clean-environment reproducibility gate)"
  ./scripts/video_stage2b_clean_env_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2C_PERF:-0}" == "1" ]]; then
  echo "[mx8] video_stage2c_perf_gate (v1.8 stage-2c decode perf threshold gate)"
  ./scripts/video_stage2c_perf_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE2D_RANGE:-0}" == "1" ]]; then
  echo "[mx8] video_stage2d_range_gate (v1.8 stage-2d S3 range planner contract gate)"
  ./scripts/video_stage2d_range_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_STAGE3A:-0}" == "1" ]]; then
  echo "[mx8] video_stage3a_backend_gate (v1.8 stage-3a decode backend parity gate)"
  ./scripts/video_stage3a_backend_gate.sh
fi

if [[ "${MX8_SMOKE_VIDEO_GA:-0}" == "1" ]]; then
  echo "[mx8] video_ga_gate (video go/no-go checklist)"
  ./scripts/video_ga_gate.sh "${MX8_SMOKE_VIDEO_GA_MODE:---quick}"
fi

echo "[mx8] smoke OK"
