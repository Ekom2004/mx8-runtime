# NVDEC Decoder Design (M8.1)

Updated: 2026-03-03

## Goal

Ship NVIDIA GPU video decode for MX8 with higher throughput than CPU decode while preserving current runtime safety and API contracts.

## Scope

In scope (M8.1):

- Add an NVDEC backend to the video loader decode path.
- Keep output contract unchanged: `THWC`, `u8`, `rgb24`, host-memory buffer.
- Preserve fail-open behavior: any NVDEC failure falls back to CPU decode backend.
- Extend video runtime autotune with GPU pressure so VRAM risk is controlled.
- Add observability and release gates for fallback, parity, pressure safety, and throughput.

Out of scope (M8.1):

- Zero-copy CUDA tensor output.
- New Python API surface for device-native video batches.
- Cross-vendor GPU decode (handled in follow-up milestones).

Zero-copy follow-up starts with a spike gate (no production contract change):

- `./scripts/zero_copy_from_blob_spike.sh`
- This validates `torch::from_blob` CUDA-pointer custom deleter timing across multiple CUDA streams before any API/implementation commitment.
- Full fix-first implementation plan is in `docs/zero_copy_v2_fix_first.md` (Phases 1-4).

## Backend Model

`MX8_VIDEO_DECODE_BACKEND` supports:

- `auto`
- `cli`
- `ffi`
- `nvdec`

Current default remains `cli`.

`auto` resolution order on Linux/NVIDIA:

1. `nvdec`
2. `ffi`
3. `cli`

If `nvdec` is requested but unavailable/not compiled/not supported for the input stream, MX8 logs a proof fallback event and continues via CPU backend.

## Output Contract

The decoded output contract does not change:

- Layout: `THWC`
- Dtype: `u8`
- Colorspace: `rgb24`
- Storage: host memory (`Vec<u8>` before Python handoff)

NVDEC path:

1. Decode compressed bitstream on GPU.
2. Convert decoded surfaces to RGB24.
3. Copy final clip bytes to host buffer.
4. Return through the same pack/deliver path as existing backends.

This keeps all higher-level batching, checkpointing, and stats semantics unchanged.

Current implementation path:

- FFmpeg CLI with CUDA hwaccel flags (`-hwaccel cuda`, `-hwaccel_output_format cuda`) under `mx8_video_nvdec` builds.
- If CUDA/NVIDIA support is unavailable on the host FFmpeg/runtime, backend fails open to `ffi`/`cli`.
- GPU pressure is sampled from `nvidia-smi` (override path via `MX8_NVIDIA_SMI_BIN`), with deterministic gate override via `MX8_VIDEO_GPU_PRESSURE_RATIO`.
- Sampling is rate-limited to at most one `nvidia-smi` query every 2 seconds; autotune ticks between samples reuse the cached value.
- Experimental v2 scaffolding: `MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT=1` makes `VideoBatch.to_torch()` allocate Torch-owned CUDA output and use explicit stream plumbing (`current_stream`, non-blocking copy, `record_stream`) with fail-open CPU fallback on runtime write/stream errors.
- `MX8_VIDEO_EXPERIMENTAL_DEVICE_DIRECT_WRITE=1` enables a direct-write scaffold mode that first attempts a native destination-writer hook via `torch.ops.mx8_video.direct_write_u8(...)`; current native-op path enqueues stream-bound raw pointer copies (`cudaMemcpyAsync` when CUDA runtime is present), then fails open to staged-copy destination writing (with explicit proof fallback events) while native decode-to-destination integration is being wired.
- `MX8_VIDEO_EXPERIMENTAL_DIRECT_DECODE_TO_DESTINATION=1` (requires direct-write + `nvdec|auto`) enables an internal decode-to-destination attempt in `to_torch()`: per clip, MX8 calls `torch.ops.mx8_video.decode_file_nvdec_into_u8(...)` to stream decoded bytes directly into the Torch-owned CUDA destination on the active stream; any unsupported case (remote URI, NVDEC failure, op unavailable) fails open to payload direct-write/staged-copy paths.
- `MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY` may point to a shared library that registers `torch.ops.mx8_video.direct_write_u8` through `torch.ops.load_library(...)`.

## Autotune Design (Dual Pressure)

Current video autotune pressure is CPU-centric (`rss_ratio` and inflight pressure).  
M8.1 adds GPU pressure:

- `gpu_ratio = used_vram / total_vram`
- Effective pressure:
  - `max(rss_ratio, inflight_ratio, gpu_ratio)`

Safety thresholds:

- `gpu_ratio >= 0.92`: downshift inflight/want.
- `gpu_ratio >= 0.97`: hard clamp to minimum safe mode.
- Recovery requires hysteresis (sustained lower pressure before scaling up).

If GPU telemetry is unavailable while NVDEC is active, behavior must be conservative:

- either forced fallback to CPU backend (recommended for GA), or
- strict low-concurrency clamp mode until telemetry is restored.

## Failure Semantics

Required fail-open behavior:

- Any NVDEC initialization/decode/convert error must not terminate the job.
- Emit fallback proof event with reason class.
- Continue decode with `ffi` or `cli`.

Examples of fallback reason classes:

- `decode_backend_unavailable`
- `gpu_telemetry_unavailable`
- `decode_failed`
- `unsupported_codec_profile`

## Observability

Expose backend and safety behavior in stats/proof logs:

- `video_decode_backend_selected`
- `video_decode_backend_fallback_total`
- `video_decode_backend_fallback_reason`
- `video_gpu_pressure`
- `video_gpu_pressure_unavailable_total`
- `video_runtime_autotune_gpu_clamps_total`
- existing `video_decode_failed_backend_unavailable_total`

These fields are required for rollout debugging and SLO validation.

## Build and Packaging

NVDEC is compile-time gated:

- `RUSTFLAGS='--cfg mx8_video_nvdec'`

CPU-only builds remain the default.  
GPU-enabled builds are optional and must preserve runtime fallback when deployed on unsupported nodes.

## Release Gates

M8.1 is complete only when all gates pass:

1. **Backend unavailable fallback**
   - Request NVDEC on a non-NVIDIA/disabled build.
   - Verify deterministic fallback and successful job completion.
   - Gate command: `./scripts/video_nvdec_fallback_gate.sh`
   - Compiled-path fallback command: `./scripts/video_nvdec_compiled_fallback_gate.sh`
2. **Decode parity**
   - Fixed fixtures compared between NVDEC and CPU path.
   - Verify shape/format contract and acceptable pixel-difference tolerance.
   - Existing parity command: `./scripts/video_stage3a_backend_gate.sh`
3. **VRAM pressure safety**
   - Stress test near VRAM limits.
   - Verify autotune downshift/hard clamp; no GPU OOM crash.
   - Gate command: `./scripts/video_nvdec_pressure_gate.sh`
4. **Throughput benchmark**
   - Measure vs `cli` baseline on representative 1080p/4K clips.
   - Record speedup and fallback rates.
   - Gate command: `./scripts/video_nvdec_throughput_gate.sh`

Unified runner:

- Full suite report:
  - `MX8_NVDEC_VALIDATION_PROFILE=full ./scripts/video_nvdec_validation_report.sh`
- GPU suite report (requires real NVDEC path):
  - `MX8_NVDEC_VALIDATION_PROFILE=gpu MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW=1 MX8_VIDEO_NVDEC_MIN_SPEEDUP=1.05 ./scripts/video_nvdec_validation_report.sh`
- Fallback-only report:
  - `MX8_NVDEC_VALIDATION_PROFILE=fallback ./scripts/video_nvdec_validation_report.sh`

The report script emits one summary and per-gate logs under:

- `MX8_NVDEC_REPORT_DIR` (default: `.artifacts/nvdec-validation/<timestamp>/`)
- files:
  - `summary.md`
  - `summary.txt`
  - `results.csv`
  - `<gate>.log`

## Promotion Criteria

Promote NVDEC to broader usage only when all of the following are true:

1. Fallback suite is green on non-GPU/unsupported nodes:
   - `MX8_NVDEC_VALIDATION_PROFILE=fallback ./scripts/video_nvdec_validation_report.sh`
2. GPU suite is green on a real NVIDIA runner with hardware-required throughput:
   - `MX8_NVDEC_VALIDATION_PROFILE=gpu MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW=1 MX8_VIDEO_NVDEC_MIN_SPEEDUP=1.05 ./scripts/video_nvdec_validation_report.sh`
3. Throughput gate confirms active NVDEC path and speedup threshold (`>= 1.05x` by default, or stricter release threshold if configured).
4. Pressure gate confirms clamp behavior with no crash/OOM regressions.
5. All runs produce `summary.md` + per-gate logs and zero failed gates.
6. CI manual workflow `nvdec-gpu-validation` stores artifacts for release review.
7. Until a GPU runner exists, `nvdec-gpu-validation` stays manual and does not block wheel publish; fallback/no-GPU profile remains required.

## Rollout

1. Ship behind `auto` with kill-switch to force CPU backend.
2. Start with canary users/workloads.
3. Monitor fallback rate, pressure excursions, clamp events, and throughput deltas.
4. Canary success targets:
   - no crash regressions
   - non-zero fallback handling on unsupported nodes
   - no repeated high-pressure clamp oscillation under steady load
5. Promote to broader usage only after stable gate and canary outcomes.
