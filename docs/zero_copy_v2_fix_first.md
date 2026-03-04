# Zero-Copy Video v2 (Fix-First Plan)

Updated: 2026-03-04  
Status: Planned (not shipped)

## Purpose

Define the full implementation plan for true video zero-copy on NVIDIA GPUs after the
`from_blob + custom deleter` spike failure. This document is fix-first: we lock safety
contracts before writing production code.

Target data path:

`FFmpeg hwaccel decode -> NV12 CUDA surface -> NPP NV12->RGB24 -> torch-owned CUDA tensor -> model`

## Scope

In scope:

- Phase 1: lock correctness contracts (stream/lifetime/pixel/autotune).
- Phase 2: build the smallest safe end-to-end path.
- Phase 3: close known risk areas with explicit mechanisms.
- Phase 4: ship gates proving safety/correctness/performance.

Out of scope for this phase:

- Multi-vendor GPU decode.
- New distributed coordinator behavior.
- CPU contract removal (CPU output remains available/fallback).

## Phase 1: Contract Lock (Required Before Implementation)

### 1) Stream Ownership Contract

Rules:

- Each decode+convert operation is bound to one explicit CUDA stream `S`.
- NVDEC surface ops and NPP conversion enqueue work only on `S`.
- Returned tensor is not globally synchronized; callers rely on stream semantics.
- Any cross-stream consumption requires an event fence (`record` on producer stream, `wait` on consumer stream).
- Torch allocator stream tracking is applied when required (`record_stream` equivalent in native path).

Invariants:

- No kernel reads output tensor memory before producer stream completion.
- No allocator reuse of output memory while in-flight work still references it.

Failure behavior:

- On contract violation detection in debug/gate mode, fail closed with explicit error.

### 2) Tensor Ownership Contract

Rules:

- Output tensor must be Torch-owned CUDA storage (`torch::empty`/equivalent).
- Native path receives `data_ptr`, shape, and stride metadata from that tensor.
- Native path never owns storage lifetime; no external custom deleter.

Explicitly disallowed:

- `torch::from_blob` on raw external CUDA pointers for production path.

Invariants:

- Tensor lifetime is governed by PyTorch refcount/allocator semantics only.

### 3) Pixel/Memory Layout Contract

Rules:

- Input decode surfaces are NV12 on device.
- Conversion target is RGB24 (`uint8`) in tensor memory.
- Plane strides/pitches come from decoder frame metadata (`linesize`/step), never assumptions.
- Destination stride is explicit and validated against tensor shape/contiguity expectations.

Color handling:

- Respect frame colorspace/range metadata (BT.601/BT.709 + limited/full range).
- Deterministic fallback mapping when metadata is absent.

Invariants:

- Output shape/stride/color contract is deterministic and test-verifiable.

### 4) Autotune/Resource Contract

Rules:

- Pressure signal is `max(cpu_pressure, gpu_pressure)`.
- `gpu_pressure` is sourced from NVML in-process telemetry.
- Hard/soft clamp thresholds and hysteresis are explicit and logged.

Invariants:

- GPU pressure spikes clamp inflight limits before OOM.
- CPU-only nodes and fallback paths remain stable.

## Phase 2: Minimal Safe Data Path

Goal: ship the smallest internal path that proves correctness under locked contracts.

Steps:

1. Add an internal experimental zero-copy backend mode (not default).
2. Allocate output tensor on CUDA via Torch-owned allocation.
3. Decode NVDEC surface and convert NV12->RGB24 directly into tensor memory on stream `S`.
4. Return tensor to Python without host copy.
5. Keep existing CPU path as automatic fallback on backend-unavailable/unsupported cases.

Constraints:

- No public API break in this phase.
- No global sync in hot path.
- No reuse of `from_blob` external storage ownership.

## Phase 3: Fixes for Known Risk Areas

### A) Wrong-Stream Write/Read Races

Fixes:

- Single-stream producer discipline for decode+convert.
- Event fencing for any stream handoff.
- Allocator stream tracking for returned tensor lifetime safety.
- Multi-stream stress gate required before release.

### B) NV12 Stride/Pitch/Color Errors

Fixes:

- Use decoder-provided source step for Y and UV planes.
- Use explicit destination step from tensor layout.
- Use NPP conversion routines with explicit ROI and step parameters.
- Validate against synthetic color patterns and known references.

### C) VRAM Blindness in Autotune

Fixes:

- Integrate NVML telemetry in-process.
- Feed GPU pressure into autotune control loop.
- Keep hard/soft clamp + hysteresis counters in stats/proof logs.
- Add stress gate that forces pressure and verifies clamp behavior.

## Phase 4: Acceptance Gates (Must Pass)

### Gate 1: Stream/Lifetime Safety

Purpose:

- Prove no premature memory release/reuse hazards under multi-stream workloads.

Pass criteria:

- Zero early-use violations across stress iterations.
- No corruption or illegal access indicators.

### Gate 2: Pixel Correctness

Purpose:

- Prove NV12->RGB24 correctness with colorspace/range handling.

Pass criteria:

- Error bounds within threshold on synthetic color bars/reference clips.
- Correct handling across BT.601/BT.709 and limited/full range inputs.

### Gate 3: Stride/Pitch Robustness

Purpose:

- Prove non-trivial dimensions and alignment edge cases work.

Pass criteria:

- No artifacts/crashes for odd/even width/height combinations and varied steps.

### Gate 4: VRAM Pressure Stability

Purpose:

- Prove autotune clamps before OOM and recovers without oscillation.

Pass criteria:

- No GPU OOM under stress scenario.
- Clamp counters/logs align with pressure excursions.

### Gate 5: Throughput Benefit

Purpose:

- Prove zero-copy path beats baseline on representative clips.

Pass criteria:

- Meets or exceeds configured speedup threshold on true NVDEC hardware path.

## Rollout Plan After Gates

1. Canary behind explicit opt-in.
2. Observe fallback/clamp/OOM/throughput metrics for one week.
3. Promote only if safety gates remain green and canary is stable.

## Implementation Start Checklist

Implementation can start when all are true:

- Contracts in Phase 1 are accepted.
- Gate pass/fail criteria are accepted.
- Experimental mode/fallback policy is accepted.
- Required GPU CI/canary environment is ready.
