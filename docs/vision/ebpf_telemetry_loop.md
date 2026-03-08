# RFC: eBPF Network Telemetry Signal for MX8 Autotune (v1)

Status: Draft  
Owner: MX8 Runtime  
Scope: Linux accelerator only, no user-facing API changes

## 1. Summary

Add an optional Linux eBPF telemetry path that produces a low-rate "network congestion pressure" signal for MX8 autotune.

The signal is used as an additional control input to the existing application-layer autotune loop. It does not replace the current RSS/inflight/wait controls and does not change Python API usage.

If eBPF is unavailable, MX8 runs exactly as it does today.

## 2. Goals

- Reduce throughput variance on network-bound workloads.
- Reduce avoidable hard-cut events caused by late congestion detection.
- Improve time spent near steady-state throughput under fluctuating network conditions.
- Keep current safety rails and fail-open behavior.

## 3. Non-Goals

- No AF_XDP transport rewrite in this RFC.
- No changes to user-facing loader arguments or environment requirements.
- No requirement for root-only operation in default deployments.
- No cross-platform kernel instrumentation commitment (Linux only for v1).

## 4. Why This Exists

Today, autotune reacts when application-level symptoms are already visible (wait ratio, inflight pressure, RSS pressure). For some network paths, congestion can be detected earlier at the kernel boundary. A lightweight eBPF signal can provide earlier warning and reduce overshoot.

This is an additive control signal, not a new control system.

## 5. Proposed Design (v1)

### 5.1 Architecture

- Kernel side: eBPF program records per-flow latency/jitter indicators into BPF maps.
- User side: a small Linux-only collector reads/aggregates map data at low frequency.
- Autotune side: existing loop consumes one normalized scalar:
  - `net_pressure_ratio` in `[0.0, 1.0]`.

Final pressure used by autotune:

`effective_pressure = max(memory_pressure, net_pressure_ratio)`

This preserves existing memory safety behavior while allowing early network-aware backoff.

### 5.2 Signal Contract

Collector exports:

- `net_pressure_ratio`: normalized pressure estimate in `[0, 1]`
- `net_signal_fresh`: boolean freshness guard
- `net_signal_age_ms`: age of last valid sample

Rules:

- If signal is stale, missing, or invalid, treat as unavailable and ignore it.
- Never allow network signal to bypass hard memory rails.
- Signal sampling is decoupled from packet rate and bounded in overhead.

### 5.3 Sampling Cadence

- eBPF event capture is kernel-native.
- User-space aggregation runs at fixed low cadence (for example 2-4 Hz).
- Autotune loop reads latest aggregated sample once per tick.

No 1 kHz polling from autotune.

### 5.4 Safety and Fallback

- Feature is disabled by default until validated.
- If eBPF attach/load fails, log once and continue with current autotune path.
- If permissions are missing (`CAP_BPF`/`CAP_NET_ADMIN` as required by host/kernel policy), continue without eBPF.
- If collector crashes or data becomes stale, continue without eBPF signal.

## 6. Integration with Existing Autotune

Autotune stays feedforward + PID + AIMD with current rails/clamps.

Changes:

1. Add optional `net_pressure_ratio` input to tick function.
2. Compute `effective_pressure = max(existing_pressure, net_pressure_ratio)`.
3. Use `effective_pressure` where soft-cut/backoff decisions are made.
4. Keep hard-cut thresholds and cooldown semantics unchanged.

No user-facing API or minimal API variable changes.

## 7. Observability

Add metrics (Linux only when enabled):

- `autotune_net_pressure_ratio`
- `autotune_net_signal_age_ms`
- `autotune_net_signal_stale_total`
- `autotune_net_assisted_backoff_total`
- `autotune_net_disabled_total` (attach/init/fallback count)

Add adjustment reason tags such as `soft_cut_net` when network pressure is the dominant driver.

## 8. Rollout Plan

Phase 1: internal feature flag, off by default
- Implement collector + integration + tests.
- Verify no regressions in existing autotune unit tests.

Phase 2: controlled benchmarking
- Compare baseline vs enabled on network-bound mixed-size datasets.
- Record throughput mean/stddev, hard-cut counts, and stall ratios.

Phase 3: optional production preview
- Enable for selected Linux environments.
- Monitor fallback rate, signal freshness, and control stability.

Phase 4: default-on decision
- Only if benchmarks show clear gain and no stability regressions.

## 9. Acceptance Criteria

- No API changes required by users.
- Baseline behavior unchanged when eBPF unavailable.
- On qualifying Linux hosts, reduced throughput variance and/or reduced hard-cut rate on representative network-bound runs.
- No increase in memory safety incidents.

## 10. Risks

- Signal quality risk: noisy telemetry can create control chatter.
- Portability risk: kernel/version/capability differences across Linux fleets.
- Operational risk: privileged setup complexity in some environments.

Mitigations:

- Freshness/staleness guards.
- Conservative blending (`max` with existing pressure) instead of replacement.
- Fast fallback to current behavior.

## 11. AF_XDP Note (Future Work, Separate RFC)

AF_XDP is a transport-path architecture decision and should be treated as a separate research RFC. It is not part of this v1 telemetry-signal integration.
