# eBPF Autotune Phase 1 Checklist

Status: In progress  
Scope: Internal scaffolding only, no user-facing API changes

## Completed (Step 1-2)

- Added `NetPressureSample` contract in `crates/mx8-py/src/net_pressure.rs`.
- Added `NetPressureSource` trait with:
  - default noop implementation
  - feature-gated Linux eBPF skeleton implementation
- Added source builder with fail-open fallback to noop:
  - env selector: `MX8_AUTOTUNE_NET_SOURCE`
  - accepted values: `off|none|ebpf`
  - unknown/unavailable sources log and degrade to noop
- Wired source construction into `DataLoader` autotune task spawn.
- Extended `autotune_loop` signature to accept and poll net source each tick.
- Kept runtime control behavior unchanged (no pressure integration yet).

## Completed (Step 3-4)

- Integrated signal into control loop:
  - computes `effective_pressure = max(memory_pressure, net_pressure_ratio)`
  - uses net pressure for soft-cut/increase gating only
  - keeps hard memory rails unchanged
- Added shared metrics + stats surfacing:
  - `autotune_net_pressure_ratio`
  - `autotune_net_signal_age_ms`
  - `autotune_net_signal_stale_total`
  - `autotune_net_assisted_backoff_total`
  - `autotune_net_disabled_total`

## Next (Step 5+)

1. Add more integration-style tests:
   - no-regression when source is noop
   - stale-signal counter behavior under enabled source
2. Add rollout notes for Linux capability/permission fallback.
3. Replace eBPF skeleton source with real collector implementation.
