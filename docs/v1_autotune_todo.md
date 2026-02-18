# MX8 v1 Autotune TODO

Status: working checklist for the hybrid AIMD + PID controller rollout.

## Controller core

- [x] Define control law contract (`docs/v1_autotune_api_contract.md`)
- [x] Implement controller tick model (AIMD inner loop + PID-like memory pressure)
- [x] Add profile rails (`safe|balanced|throughput`)
- [x] Add cooldown + single-change-per-interval behavior

## Runtime wiring

- [x] Wire distributed runtime knobs to dynamic setters (`want`, `prefetch_batches`, `max_queue_batches`)
- [x] Add data-wait signal collection from iterator wait time
- [x] Add periodic autotune background loop (2s cadence)
- [x] Emit proof event for runtime adjustments (`autotune_runtime_adjustment`)
- [x] Emit startup proof event (`autotune_startup_caps_selected`)

## User surface

- [x] Add env-gated preview in v0 (`MX8_AUTOTUNE`, `MX8_AUTOTUNE_PROFILE`)
- [x] Expose effective autotune fields in distributed `loader.stats()`
- [ ] Add first-class v1 public API (`profile=...`, `autotune=True`, `constraints=...`)

## Validation gates

- [ ] Add deterministic controller simulation test (unit test for tick transitions)
- [ ] Add distributed gate showing reduced `data_wait_ratio` vs static baseline
- [ ] Add regression threshold for no cap breach under autotune
