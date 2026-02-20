# MX8 Mix Gate Runbook

This runbook covers the `mx8.mix` acceptance gate modes and smoke toggles.

## Quick Local Gate

Run the default local gate:

- `./scripts/mix_gate.sh`

Default checks:
- deterministic replay digest for 3 same-seed/epoch runs,
- weighted ratio tolerance check,
- shared inflight + process RSS cap checks,
- epoch+1 replay determinism check.

## Strict (CI-Oriented) Gate

Run strict mode:

- `MX8_MIX_GATE_STRICT=1 ./scripts/mix_gate.sh`

Strict defaults:
- `MX8_TOTAL_SAMPLES=4096` (per source manifest for ratio stability at strict step count)
- `MX8_MIX_GATE_STEPS=600`
- `MX8_MIX_GATE_RATIO_TOL=0.01`
- `MX8_MIX_GATE_EXPECT_EPOCH_DRIFT=0`
- `MX8_MIX_SNAPSHOT_PERIOD_TICKS=16`

Optional strict-plus check:
- set `MX8_MIX_GATE_EXPECT_EPOCH_DRIFT=1` to require epoch+1 digest differs from base epoch digest.

## Smoke Integration

Enable mix gate inside smoke:

- `MX8_SMOKE_MIX=1 ./scripts/smoke.sh`

Enable strict mix mode inside smoke:

- `MX8_SMOKE_MIX=1 MX8_SMOKE_MIX_STRICT=1 ./scripts/smoke.sh`

## Useful Overrides

- `MX8_MIX_GATE_WEIGHTS=0.8,0.2`
- `MX8_MIX_GATE_STEPS=400`
- `MX8_MIX_GATE_RATIO_TOL=0.02`
- `MX8_MIX_GATE_MAX_INFLIGHT_BYTES=134217728`
- `MX8_MIX_GATE_MAX_PROCESS_RSS_BYTES=2147483648`
