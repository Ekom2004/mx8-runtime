# MX8 Operator TUI (read-only, local)

`mx8-tui` is a read-only terminal UI that combines:

- live coordinator/lease state
- live node runtime stats from coordinator heartbeats
- manifest browsing/search from `GetManifestStream`

## Run

```bash
cargo run -p mx8-tui -- \
  --coord-url http://127.0.0.1:50051 \
  --job-id demo2
```

## Panels

- **Coordinator / Lease Overview**
  - job readiness/drained state
  - registered nodes, active leases, available ranges
  - coordinator counters (register/heartbeat/lease/progress)
  - derived cluster health (`healthy|degraded|stalled`)
  - stale heartbeat count and lease-stall detection
  - cluster memory envelope totals
- **Runtime Panel**
  - per-node heartbeat age/state
  - per-node `inflight_bytes`, `ram_high_water_bytes` and cap ratios
  - fetch/decode/pack queue depths (from latest heartbeat)
  - heartbeat-carried autotune fields when available:
    - `autotune_enabled`, `effective_want`, `effective_prefetch_batches`, `effective_max_queue_batches`
    - `autotune_pressure_milli`, `autotune_cooldown_ticks`
    - `batch_payload_p95_over_p50_milli`, `batch_jitter_slo_breaches_total`
  - note: zeros mean those fields are unavailable for that node
- **Manifest Explorer**
  - sample id, byte length, location rows
  - search filter over location/decode-hint

## Keybindings

- `q`: quit
- `j` / `Down`: next row
- `k` / `Up`: previous row
- `PgDn` / `PgUp`: page navigation
- `/`: enter search mode
  - type text, `Enter` to apply, `Esc` to cancel
- `g`: jump-to-sample-id mode
  - type numeric id, `Enter` to jump, `Esc` to cancel

## Headless mode (for gates)

```bash
cargo run -p mx8-tui -- \
  --coord-url http://127.0.0.1:50051 \
  --job-id demo2 \
  --poll-ms 300 \
  --headless-polls 6 \
  --stale-heartbeat-ms 5000 \
  --lease-stall-ms 10000
```

Headless mode exits non-zero if lease/runtime/manifest panels never become non-empty.

## Gate script

- `./scripts/tui_gate.sh`
- optional smoke toggle: `MX8_SMOKE_TUI=1 ./scripts/smoke.sh`

## Troubleshooting

- **`lease panel remained empty`**: verify coordinator is up and the `job_id` matches the active job.
- **`runtime panel remained empty`**: verify at least one agent is heartbeating (node stats are heartbeat-driven).
- **`manifest panel remained empty`**: verify manifest exists in coordinator `manifest_store`; use `--manifest-path` as local fallback.
