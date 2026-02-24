# Operator TUI

The MX8 TUI is a read-only terminal interface that gives operators a live view of a running job. It shows coordinator and lease state, per-node runtime stats from heartbeats, and a browsable manifest explorer. Use it during development to verify job health and during incidents to diagnose stale nodes, lease stalls, and manifest failures.

Primary implementation: `crates/mx8-tui/src/main.rs`.


## Running the TUI

```bash
cargo run -p mx8-tui -- \
  --coord-url http://coordinator-host:50051 \
  --job-id your-job-id
```

The TUI connects to the coordinator, polls for state, and refreshes the display automatically.


## Panels

The coordinator and lease panel shows job readiness and drain state, registered node count, active lease count, available range count, coordinator event counters (register, heartbeat, lease, progress), distributed resume counters (`resume_ok`, `resume_reject`, `resume_ranges`), a derived cluster health signal (`healthy`, `degraded`, or `stalled`), stale heartbeat count, and cluster-wide memory envelope totals.

The runtime panel shows per-node heartbeat age and state, per-node `inflight_bytes` and `ram_high_water_bytes` with cap ratios, fetch, decode, and pack queue depths from the latest heartbeat, and autotune fields when available: `autotune_enabled`, `effective_want`, `effective_prefetch_batches`, `effective_max_queue_batches`, `autotune_pressure_milli`, `autotune_cooldown_ticks`, `batch_payload_p95_over_p50_milli`, and `batch_jitter_slo_breaches_total`. Fields show as zero when the heartbeat does not carry them.

The manifest explorer shows sample ID, byte length, and location rows from the manifest. You can search and filter by location or decode hint.


## Keybindings

`q` quits. `j` or the down arrow moves to the next row. `k` or the up arrow moves to the previous row. `PgDn` and `PgUp` page through results. `/` enters search mode — type a filter string, press `Enter` to apply, press `Esc` to cancel. `g` enters jump-to-sample-id mode — type a numeric ID, press `Enter` to jump, press `Esc` to cancel.


## CLI arguments

| Argument | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `--coord-url` | `MX8_COORD_URL` | `http://127.0.0.1:50051` | coordinator address |
| `--job-id` | `MX8_JOB_ID` | `local-job` | job to inspect |
| `--poll-ms` | `MX8_TUI_POLL_MS` | `1000` | polling interval |
| `--headless-polls` | `MX8_TUI_HEADLESS_POLLS` | `0` | poll count for headless mode |
| `--manifest-hash` | `MX8_TUI_MANIFEST_HASH` | unset | optional initial manifest hash |
| `--search` | `MX8_TUI_SEARCH` | `""` | initial search filter for manifest panel |
| `--rows-per-page` | `MX8_TUI_ROWS_PER_PAGE` | `12` | manifest rows per page |
| `--manifest-path` | `MX8_TUI_MANIFEST_PATH` | unset | local TSV fallback when coordinator manifest is unavailable |
| `--stale-heartbeat-ms` | `MX8_TUI_STALE_HEARTBEAT_MS` | `5000` | heartbeat age threshold to mark a node stale |
| `--lease-stall-ms` | `MX8_TUI_LEASE_STALL_MS` | `10000` | no-progress threshold with active leases before marking job stalled |


## Headless mode

Headless mode is useful in gate scripts and CI. It polls a fixed number of times and exits non-zero if any panel never becomes non-empty.

```bash
cargo run -p mx8-tui -- \
  --coord-url http://127.0.0.1:50051 \
  --job-id demo2 \
  --poll-ms 300 \
  --headless-polls 6 \
  --stale-heartbeat-ms 5000 \
  --lease-stall-ms 10000
```

Gate script: `./scripts/tui_gate.sh`. To include in the smoke suite: `MX8_SMOKE_TUI=1 ./scripts/smoke.sh`.


## Troubleshooting

If the lease panel stays empty, verify the coordinator is reachable at `--coord-url` and that `--job-id` matches the active job.

If the runtime panel stays empty, verify at least one agent is heartbeating. The panel is built entirely from node heartbeat stats.

If the manifest panel stays empty, verify the manifest exists in the coordinator's `manifest_store`. Use `--manifest-path` to point the TUI at a local TSV file as a temporary fallback.
