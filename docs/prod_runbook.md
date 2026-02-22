# MX8 Production Incident Runbook (v1.8)

This runbook maps common operator symptoms to exact checks, mitigations, and rollback actions.

## Preflight

Run from repo root:

```bash
./scripts/prod_readiness.sh
```

For live jobs, keep these identifiers ready:

- `MX8_COORD_URL`
- `MX8_JOB_ID`
- node IDs/ranks involved
- pinned `manifest_hash`

## 1) Stale heartbeat

Symptom:

- one or more nodes show stale heartbeat age in operator output/TUI
- coordinator counters progress slows while active leases remain

Checks:

```bash
cargo run -p mx8-tui -- --coord-url "$MX8_COORD_URL" --job-id "$MX8_JOB_ID" --headless-polls 6 --poll-ms 300
```

Look for:

- stale node heartbeat age (`hb_age_ms` above threshold)
- `active_leases > 0` with minimal progress counter movement

Action:

1. Verify node process health and network reachability to coordinator.
2. Restart only stale node workers/agent.
3. Wait for lease TTL expiry + requeue.

Expected outcome:

- stale node count drops
- coordinator `ranges_requeued_total` increments
- progress resumes

Rollback/escalation:

- if stale persists across `2 * lease_ttl`, drain job and restart coordinator + agents for that job.

## 2) Lease stall (active leases, no progress)

Symptom:

- leases remain active, but progress counter/cursor does not advance

Checks:

```bash
RUST_LOG=info \
MX8_TOTAL_SAMPLES=12000 \
MX8_DEV_BLOCK_SIZE=1000 \
MX8_SINK_SLEEP_MS=25 \
MX8_KILL_AFTER_MS=25 \
cargo run -p mx8-runtime --bin mx8-demo2
```

Action:

1. Confirm workers are delivering (not only fetching/decoding).
2. Force restart of stuck worker process on affected node.
3. Let coordinator expire lease and reassign remainder.

Expected outcome:

- `leases_expired_total` and `ranges_requeued_total` rise
- job returns to progressing/drained state

Rollback/escalation:

- if cursor remains unchanged after requeue, stop job and relaunch from pinned snapshot.

## 3) Manifest fetch failure

Symptom:

- TUI/clients show manifest fetch errors
- startup fails to resolve manifest or stream chunks

Checks:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --headless-polls 6 \
  --poll-ms 300
```

Action:

1. Validate manifest exists in configured `manifest_store`.
2. Verify coordinator can serve `GetManifestStream`.
3. Use local fallback manifest path only as temporary mitigation:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --manifest-path /path/to/manifest.tsv
```

Expected outcome:

- manifest panel populates
- workers resume normal range processing

Rollback/escalation:

- if manifest mismatch/corruption is suspected, restart run from explicit pinned `@sha256:<manifest_hash>`.

## 4) RSS breach / memory pressure

Symptom:

- fail-fast error: `process rss ... exceeds max_process_rss_bytes ...`

Checks:

```bash
cargo test -p mx8-runtime process_rss_cap_fails_fast_when_too_small
```

Action:

1. Confirm cap source (`max_ram_bytes`, `MX8_MAX_PROCESS_RSS_BYTES`, or derived default in loader startup logs).
2. Reduce pressure knobs first:
   - lower `max_inflight_bytes`
   - lower `prefetch_batches`
   - lower `max_queue_batches`
   - lower distributed `want`
3. Re-run with corrected caps.

Expected outcome:

- no RSS breach
- stable `inflight_bytes` and `ram_high_water_bytes`

Rollback/escalation:

- if cap must be raised to keep throughput, raise in bounded increments and re-run readiness gates.

## Coordinator HA and training elasticity

Current v1.8 contract:

- coordinator HA is not provided in default architecture
- training is non-elastic (node-loss-tolerant continuation is not guaranteed)

Coordinator failure action (v1.8):

1. Restart coordinator for the affected job.
2. Restart affected agents/workers if they do not reconnect cleanly.
3. Verify lease/progress recovery from fresh coordinator state.

Checks:

```bash
cargo run -p mx8-tui -- --coord-url "$MX8_COORD_URL" --job-id "$MX8_JOB_ID" --headless-polls 6 --poll-ms 300
```

Expected outcome:

- coordinator responds to snapshot RPCs
- nodes re-register and heartbeat resumes
- progress counters move again or job drains

Rollback/escalation:

- if coordinator repeatedly fails or job cannot resume progress, drain and relaunch from pinned `@sha256:<manifest_hash>`.

Planned v1.9 HA scope:

- single-writer leader with fencing
- durable lease/progress state
- automatic leader failover for inference/ETL jobs

Canonical HA plan and acceptance gates:

- `docs/ha_contract.md`

Training note:

- even after coordinator HA ships, training remains non-elastic unless explicitly upgraded in a later contract.
