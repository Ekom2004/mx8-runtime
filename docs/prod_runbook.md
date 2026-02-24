# Production Incident Runbook

This runbook maps common operator symptoms to checks, actions, and escalation paths. Keep it open during production jobs. Before any new deployment, run the readiness suite:

```bash
./scripts/prod_readiness.sh
```

Have these identifiers ready when investigating a live job: `MX8_COORD_URL`, `MX8_JOB_ID`, the node IDs or ranks involved, and the pinned `manifest_hash`.


## Stale heartbeat

One or more nodes show a stale heartbeat age in the TUI or operator logs. Progress slows while active leases remain open.

Start by checking the TUI:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --headless-polls 6 \
  --poll-ms 300
```

Look for nodes with `hb_age_ms` above the stale threshold and active leases with no progress counter movement.

If you find stale nodes, verify that the node process is alive and that it has network reachability to the coordinator. Restart only the stale node's agent. Then wait for the lease TTL to expire and the coordinator to requeue the remainder. You should see `ranges_requeued_total` increment and progress resume.

If stale heartbeats persist across two full lease TTL cycles, drain the job and restart the coordinator and agents for that job.


## Lease stall

Leases are active but the progress cursor is not advancing. The job appears to be running but no data is being delivered.

First confirm that workers are delivering, not just fetching and decoding. A lease can show activity at the fetch stage while being stalled at delivery. Force-restart the stuck worker process on the affected node and let the coordinator expire and reassign the lease. You should see `leases_expired_total` and `ranges_requeued_total` rise and the job return to a progressing or drained state.

If the cursor remains unchanged after the lease is requeued and reassigned, stop the job and relaunch from the pinned snapshot.


## Manifest fetch failure

The TUI or client logs show manifest fetch errors. Startup fails to resolve the manifest or stream chunks.

Check the TUI:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --headless-polls 6 \
  --poll-ms 300
```

Validate that the manifest exists in the configured `manifest_store` and that the coordinator can serve `GetManifestStream`. As a temporary mitigation, point the TUI at a local manifest file:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --manifest-path /path/to/manifest.tsv
```

If the manifest panel populates with the local file, the issue is with the coordinator's access to the manifest store, not the manifest itself. If you suspect manifest mismatch or corruption, restart the run from an explicit pinned link using `@sha256:<manifest_hash>`.


## RSS breach and memory pressure

The loader fails fast with an error like `process rss ... exceeds max_process_rss_bytes`. This is expected behavior — MX8 is protecting the process from being OOM-killed by the OS.

First find the cap source by checking the loader startup logs for `MX8_MAX_PROCESS_RSS_BYTES`, `max_ram_bytes`, or the derived default. Then reduce pressure: lower `max_inflight_bytes`, lower `prefetch_batches`, lower `max_queue_batches`, and lower `want` on distributed loaders. Rerun with the corrected values.

If you need to raise the cap to maintain throughput, raise it in bounded increments and rerun the readiness gates each time.


## Resume token rejection

`RegisterNode` fails with errors like `invalid resume_from token`, `resume_from manifest_hash mismatch`, `resume_from epoch mismatch`, `conflicting resume_from token`, or `cannot apply resume checkpoint after lease issuance has started`.

First verify coordinator counters in the TUI. A rising `resume_reject` confirms bad or late resume tokens. `resume_ok` should increase only when the checkpoint is accepted.

Then enforce one token source for the whole job. Every rank must load the exact same checkpoint artifact and pass the same `resume_from` bytes at loader construction time.

If the error says leases were already issued, stop all ranks, restart the coordinator for a clean run, and relaunch all ranks with `resume_from` set before any data requests.

If the error says manifest/epoch mismatch, ensure the restart uses the same pinned dataset (`@sha256:<manifest_hash>`) and epoch as the checkpoint-producing run.

If rejections continue, relaunch without `resume_from` and treat it as a fresh epoch run. Preserve the failed token and coordinator log for incident review.


## Coordinator failure

In default v1.8 deployments, the coordinator is run as a single process per job. If it dies, the control plane pauses until restart. Optional HA foundations (`MX8_COORD_HA_ENABLE=1`) can promote a follower, but require shared lease/state paths and are still an operator-managed rollout.

Restart the coordinator for the affected job. Restart affected agents if they do not reconnect on their own. Then verify recovery in the TUI:

```bash
cargo run -p mx8-tui -- \
  --coord-url "$MX8_COORD_URL" \
  --job-id "$MX8_JOB_ID" \
  --headless-polls 6 \
  --poll-ms 300
```

You should see nodes re-register, heartbeats resume, and progress counters start moving again. If the coordinator repeatedly fails or the job cannot resume progress, drain it and relaunch from a pinned `@sha256:<manifest_hash>`.

When lease logging is enabled (default for non-dev manifest hashes), restart recovery replays durable completion and durable progress cursors. This means partially completed ranges should resume near their previous cursor instead of restarting from the full block.

If `MX8_COORD_HA_ENABLE=1` is enabled, mutating RPCs are fenced on followers/stale leaders with `FAILED_PRECONDITION` and a `not leader for mutating operation` message. In that case, direct agents to the active lease holder or wait for leader lease transition before retrying.

If HA is enabled, keep `MX8_COORD_STATE_STORE_PATH` on a shared durable filesystem visible to all candidate coordinators. If this path is not shared, leader transition can fence stale writers but cannot continue from latest shared state.

The v1.9 HA plan — single-writer leader with fencing, durable lease state, and automatic failover for inference and ETL jobs — is documented in `docs/ha_contract.md`. Training remains non-elastic even after coordinator HA ships.
