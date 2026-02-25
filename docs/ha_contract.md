# Coordinator HA Contract

Status: available in v1.8.6 as opt-in mode.

This document defines what coordinator HA means for MX8, what it explicitly does not include, and what must be true before it is considered default production posture.


## Current behavior in v1.8.6

Default v1.8 mode is still a single coordinator process per job. If that process dies, the control plane pauses until restart.

Opt-in HA mode is available now behind `MX8_COORD_HA_ENABLE=1` with shared lease/state paths across coordinator candidates.

Shipped foundations:

v1.8.3: coordinator restart recovery replays durable completion and durable cursor progress from the lease log.

v1.8.4: lease-file leader election + term fencing (`MX8_COORD_HA_ENABLE=1`) fences mutating RPCs on stale coordinators.

v1.8.5: durable coordinator state snapshots (`MX8_COORD_STATE_STORE_ENABLE=1`, auto-enabled with HA) persist membership, lease index, progress cursors, completed ranges, and counters; new leaders replay this state before accepting mutating work.

v1.8.6: deterministic kill-leader failover gates (`./scripts/ha_failover_gate.sh`) validate follower promotion continuity, no-overlap for newly issued leases after promotion, and duplicate progress replay acceptance across failover.

For incident procedure, see `docs/prod_runbook.md`.


## Guarantees in opt-in HA mode

HA covers per-job coordinator availability for inference, ETL, and preprocessing jobs. It does not cover training elasticity.

Exactly one coordinator leader may mutate lease state at any time. Stale leaders are fenced and cannot write after losing leadership.

Lease grants, expiry, range requeue, membership, and progress cursors survive leader failure and are recoverable by the promoted leader from shared durable state.

Followers can promote to leader without manual state repair. Agents reconnect and continue requesting leases and reporting progress.

The no-overlap invariant is preserved across leader transitions. Progress report handling remains monotonic so duplicate replays across failover do not corrupt cursor state.

Current failover SLO target is control-plane recovery within 15 seconds at p95 under normal cluster conditions.


## Design boundary

Current HA implementation uses shared durable files for leader lease and coordinator state (`MX8_COORD_HA_LEASE_PATH`, `MX8_COORD_STATE_STORE_PATH`) plus coordinator-side fencing and replay logic.

If shared HA paths are misconfigured or not truly shared, stale-writer fencing still works but continuity across leader switch is not guaranteed.

Future storage/election backends may be added, but this contract is defined by invariants and gates, not by a specific backend.


## Failure behavior

When the active leader crashes or is killed, a follower can promote, replay shared state, and continue serving mutating RPCs.

When a stale leader keeps receiving traffic after leadership loss, mutating RPCs are rejected with `FAILED_PRECONDITION` (`not leader for mutating operation`).

When shared HA paths are unavailable, the control plane must fail safe: no split-brain writes and no un-fenced mutating progress.


## Acceptance gates

Before HA is treated as default production posture, all of the following gates must pass repeatedly:

Kill the active leader during a multi-node run with active leases. Verify failover completes within target and job drains correctly.

Assert no-overlap across the full leader transition.

Verify the demoted stale leader fails mutating operations after leadership loss.

Replay duplicate progress/report requests across failover boundary. Verify monotonic cursors and consistent completion.

Run repeated leader churn over soak duration and verify no invariant violations accumulate.


## Training note

Training remains non-elastic in v1.8.x and this HA contract does not change that. DDP rank/node loss tolerance requires a separate training-elasticity contract.
