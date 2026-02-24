# Coordinator HA Contract

Status: planned for v1.9, not shipped in v1.8.

This document defines what coordinator HA means for MX8, what it explicitly does not include, and what must be true before it is marked production-ready.


## Current behavior in v1.8

The coordinator is a single process per job. If it dies, the job control plane pauses until the coordinator is restarted and agents reconnect. This is a known limitation. Until v1.9 ships, treat the coordinator as a single point of failure and run it on a stable, non-preemptible node.

v1.8.3 foundation: coordinator restart recovery replays durable completion and durable cursor progress from the lease log. This reduces replay after restart.

v1.8.4 foundation: optional lease-file leader election + term fencing (`MX8_COORD_HA_ENABLE=1`) fences mutating RPCs on stale coordinators. This enforces single-writer control-plane semantics but still does not provide shared-state continuity after leader switch.

For current incident procedure, see `docs/prod_runbook.md`.


## What HA means in v1.9

HA covers per-job coordinator availability for inference, ETL, and preprocessing jobs. It does not cover training elasticity — DDP rank or node loss tolerance mid-epoch is out of scope for this contract. Multi-job scheduling and fairness are also out of scope.

The v1.9 guarantees:

Exactly one coordinator leader may mutate lease state at any time. Stale leaders are fenced and cannot write after losing leadership.

Lease grants, expiry, range requeue, membership, and progress cursors survive leader failure and are recoverable by the new leader.

Followers can promote to leader without manual operator intervention. Agents reconnect to the new leader and continue requesting leases and reporting progress.

The no-overlap invariant is preserved across leader transitions. Progress report RPCs become idempotent with monotonic cursor enforcement so replays during failover do not corrupt state.

The initial failover target is control-plane recovery within 15 seconds at p95 under normal cluster conditions.


## Design

State durability is introduced by abstracting coordinator state behind a `CoordinatorStore` boundary covering membership, the lease index, progress cursors, and counters. This store is backed by a strongly consistent KV store — etcd is the recommended choice.

Leader election uses per-job elections in the same consistent store. Every mutating operation carries a `(term, leader_id)` tag. The coordinator rejects writes from stale terms, which enforces the single-writer guarantee and prevents split-brain.

Progress and report RPC paths are made idempotent with monotonic cursor checks. Lease state transitions — granted, active, expired, requeued, completed — are persisted atomically.

Agents are updated to accept a coordinator endpoint set rather than a single URL. On leader change, the agent retries register, heartbeat, request, and progress RPCs against the new leader using bounded exponential backoff with jitter.


## Failure behavior

When the active leader crashes or restarts, a follower is promoted, agents reconnect, and the job continues from the durable state. No manual action is required.

When the leader is partitioned from the cluster, it loses the election, gets fenced, and cannot mutate state. Only the quorum side continues processing.

When the backing store loses quorum, the control plane enters a safe degraded mode — no new leases are granted, and a clear operator alert is emitted. There are no split-brain writes.


## Acceptance gates

Before HA is marked production-ready, all of the following gates must pass:

Kill the active leader during a multi-node run with active leases. Verify failover completes within the target window and the job drains correctly.

Assert the no-overlap invariant holds across the full leader transition.

Verify that the demoted stale leader fails all mutating operations after losing leadership.

Replay duplicate progress and report requests across a failover boundary. Verify monotonic cursors and consistent completion.

Run repeated leader churn over a long soak run and verify no invariant violations accumulate.


## Rollout plan

Phase 0 locks this contract in docs and architecture (current state).

Phase 0.5 ships durable restart replay (`C` + `P` lease log lines) and a deterministic restart gate (`./scripts/coordinator_restart_gate.sh`).

Phase 0.6 ships lease-file leader election + term fencing and a deterministic gate (`./scripts/leader_fencing_gate.sh`).

Phase 1 enables HA behind a feature flag for selected internal jobs and runs the kill-leader gates in nightly CI.

Phase 2 broadens availability to inference and ETL workloads and monitors failover latency and fence violations.

Phase 3 makes HA the default for new coordinator deployments and keeps the non-HA fallback documented for rollback.


## Training note

Even after coordinator HA ships in v1.9, training remains non-elastic unless a separate training elasticity contract is added. The HA contract only covers the control plane recovery path, not the DDP rank membership changes that elastic training requires.
