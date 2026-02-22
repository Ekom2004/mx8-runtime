# MX8 Coordinator HA Contract and Rollout Plan (v1.9 Target)

Status: planned, not shipped in v1.8.

This document defines what "Coordinator HA" means for MX8, what it does not mean, and what must be true before we mark it production-ready.

## Scope

In scope:
- per-job coordinator high availability for inference/ETL/preprocessing jobs
- automatic leader failover with lease correctness preserved
- no-overlap invariant preserved through failover

Out of scope:
- training elasticity (DDP rank/node loss tolerance mid-epoch)
- multi-job scheduling or fairness

## Current v1.8 behavior

- Single coordinator process per job.
- If coordinator dies, the job control plane stops/pauses until restart.
- This is explicitly a known gap in `ARCHITECTURE.MD` and `docs/prod_runbook.md`.

## v1.9 HA target guarantees

1. Single active writer per job:
- exactly one coordinator leader may mutate lease state at any time
- stale leaders are fenced and cannot mutate state after leadership loss

2. Durable control-plane state:
- lease grants, lease expiry, range requeue, membership, and progress cursors survive leader failure

3. Automatic failover:
- followers can promote without manual operator intervention
- agents reconnect and continue progress reporting/lease requests

4. Invariants preserved during failover:
- no overlap for live leases
- idempotent progress/report semantics under retries

5. Defined recovery objective:
- failover target (initial): control-plane recovery in <= 15s p95 in normal cluster conditions

## Design plan

### 1) State abstraction and durability
- Introduce a `CoordinatorStore` boundary for:
  - membership
  - lease index
  - progress cursor
  - counters and manifest pin metadata
- Back with a strongly consistent KV store (etcd recommended).

### 2) Leader election and fencing
- Use per-job leader election in the same consistent store.
- Attach `(term, leader_id)` to mutating operations.
- Reject writes from stale terms.

### 3) Idempotent write model
- Progress/report RPC paths become idempotent with monotonic cursor checks.
- Lease transitions (`granted -> active -> expired/requeued -> completed`) are persisted atomically.

### 4) Agent failover behavior
- Agent accepts coordinator endpoint set (not single URL only).
- Retry policy: bounded exponential backoff with jitter.
- On leader move, agent retries register/heartbeat/request/progress against new leader.

## Failure behavior contract

Leader crash/restart:
- expected: follower promoted, agents reconnect, job continues from durable state.

Leader isolated (partition):
- expected: isolated leader loses election/lease and is fenced.
- expected: only quorum side can continue mutating state.

Backing store quorum loss:
- expected: control plane enters safe degraded mode (no new lease grants).
- expected: explicit operator alert; no split-brain writes.

## Acceptance gates (must pass before "HA ready")

1. Leader kill gate:
- kill active leader during multi-node run with active leases
- verify failover within target window and job drains

2. No-overlap failover gate:
- assert no-overlap invariant across leader transition

3. Stale leader fencing gate:
- old leader must fail all mutating attempts after demotion

4. Retry/idempotency gate:
- replay duplicate progress/report requests across failover
- verify monotonic cursors and consistent completion

5. Soak gate with repeated failovers:
- repeated leader churn over long run without invariant violation

## Rollout phases

Phase 0 (docs/design lock):
- lock this contract in docs and architecture.

Phase 1 (internal alpha):
- enable HA behind feature flag for selected jobs.
- run kill-leader gates in CI/nightly.

Phase 2 (beta):
- broaden usage for inference/ETL workloads.
- monitor failover latency and fence violations.

Phase 3 (GA):
- make HA default for coordinator deployments.
- keep non-HA fallback documented for rollback.

## Operator notes until v1.9 ships

- Treat coordinator as SPOF in v1.8.
- On coordinator failure, restart coordinator and then agents for the affected job.
- Do not advertise "no-pause" availability SLA for coordinator failures until HA gates are green.
