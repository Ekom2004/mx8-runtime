# gRPC Contract

This document defines the wire protocol between the coordinator and agents.

Source of truth: `crates/mx8-proto/proto/mx8.proto`, `crates/mx8-coordinator/src/main.rs`, `crates/mx8d-agent/src/main.rs`.

API package: `mx8.v0`.


## Transport

The coordinator speaks gRPC over HTTP/2. It binds to `0.0.0.0:50051` by default, controlled by `MX8_COORD_BIND_ADDR`. The default maximum gRPC message size is 64MB on both sides, controlled by `MX8_GRPC_MAX_MESSAGE_BYTES`. Set this consistently on the coordinator and all agents. The manifest stream chunk size is 1MB.


## RPCs

The `Coordinator` service exposes eight RPCs.

`RegisterNode` registers or re-registers a node. It is idempotent for the same `job_id` and `node_id` combination and updates node caps on re-register. It requires non-empty IDs and a `caps` payload, and it respects the membership barrier and frozen membership rules. `resume_from` is an optional distributed resume token; when provided it is validated and applied before lease issuance.

`Heartbeat` reports node liveness and stats. It is idempotent with last-write-wins semantics. The node must be registered before sending heartbeats.

`RequestLease` acquires work leases for the node. It is not idempotent — each successful call can grant additional leases. The node must be registered and the job must have passed the membership barrier. `want` must be greater than zero.

`ReportProgress` advances the cursor for an active lease. Cursor movement is monotonic — the coordinator rejects any cursor that is lower than the previous value. The node must own the lease, and the cursor must be within the lease range.

`GetManifest` fetches manifest bytes by hash as a single unary response. `GetManifestStream` fetches the same data as a sequence of ordered 1MB chunks. Both are read-only and idempotent. If an agent receives `UNIMPLEMENTED` on `GetManifestStream`, it falls back to the unary `GetManifest` call automatically.

`GetJobSnapshot` returns a read-only cluster snapshot for operators and tooling. It includes cluster liveness state, lease counts, per-node heartbeat and stats, and coordinator event counters.

`GetResumeCheckpoint` returns an opaque distributed checkpoint token representing globally committed lease ranges for the job.


## Error codes

`INVALID_ARGUMENT` is returned for malformed requests — `want=0`, missing IDs, or invalid hash format.

`FAILED_PRECONDITION` is returned when a precondition is not met — node not registered, job not ready, membership frozen violations, or cursor regression on a progress report.

When coordinator leader fencing is enabled (`MX8_COORD_HA_ENABLE=1`), mutating RPCs (`RegisterNode`, `Heartbeat`, `RequestLease`, `ReportProgress`) return `FAILED_PRECONDITION` on followers or stale leaders with a `not leader for mutating operation` message.

`NOT_FOUND` is returned for unknown lease IDs or missing manifest hashes.

`UNAVAILABLE` signals a transient server issue. Clients should retry with backoff.

`INTERNAL` signals an unexpected server error.


## Key behavioral details

Membership freezes once `registered_nodes >= world_size`. After that, new unknown node IDs are rejected. `RegisterNodeResponse.assigned_rank` comes from the membership ordering and is stable for the life of the job.

Lease ranges are half-open intervals `[start_id, end_id)`. When `RequestLeaseResponse.wait_ms` is zero, the client should retry immediately. A non-zero value is a backoff hint for when no leases are available.

The cursor advances only after delivery to the consumer — not after fetch or decode. Lease completion is triggered when `cursor >= end_id`. The coordinator removes the lease and marks that range done.

The manifest stream emits ordered chunks with a `schema_version` field. The client concatenates them to reconstruct the canonical manifest bytes. Schema mismatches or truncated streams are fail-closed.

Distributed resume token contract:

`GetResumeCheckpoint` returns an opaque token that encodes `manifest_hash`, `epoch`, and completed lease ranges. The wire format is coordinator-owned and may change between major versions.

`RegisterNode.resume_from` is only valid before the coordinator has issued any lease for that run. Late tokens are rejected with `FAILED_PRECONDITION`.

If multiple nodes send `resume_from`, all tokens must be byte-equivalent in content (same checkpoint fingerprint). Conflicting tokens are rejected with `FAILED_PRECONDITION`.

A token is rejected when `manifest_hash` or `epoch` does not match the active run, when the token is malformed, or when a token range does not map to pending work.

`GetJobSnapshot.counters` exposes resume observability: `resume_checkpoint_applied_total`, `resume_checkpoint_rejected_total`, and `resume_ranges_applied_total`.


## Retry guidance

`Heartbeat`, `GetManifest`, `GetManifestStream`, and `GetJobSnapshot` are safe to retry immediately or with backoff.

`ReportProgress` should be retried with monotonic cursor discipline — always send the latest known cursor, never a stale one.

`RequestLease` should be retried carefully. Each successful call can grant additional leases, so duplicate successful calls may result in more leases than intended.


## Versioning

The current wire namespace is `mx8.v0`. Protobuf field numbers are compatibility anchors and must not be reused. The full compatibility and deprecation policy is in `docs/compatibility_policy.md`.
