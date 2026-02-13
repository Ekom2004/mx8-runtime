# MX8 v0 Implementation Plan

Date: 2026-02-12

This document is the step-by-step plan to bring `ARCHITECTURE.MD` to life. It is written to be executable by a small, high-performing systems team: tight milestones, explicit acceptance criteria, and clear risk control.

## Ground Rules (v0)
- Hot path is Rust. Python is a thin API surface (PyO3).
- v0 is **frozen within a run** (no mid-run dataset pickup).
- MX8 always runs off a **pinned snapshot** (`manifest_hash`) even when the user passes a prefix/path.
- Multi-node: coordinator is job-scoped single leader; no HA.
- Coordinator never proxies dataset **data** bytes. Manifest proxy is allowed.
- Video decode: CPU everywhere; GPU decode v0 targets **Linux + NVIDIA NVDEC** with fail-open fallback to CPU.

## Deliverables (v0)
- Demo 1: single node stability (bounded RAM + stable throughput) on a large dataset.
- Demo 2: 4-node S3 job with deterministic sharding + leases, kill-a-node recovery, and stable throughput.

## Execution Plan (Integration-First)

This section exists to prevent “milestone drift” where crates/features are built in isolation and only attempt to integrate late. The milestone list below (M0–M8) is correct, but execution should follow these integration checkpoints.

**Principle:** no workstream runs more than ~3–5 days without an end-to-end runnable slice that exercises the boundary (types/proto/`manifest_hash`/leases/cursor).

### Phase 1 — Freeze Contracts (finish M1)
**Deliverable:** `mx8-core` is the canonical shared model (types + logical manifest schema), and `mx8-proto` matches it with stable semantics.

**Gate:**
- Types compile; round-trip serialize where applicable.
- Proto encode/decode smoke tests pass.
- Cursor + half-open range semantics are documented and tested.

### Phase 2 — Observability Skeleton (start M2 early)
**Deliverable:** consistent structured logs and a minimal metrics surface across runtime/agent/coordinator.

**Gate:**
- Logs include `job_id`, `node_id`, `lease_id`, `manifest_hash`, `epoch` (where applicable).
- “Proof log” line formats exist (ownership + cursor progression), even before full functionality.

### Phase 3 — Vertical Slice A (single node bounded runtime)
**Deliverable:** single-node runtime consumes a local manifest and runs a bounded Fetch→Decode(stub)→Pack→Deliver pipeline with hard caps + backpressure.

**Gate:**
- Demo-1 skeleton runs on one machine and shows RAM bounded + stable throughput on a synthetic/known dataset.

### Phase 4 — Snapshot Resolver (M3) wired into the slice
**Deliverable:** link resolution (`plain`, `@refresh`, `@sha256:`) produces a pinned `manifest_hash` and supplies manifest bytes/URL to consumers.

**Gate:**
- “Paste link → pinned snapshot” works deterministically.
- Concurrent startup does not cause LIST storms (single-writer indexing/locking semantics).

### Phase 5 — Distributed Control Plane (M7) in two passes
**Pass 1 (minimum):** membership barrier + lease issuance + agent heartbeat/progress wired end-to-end against a local manifest.

**Pass 2 (recovery):** lease TTL expiry + requeue remainder `[cursor,end)` + reassignment; add no-overlap proof logs.

**Gate:**
- Multi-node skeleton runs; kill-a-node produces a dip then recovery for inference/ETL semantics.

### Phase 6 — Real IO + tuning (expand M4)
**Deliverable:** S3 fetch and bounded inflight bytes integrated into the same pipeline; retries/backoff are correct.

**Gate:**
- Stable throughput from S3 without bursty request patterns (visible in metrics/logs).

### Phase 7 — Python veneer (M5)
**Deliverable:** PyO3 wrapper is thin; Python never owns core logic.

**Gate:**
- Minimal Python example runs end-to-end and prints/resolves `manifest_hash`.

### Phase 8 — Video + Hardening + Launch Demos (M6 + M8)
**Deliverable:** CPU decode baseline; NVDEC on Linux/NVIDIA with fail-open; soak + fault injection + packaging.

**Gate:**
- Demo 1 + Demo 2 are repeatable on a clean environment with a short runbook.

## Milestone Plan

### M0 — Repo + Build Baseline (1–2 days)
**Goal:** A stable workspace for rapid iteration with reproducible builds.

**Work:**
- Define crate/workspace layout matching `ARCHITECTURE.MD` (even if some crates are stubs initially).
- Establish logging conventions and a single config system approach (env + config file for daemons; explicit args for libraries).
- Add minimal CI checks (build + unit tests) once applicable.

**Acceptance:**
- `cargo build` succeeds for workspace skeleton.
- `ARCHITECTURE.MD` and this plan are referenced from `README` (if/when added).

---

### M1 — Core Types + Protocol Boundaries (2–4 days)
**Goal:** Lock the “shape” of v0 without overbuilding.

**Work:**
- Define shared types:
  - `WorkRange`, `Lease`, node stats, progress reports.
  - Dataset link parsing: plain, `@sha256:<hash>`, `@refresh`.
- Define coordinator/agent RPC surface (request/response structs and error semantics).
- Define the manifest record schema (logical schema; Parquet physical spec later).

**Acceptance:**
- Types compile and round-trip serialize (where applicable).
- RPC boundary is stable enough that runtime/agent/coordinator can be implemented independently.

---

### M2 — Observability Foundations (2–4 days, parallelizable)
**Goal:** Make performance and correctness visible early.

**Work:**
- Metrics model:
  - Throughput: samples/sec, bytes/sec.
  - Stage latency histograms: fetch/decode/pack.
  - Queue depths and RAM high-water.
  - Lease/cluster metrics: active leases, expiries, requeues.
- Logging:
  - Structured logs with `job_id`, `node_id`, `lease_id`, `manifest_hash`, `epoch`.
  - “Proof logs” for Demo 2 (ownership + cursor progression).

**Acceptance:**
- A “metrics snapshot” can be printed or scraped (format is flexible in v0; pick one and standardize).

---

### M3 — Manifest Store + Snapshot Resolver (S3 + FS) (1–2 weeks)
**Goal:** Make “paste link → pinned snapshot” real, storm-safe, and portable.

#### M3 Checklist (build in this order; keep it demo-runnable)
1) **FS store skeleton (no S3 yet)**
   - Define `mx8-manifest-store` crate with:
     - `ManifestStore` trait (put/get by hash; get/set “intent → current snapshot” pointer).
     - FS backend rooted at `MX8_MANIFEST_STORE_ROOT` (default `/var/lib/mx8/manifests`).
     - Directory layout:
       - `by-hash/<manifest_hash>`: immutable manifest bytes (content addressed).
       - `intent/<intent_key>/current`: points to `<manifest_hash>` (atomic swap).
       - `locks/<intent_key>.lock`: single-writer lock file (atomic create).
   - **Success:** unit tests prove atomic pointer update and immutability-by-hash.

2) **Intent + locking semantics**
   - Canonical `intent_key` derivation from dataset link “intent” (backend + bucket/path + prefix + options).
   - Lock acquisition:
     - FS: atomic create (e.g., `create_new`) + write owner metadata (pid/node_id/unix_time_ms).
     - Stale lock expiry policy (configurable TTL) to avoid permanent wedge after crash.
   - **Success:** concurrency test spawns N tasks; exactly one becomes indexer; others wait/back off.

3) **Canonical hashing (`manifest_hash`)**
   - Implement canonical hash over a stable logical record stream + `MANIFEST_SCHEMA_VERSION`.
   - Hashing must be independent of Parquet file bytes (parquet comes later; hash the logical rows).
   - **Success:** deterministic hash for the same records across runs; tests include record-order stability rule (explicitly define whether order matters and enforce it).

4) **Snapshot resolution API**
   - Add `mx8-snapshot` crate (or module) with:
     - Parse dataset links (`plain`, `@refresh`, `@sha256:` already exist).
     - `resolve_snapshot(link) -> { manifest_hash, manifest_bytes_or_url }` for v0.
   - Semantics:
     - Plain: return current snapshot for intent; if missing, create (index) then set current.
     - `@refresh`: create new snapshot at job start, update current.
     - `@sha256:`: bypass intent/locks; validate presence in `by-hash/` (or return “not found”).
   - **Success:** “paste link → pinned hash” works offline with FS store and a dev manifest input.

5) **Dev indexing path (minimal)**
   - For M3, do not implement full S3 listing yet.
   - Support one dev input form to create a snapshot:
     - `MX8_DEV_MANIFEST_PATH=/path/to/manifest.jsonl` (or CSV) → load into `mx8-core::ManifestRecord`.
   - **Success:** coordinator can generate a pinned snapshot without touching S3.

6) **Coordinator integration**
   - `mx8-coordinator` calls resolver at startup and stores `manifest_hash`.
   - `RegisterNodeResponse.manifest_hash` returns the pinned hash.
   - Implement `GetManifest(manifest_hash)` (small manifests) and `GetManifestStream(manifest_hash)` (chunked) using FS store bytes (control-plane only).
   - **Success:** agent can fetch/cache manifests even when bytes exceed gRPC message limits; coordinator never proxies dataset bytes.

7) **Observability + proof logs**
   - Emit proof logs:
     - `snapshot_resolved(intent_key, manifest_hash, mode=plain|refresh|pinned)`
     - `snapshot_indexer_elected(intent_key)` and `snapshot_index_wait(wait_ms)`
   - Metrics snapshots include manifest counts, lock waits, and resolver latencies.
   - **Success:** demo replay shows “single writer indexing” without LIST storms.

**Work:**
1) `manifest_store` backends (Rust):
   - FS store: writes content-addressed manifests and intent pointers under a configured root path.
   - S3 store: same semantics under a configured S3 prefix.
2) Locking and single-writer indexing:
   - Implement intent key derivation for `(backend, bucket/path, prefix, options)`.
   - Implement lock acquisition (S3 conditional put; FS atomic create).
   - Implement stale-lock expiry rules (avoid wedging after crashes).
3) Snapshot resolution semantics:
   - Plain link resolves to current pinned snapshot for the intent (or creates one if missing).
   - `@refresh` forces a new snapshot (job-start only) and updates intent pointer.
   - `@sha256:<hash>` bypasses listing/indexing.
4) Canonical hashing:
   - Specify and implement `manifest_hash` as a canonical hash over records + schema version.
5) Manifest caching:
   - Local cache by hash (NVMe/FS) for faster startup.
6) Coordinator manifest proxy (control-plane only):
   - If agents cannot fetch `manifest_url` directly, they request manifest bytes by `manifest_hash` from coordinator.

**Acceptance:**
- Single-node and multi-node both deterministically resolve to the same `manifest_hash`.
- Concurrent multi-node startup does not cause LIST storms (only one indexer).
- `@refresh` creates a new manifest and updates the “current snapshot” pointer.

**Notes / risk control:**
- v0 auto-index supports:
  - one-file/object-per-sample
  - exactly one packed format (bless one; default candidate: uncompressed WebDataset tar)
- Everything else is “bring your own manifest”.

---

### M4 — Single-Node Runtime (bounded pipeline) (2–4 weeks)
**Goal:** Deliver Demo 1-level stability on one box before distributed complexity.

**Work:**
1) Pipeline stages with bounded queues:
   - Fetch (local + S3 streaming) with strict concurrency limits.
   - Decode/parse (text + image; video stubbed to CPU decode API at first).
   - Pack into batches.
   - Deliver to consumer (DLPack → torch).
2) Backpressure correctness:
   - Downstream full ⇒ upstream blocks (no runaway buffers).
   - Hard caps on inflight bytes and per-stage queue lengths.
3) Storage IO characteristics:
   - S3: range reads; readahead policy; retry/backoff.
   - Local: sequential reads; optional OS cache hints where safe.
4) Demo 3 — “S3-like latency simulator” (local):
   - Add a local HTTP range server + injected latency/bandwidth caps to simulate S3 behavior.
   - Use it to prove prefetch/readahead hides fetch latency while respecting inflight/RAM caps.
   - Keep it fully offline (no AWS credentials required).

**Acceptance:**
- Stable RAM under a configured cap across extended run.
- Throughput is stable (no oscillation from prefetch bursts).
- Metrics show bounded queue depths and predictable p95 stage times.
 - Demo 3 is repeatable and shows: with injected latency, prefetch improves throughput/idle time without violating caps.

---

### M5 — Python API (PyO3) (1–2 weeks)
**Goal:** “Just point at data” from Python with minimal friction.

**Work:**
- `mx8.DataLoader` / `mx8.VideoDataLoader` API surface:
  - Accept dataset link and options (batch size, shuffle seed/epoch, caps).
  - Expose/print resolved `manifest_hash` for reproducibility.
- DLPack bridge:
  - Return `torch.Tensor` efficiently.
- Operational ergonomics:
  - Clear error messages (bad link, missing creds, unsupported format).

**Acceptance:**
- A minimal Python example runs end-to-end on local + S3.
- “Switch loaders, keep training code” story holds.

---

### M6 — Video Decode v0 (CPU everywhere + NVDEC on Linux/NVIDIA) (3–6 weeks)
**Goal:** Video is the credibility wedge; ship it safely with strong fallbacks.

**Work (phased):**
1) CPU decode backend (baseline):
   - Implement clip extraction API (byte-range or time-based, consistent contract).
   - Implement robust error handling (corrupt frames, missing packets).
2) Output contract:
   - Normalize color space, shape, dtype, and stride expectations across backends.
3) GPU decode backend (Linux + NVIDIA NVDEC):
   - Capability probe.
   - Decode path with bounded queues; enforce max concurrent decode streams.
   - Fail open: any GPU decode error falls back to CPU without crashing the job.
4) Performance validation:
   - Ensure GPU decode helps end-to-end (watch PCIe transfers and pipeline balance).

**Acceptance:**
- CPU decode works on all supported environments.
- On Linux + NVIDIA, GPU decode can be enabled and automatically falls back to CPU on failure.
- Memory remains bounded under decode-heavy workloads.

---

### M7 — Distributed Control Plane (Coordinator + Agent) (3–6 weeks)
**Goal:** “Feels real” multi-node sharding + leases + recovery for inference/ETL.

**Work:**
1) Coordinator:
   - Membership barrier to `world_size=N`, then freeze membership for the run.
   - Rank assignment by stable rule (sort `node_id`) unless explicit ranks are provided.
   - Deterministic block schedule for a pinned snapshot + `(seed, epoch)`.
   - Lease issuance: **1 lease = 1 block**; nodes may request multiple leases.
   - Expiry + requeue remainder `[cursor,end)`; reassignment policy (FIFO/RR).
2) Node agent:
   - Enforce local caps (fetch streams, inflight bytes, memory budgets).
   - Serve WorkRanges to local runtimes.
   - Heartbeat + progress reporting to coordinator.
3) Manifest distribution:
   - Coordinator resolves snapshot once; agents/runtimes use `manifest_hash`.
   - Manifest proxy fallback works (no shared FS assumptions).

**Acceptance:**
- Kill-a-node mid-run causes lease expiry and reassignment; job continues (ETL/inference).
- No overlapping live leases (provable via logs/metrics).
- Cluster concurrency caps are honored (via lease issuance + per-node caps).

---

### M8 — Launch Demos + Hardening (2–4 weeks)
**Goal:** Product-quality demos and minimum-viable operational polish.

**Demo 1 (single node, large dataset):**
- Compare baseline DataLoader vs MX8.
- Show: RAM bounded, stable p95 fetch/decode, stable throughput/GPU util.

**Demo 2 (4 nodes from S3):**
- Show stable throughput; then kill one node.
- Show reassignment and recovery (throughput dip then recovery).
- Provide “no-overlap proof” logs and manifest hash pinning.

**Hardening checklist:**
- Soak tests (hours): memory, leaks, retries, partial failures.
- Fault injection: kill agent, kill runtime, drop coordinator.
- Backward compatibility: manifest schema versioning.
- Packaging:
  - `mx8-coordinator` binary
  - `mx8d-agent` binary
  - Python wheel(s)

**Acceptance:**
- Both demos are repeatable on a clean environment with a short runbook.

## Parallelization Strategy (to move fast)
- M3 (manifest) can proceed while M4 pipeline is built against a local manifest.
- M2 observability is parallel to everything; integrate early to avoid blind tuning later.
- M6 video work should start with CPU early (avoid waiting for GPU decode).
- M7 distributed can start once shared types + manifest resolution + a minimal runtime loop exist.

## Major Risks and How We Control Them
- **S3 listing/indexing costs:** centralized single-writer indexing; `@refresh` explicit; cache by intent+hash.
- **Video decode compatibility:** CPU baseline first; NVDEC only on Linux/NVIDIA in v0; strict fallback.
- **Determinism drift:** determinism is defined for a pinned snapshot + frozen membership; log `manifest_hash` + membership at job start.
- **Coordinator becoming bottleneck:** coordinator never proxies data; only control-plane + optional manifest bytes.
- **Scope creep on formats:** v0 auto-index supports exactly one packed format + one-file-per-sample; everything else is explicit manifest.

## Open Decisions (track explicitly)
- Exact packed format to bless for v0 auto-index (if not WebDataset tar).
- Precise Parquet canonical hashing spec and schema evolution rules.
- Membership timeout and operator experience if `<N` nodes register.
