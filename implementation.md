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

**Acceptance:**
- Stable RAM under a configured cap across extended run.
- Throughput is stable (no oscillation from prefetch bursts).
- Metrics show bounded queue depths and predictable p95 stage times.

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

