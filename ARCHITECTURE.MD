MX8 v1.8 Architecture (Distributed Data Runtime + Tiny Authority Layer)

What MX8 Is:
MX8 is a high-performance Rust in-process data runtime (exposed to Python) plus a small, per-job distributed authority layer that:

- Keeps the hot-path (fetch/decode/pack/deliver) **in-process and bounded**.
- Centralizes cluster-level decisions (ownership, pacing) so multi-node behaves like **one coordinated consumer**, not many independent loaders.

Primary v1.8 workloads: offline inference / ETL / preprocessing / large corpus processing. Training is supported as a consumer (stability/throughput), but v1.8 does not keep DDP alive after node failure.

## Goals
- Stop “each node pulls independently” by introducing a tiny cluster authority layer.
- Stable throughput (no synchronized bursts / oscillation).
- Bounded memory (hard caps + backpressure; no RAM blowups).
- Deterministic sharding for a **pinned snapshot** (manifest hash) when failure-free.
- Kill-a-node recovery for inference/ETL via leases + cursor + reassignment.

## Non-Goals (v1.8)
- Multi-job QoS / fair-sharing across jobs.
- Fleet scheduler / GPU placement.
- Coordinator HA (if coordinator dies, the job stops/pauses).
- Mid-run “live pickup” of newly added data (refresh is job-start only in v1.8).
- Exactly-once delivery guarantees (at-least-once at boundaries is acceptable for v1.8 inference/ETL).

## Core Invariants
1) **No overlap**: a sample ID belongs to at most one *live* lease at a time.
2) **Bounded memory**: every pipeline stage has hard caps; backpressure prevents runaway prefetch.
3) **Deterministic (failure-free)**: for a pinned snapshot `(manifest_hash)` and `(seed, epoch, world_size, frozen membership)`, sharding/assignment is deterministic.
4) **Recovery**: if a node dies, unfinished portion of its lease can be reassigned (inference/ETL).

## Components
### A) `mx8` (in-process Rust runtime; called from Python)
Responsibilities:
- Bounded pipeline stages: Fetch → Decode/Parse → Pack → Deliver.
- Local planner consumes `WorkRange`s but does not decide global ownership.
- Metrics: samples/sec, bytes/sec, stage p50/p95, queue depths, RAM high-water mark.

### B) `mx8d-agent` (per node daemon)
Responsibilities:
- Local authority for budgets and feeding work to runtimes on the machine.
- Enforces per-node limits (fetch concurrency, inflight bytes, memory caps).
- Talks to coordinator for leases/heartbeats/progress.
- Hands subranges to local processes/ranks (node is the scheduling unit in v1.8).

### C) `mx8-coordinator` (per job leader)
Responsibilities:
- Resolves dataset link to a **pinned snapshot** (`manifest_hash`).
- Freezes membership for the run (barrier to `world_size`).
- Assigns leases for `WorkRange`s, expires leases, requeues remainders.
- Applies simple cluster-wide caps via lease issuance.

## Data Model
### Manifest (pinned snapshot)
Everything is a global sample ID space. Files/objects are implementation detail.

- `sample_id: u64` in `[0..N)`
- record contains at least:
  - location (path/key)
  - byte offsets/lengths when applicable
  - decode hints (optional; may include lightweight semantics like ImageFolder label IDs)

Vision v1.8 convention (optional):
- `decode_hint="mx8:vision:imagefolder;label_id=<n>"` enables classification-style training loops to consume
  `label_id` without per-sample label strings.
- When packing many small objects into `_mx8/manifest.tsv`, also write `_mx8/labels.tsv` once per dataset
  (`label_id<TAB>label`, percent-encoded) so users can map IDs back to strings.

v1.8 manifest format: **canonical TSV** (`_mx8/manifest.tsv` for packed datasets), with a **canonical hashing rule** for `manifest_hash` (hash the canonical record stream + schema version; do not rely on raw file bytes).

### WorkRange (unit of assignment)
Contiguous physical ID interval in pinned sample-ID space:

```
WorkRange { start_id: u64, end_id: u64 /* half-open */, epoch: u32, seed: u64 }
```

Shuffle in v1.8 is block-based in physical ID space:
- Split `[0..N)` into blocks (e.g., 64k–1M samples).
- Shuffle block order deterministically by `(seed, epoch)`.
- Iterate sequentially within a block (or tiny local shuffle).

### Lease (temporary ownership)
```
Lease { lease_id, node_id, range: WorkRange, expires_at, cursor }
```

Cursor semantics (v1.8):
- Cursor advances only after **DELIVER** (batch handed to Python/consumer).

## Dataset Links, Snapshotting, and `manifest_store`
Product rule:
- MX8 always runs off a **pinned snapshot** (`manifest_hash`), even if the user only provides a prefix/path.

Link forms (v1.8):
- Plain: `s3://bucket/prefix/` or `/mnt/data/prefix/` → resolves to the current pinned snapshot for the intent.
- Pinned: `...@sha256:<manifest_hash>` → uses that snapshot; no listing/indexing.
- Refresh: `...@refresh` → forces a new snapshot at job start; updates the intent pointer.

Packing convention (v1.8, optional):
- If a prefix contains a precomputed canonical manifest at `s3://bucket/prefix/_mx8/manifest.tsv`,
  the snapshot resolver should use it instead of listing objects.

Refresh scope:
- v1.8 refresh is **job-start only** (frozen within a run).

`manifest_store`:
- Configured location for manifests, locks, and intent pointers.
- Supports both:
  - S3-backed store: `s3://bucket/prefix/...`
  - Filesystem-backed store: `/var/lib/mx8/manifests` (or any path)

Portability rule (multi-node):
- Do **not** assume filesystem paths are shared/mounted identically across nodes.
- Agents should not need direct access to `manifest_store`.
- Coordinator provides a **manifest proxy fallback**: if an agent cannot fetch `manifest_url` directly, it requests manifest bytes by `manifest_hash` from the coordinator and caches locally.
- Coordinator never proxies dataset *data* bytes (only manifests/control-plane).

## Control Loops
Coordinator loop (~1s):
- Expire leases for nodes missing heartbeats.
- Requeue unfinished remainder ranges `[cursor, end)`.
- Grant new leases (fairness: simple RR/FIFO; obey caps).

Agent loop (~200ms):
- Heartbeat to coordinator with node stats.
- If local runtime has capacity, request leases.
- Hand ranges to local runtimes; report progress cursor periodically.

Heartbeat observability contract (v1.8):
- Node stats include memory/backpressure (`inflight_bytes`, `ram_high_water_bytes`, queue depths).
- When available from the runtime surface, heartbeats also include autotune/jitter summaries
  (effective knobs, pressure/cooldown, jitter ratio, breach counters).

Runtime loop:
- Get next range from local agent.
- Fetch/decode/pack/deliver with bounded queues + backpressure.

## Default Lease + Membership Policies (v1.8)
- Membership: dynamic registration + barrier to `world_size=N`, then freeze for the run.
  - Rank assignment: stable rule (sort by `node_id`) unless explicit ranks are provided.
- Heartbeat: 1s; coordinator tick: 1s; lease TTL: ~10s.
- Range sizing: adaptive to target ~5 seconds of work.
- Lease sizing: **1 lease = 1 block**, nodes may request multiple concurrent leases.

## Protocol Sketch (minimal)
Agent → Coordinator:
- `RegisterNode(node_id, caps)`
- `Heartbeat(node_id, stats)`
- `RequestLease(node_id, want)`
- `ReportProgress(node_id, lease_id, cursor)`

Coordinator → Agent:
- `LeaseGranted(lease)`
- `NoLease(wait_ms)`
- `GetManifest(manifest_hash)` (manifest proxy fallback; optional)

Process → Agent (localhost / unix socket):
- `RegisterWorker(job_id, process_id, demand_hint)`
- `GetNextRange(job_id) -> WorkRange`
- `ReportLocalStats(job_id, stats)`

## Process Diagrams
These are ASCII diagrams (not Mermaid) so they render the same everywhere.

### Component View (control-plane vs data-plane)

```
 ┌──────────────────────────┐        local RPC         ┌──────────────────────────┐        RPC         ┌──────────────────────────┐
 │      User Process         │  (unix / localhost)      │           Node           │   (HTTP / gRPC)    │           Job            │
 │                          │◀────────────────────────▶│                          │◀──────────────────▶│                          │
 │  Python code              │                          │  mx8d-agent (daemon)     │                    │  mx8-coordinator         │
 │  + mx8 runtime (Rust)     │                          │  - local caps/budgets    │                    │  - leases/ownership      │
 │  - bounded pipeline       │                          │  - feeds WorkRanges      │                    │  - resolves snapshots    │
 └─────────────┬────────────┘                          └──────────────────────────┘                    └─────────────┬────────────┘
               │                                                                                                        │
               │ DATA bytes (direct)                                                                                   │ read/write
               ▼                                                                                                        ▼
 ┌──────────────────────────┐                                                                             ┌──────────────────────────┐
 │      Dataset Storage       │                                                                             │       manifest_store       │
 │  S3 / NVMe / filesystem    │                                                                             │  S3 prefix or FS path      │
 └──────────────────────────┘                                                                             └──────────────────────────┘

 Note: If an agent cannot fetch the manifest URL directly (e.g., FS path not shared),
       the coordinator can proxy *manifest bytes by manifest_hash* (never dataset data).
```

### Lease lifecycle (happy path + failure)

```
 Runtime (mx8)            Node Agent (mx8d)                              Coordinator
     |                          |                                            |
     |  GetNextRange()          |                                            |
     |------------------------->|                                            |
     |                          |  RequestLease(want=N)                      |
     |                          |------------------------------------------->|
     |                          |  LeaseGranted(WorkRange, lease_id, ttl)    |
     |                          |<-------------------------------------------|
     |  WorkRange(...)          |                                            |
     |<-------------------------|                                            |
     |  fetch/decode/pack/deliver (bounded queues + backpressure)            |
     |  cursor advances after DELIVER                                        |
     |                          |  ReportProgress(lease_id, cursor)          |
     |------------------------->|------------------------------------------->|
     |                          |  Heartbeat(stats) every ~1s                |
     |                          |------------------------------------------->|
     |                          |                                            |
     |                (if node dies / heartbeats stop)                       |
     |                          |                                            |
     |                          |                    expire lease after TTL  |
     |                          |                    requeue [cursor,end)   |
     |                          |                    reassign remainder      |
     |                          |<-------------------------------------------|
     |  WorkRange(remainder)    |                                            |
     |<-------------------------|                                            |
```

## Locked Decisions (Canonical)
Date: 2026-02-12

### Workload scope
- v1.8 primary: offline inference / ETL / preprocessing / corpus processing.
- Training supported as a consumer, but v1.8 is **non-elastic** (DDP node death kills the job).

### Runtime + authority split
- Hot path stays in-process in Rust (`mx8` runtime).
- Cluster ownership/sharding is decided by a per-job coordinator; local budgets enforced by per-node agent.

### Snapshot rule
- MX8 always runs off a **pinned snapshot** (manifest + `manifest_hash`), even if the user only provides a prefix/path.
- v1.8 is **frozen within a run** (no mid-run pickup of new data).
- Default: frozen unless explicitly refreshed.
- Link syntax supports:
  - `@sha256:<manifest_hash>` pinned snapshot
  - `@refresh` job-start refresh

### Manifest store
- `manifest_store` is implemented in Rust; Python is a thin binding layer.
- `manifest_store` supports both S3-backed (`s3://...`) and filesystem-backed (`/path`).
- Multi-node portability rule: do **not** assume filesystem paths are shared/mounted identically across nodes.
- Agents should not need direct access to `manifest_store`.
- Coordinator provides a **manifest proxy fallback** by `manifest_hash` when direct fetch is not possible.
- Coordinator never proxies dataset **data** bytes (only manifests/control-plane).

### Membership + determinism
- Membership model: dynamic registration + barrier to `world_size=N`, then freeze membership for the run.
- Rank assignment: stable rule (sort by stable `node_id`) unless explicit ranks are provided.
- Determinism: failure-free determinism is defined for a pinned snapshot and frozen membership.

### Manifest format
- v1.8 manifest format: canonical TSV (`_mx8/manifest.tsv` for packed datasets).
- `manifest_hash` must be computed via a canonical hashing rule over records + schema version (not raw file bytes).

### Auto-index scope
- “Paste link and go” is supported for built-in indexers.
- v1.8 auto-index includes:
  - one-file/object-per-sample
  - exactly one packed format (customer-driven; likely WebDataset tar)
- All other layouts use explicit manifests (“bring your own manifest”) until additional indexers are shipped.

### Video decode (CPU + GPU)
- v1.8 baseline: CPU decode is supported everywhere.
- v1.8 GPU decode: Linux + NVIDIA (NVDEC) first.
- GPU decode is best-effort and must fail open to CPU (any GPU-decode error falls back to CPU) with a consistent output contract.

### Leases and progress semantics
- Cursor advances only after **DELIVER** to Python/consumer (not after fetch/decode).
- Inference/ETL: at-least-once at boundaries acceptable; “no gaps” preferred.
- Training: best-effort no-duplicates within an epoch; crash cases are not a hard guarantee (DDP would terminate).

### Lease sizing and timings
- Heartbeat: ~1s; coordinator tick: ~1s; lease TTL: ~10s.
- No mid-flight splitting in v1.8.
- Range sizing targets ~5 seconds of work (adaptive based on observed throughput).
- Lease unit: **1 lease = 1 shuffle block**; nodes may request multiple concurrent leases.

### Concurrency caps
- Enforce per-node caps via agent (fetch streams, inflight bytes/memory).
- Process-level RSS watchdog is enabled by default in production-facing loader surfaces (`max_ram_bytes` / `MX8_MAX_PROCESS_RSS_BYTES` override derived defaults); MX8 fails fast with an explicit error before OS OOM kill.
- Cluster-wide caps are applied indirectly via coordinator lease issuance (one job at a time in v1.8; no multi-job QoS).

### Autotune rollout status (v1.8)
- Startup profile autotune is available on v1-style API calls (for example `profile`, `constraints`, and `runtime` args).
- Runtime autotune loops are controlled through loader API args (`profile`, `autotune`, caps/constraints) on supported surfaces.
- Runtime autotune is intentionally rolled out incrementally across surfaces/modes to keep contract and stability risk bounded in v1.8.
