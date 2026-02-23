# MX8 Documentation Backlog (Execution Plan)

Status: active backlog  
Scope: product documentation completeness for shipped MX8 capabilities and near-term roadmap items.

## Purpose

This backlog tracks documentation work required to make MX8 operationally usable and contract-clear for users and operators.

Goals:
- remove high-risk onboarding/operations ambiguity
- align docs with shipped behavior
- clearly separate shipped vs planned surfaces

## Priority Model

- `P0` = adoption blocker or production risk
- `P1` = major capability discoverability gap
- `P2` = governance/consistency improvement

---

## P0 - Critical Gaps (Do First)

- [x] `DOC-001` Coordinator + agent deployment guide
  - Missing: how to launch distributed jobs end-to-end.
  - Deliverable: `docs/deployment_guide.md`
  - Must include: ports, topology, coordinator placement, startup order, failure/restart flow, minimal production config.

- [x] `DOC-002` CLI tools reference
  - Missing: user-facing docs for `mx8-pack-s3`, `mx8-snapshot-resolve`, `mx8-seed-s3`.
  - Deliverable: `docs/cli_reference.md`
  - Must include: all args/env vars, examples, expected outputs, failure notes.

- [x] `DOC-003` gRPC wire contract
  - Missing: coordinator <-> agent protocol reference for operators/integrators.
  - Deliverable: `docs/grpc_contract.md`
  - Must include: RPCs, message fields, semantics, idempotency, error codes, versioning policy.

- [x] `DOC-004` MX8 env var reference
  - Missing: no canonical page for full `MX8_*` config surface.
  - Deliverable: `docs/env_reference.md`
  - Must include: variable, default, scope, valid values, stability (`stable|experimental|internal`).

- [x] `DOC-005` Security/auth surface
  - Missing: explicit security model and supported auth patterns.
  - Deliverable: `docs/security_model.md`
  - Must include: transport assumptions, authn/authz scope, secret handling, IAM patterns, hardening checklist.

- [x] `DOC-006` Version/compatibility guarantees
  - Missing: format/API compatibility guarantees and break policy.
  - Deliverable: `docs/compatibility_policy.md`
  - Must include: manifest format stability, stats schema policy, protocol compatibility, deprecation window.

---

## P1 - Major Capability Coverage Gaps

- [ ] `DOC-101` Zero-manifest deterministic resolution
  - Missing: user-facing explanation and usage contract.
  - Deliverable: section in `docs/python_api.md` + deep dive in `docs/s3_runtime_tuning.md`
  - Must include: enable/disable behavior, reservoir semantics, determinism expectations.

- [ ] `DOC-102` S3 surgical range-seek for compressed video (Stage2D)
  - Missing: planner/sidecar/range-merge behavior docs.
  - Deliverable: `docs/video_range_seek.md`
  - Must include: sidecar format, merge policy, fallback behavior, relevant stats fields.

- [ ] `DOC-103` Deterministic distributed clip traversal
  - Missing: how clips map to leases across nodes and epochs.
  - Deliverable: section in `docs/python_api.md` + `docs/video_ga_checklist.md`
  - Must include: determinism inputs (`seed`, `epoch`, membership), failure caveats.

- [ ] `DOC-104` Fail-closed decoding + corruption taxonomy
  - Missing: formal decode failure class contract.
  - Deliverable: `docs/decode_failure_taxonomy.md`
  - Must include: each failure class, runtime behavior, operator action.

- [ ] `DOC-105` Autotune coverage parity
  - Missing: mix runtime autotune + video runtime autotune details are incomplete.
  - Deliverable: expand `docs/python_api.md` and `docs/v1_autotune_api_contract.md`
  - Must include: inputs, runtime signals, rails, emitted stats fields.

- [ ] `DOC-106` TUI args and thresholds completeness
  - Missing: full CLI/env arg coverage and stale/stall threshold semantics.
  - Deliverable: update `docs/tui.md`
  - Must include: every CLI arg, env mapping, threshold interpretation.

- [ ] `DOC-107` Video loader env/stats completeness
  - Missing: stage2d env vars, range stats fields, ffprobe disable flags.
  - Deliverable: update `docs/python_api.md`
  - Must include: field-by-field list and operational tuning notes.

- [ ] `DOC-108` `loader.stats()` schema completeness
  - Missing: mix/video/data loader fields are only partially documented.
  - Deliverable: `docs/stats_schema.md` + links from `docs/python_api.md`
  - Must include: per-loader stats schemas, types, units, stability notes.

- [ ] `DOC-109` Memory contract internals
  - Missing: RSS sampling mechanism and node RAM detection chain details.
  - Deliverable: update `docs/memory_contract.md`
  - Must include: detection order (`cgroup -> /proc/meminfo -> sysctl`), caveats.

- [ ] `DOC-110` DistributedDataLoader arg completeness
  - Missing: `progress_interval_ms` and `grpc_max_message_bytes` contract clarity.
  - Deliverable: update `docs/python_api.md`
  - Must include: defaults, tuning guidance, failure implications.

---

## P2 - Consistency and Governance

- [ ] `DOC-201` Version consistency pass
  - Issue: version labeling mismatch across docs (e.g., v0 vs v1.8).
  - Deliverable: align `README.md`, `docs/python_api.md`, `ARCHITECTURE.MD`.

- [ ] `DOC-202` Mark `mx8._internal.*` as unstable/internal
  - Issue: internal Python module exists without explicit stability warning.
  - Deliverable: warning note in `docs/python_api.md`.

- [ ] `DOC-203` Monitoring bridge
  - Issue: no documented bridge from `loader.stats()` to Prometheus/Datadog.
  - Deliverable: `docs/monitoring_bridge.md`
  - Must include: sampling patterns, export adapters, metric naming conventions.

- [ ] `DOC-204` Public roadmap page
  - Issue: no single public roadmap for key planned items.
  - Deliverable: `docs/roadmap.md`
  - Must include: coordinator HA, mid-epoch resume, GPU decode timeline, status tags.

---

## Execution Order (Recommended)

1. `DOC-001` through `DOC-006` (P0)
2. `DOC-108` and `DOC-110` (stats + distributed arg clarity)
3. `DOC-101` through `DOC-107` and `DOC-109`
4. `DOC-201` through `DOC-204`

## Definition of Done (Per Doc Item)

For each completed item:
- document is present and linked from relevant docs page(s)
- examples are copy-pastable and validated against current APIs/binaries
- defaults/env vars/units are explicit
- shipped vs planned behavior is clearly labeled
- associated gate/validation command is included where applicable
