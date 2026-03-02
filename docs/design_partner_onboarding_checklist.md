# Design Partner Onboarding Checklist

Updated: 2026-03-01

This checklist is the operator-facing readiness pass for onboarding a new design partner onto MX8.

## 1) Runtime + API readiness

- [x] `mx8-py` source split completed (`lib.rs` reduced to module wiring + `#[pymodule]` entrypoint; loader logic extracted into focused files).
- [x] Loader surfaces validated as present: `mx8.load`, `mx8.image`, `mx8.video`, `mx8.text`, `mx8.audio`, `mx8.mix`, `mx8.stats`.
- [x] Distributed lease/recovery code paths remain unchanged functionally after module extraction.

## 2) Reliability gate readiness

- [x] `cargo fmt --all`
- [x] `cargo clippy --all-targets --all-features`
- [x] `cargo test --workspace`
- [x] `./scripts/smoke.sh`
- [x] `./scripts/distributed_resume_gate.sh`
- [x] `./scripts/training_epoch_boundary_gate.sh`
- [x] `WORLD_SIZE=4 MX8_TORCH_DDP_NODUPES=1 ./scripts/torch_ddp_gate.sh`
- [ ] MinIO soak slice from production readiness:
  `WORLD_SIZE=8 TOTAL_SAMPLES=400000 KILL_COUNT=2 KILL_INTERVAL_MS=30000 WAIT_DRAIN_TIMEOUT_MS=600000 LEASE_TTL_MS=30000 ./scripts/soak_demo2_minio_scale.sh`
  (blocked locally on 2026-03-01: Docker daemon unavailable)

## 3) Operator docs readiness

- [x] Deployment baseline documented: `docs/deployment_guide.md`
- [x] Incident response documented: `docs/prod_runbook.md`
- [x] User day-to-day workflows documented: `docs/user_guide.md`
- [x] Python API contract documented: `docs/python_api.md`

## 4) Design partner handoff pack

- [x] Quickstart API examples in `README.md` and `docs/user_guide.md`
- [x] Control-plane env var/operator setup in `docs/deployment_guide.md`
- [x] Failure triage playbook in `docs/prod_runbook.md`
- [x] Troubleshooting reference in `docs/troubleshooting.md`

## 5) Release train readiness

- [x] Next patch version prepared in `Cargo.toml` files (`1.0.5`).
- [x] Release note draft added at `docs/releases/v1.0.5.md`.
- [ ] Publish/tag/push execution (owner action): build + publish wheel, then push commit/tag.

## Launch-day command list

```bash
cargo fmt --all
cargo clippy --all-targets --all-features
cargo test --workspace
./scripts/smoke.sh
```
