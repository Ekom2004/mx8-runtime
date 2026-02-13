# Repository Guidelines

## Project Structure & Module Organization

- `ARCHITECTURE.MD`: canonical architecture + locked decisions (read first).
- `implementation.md`: milestone-based implementation plan.
- `main.rs`: placeholder (codebase is still being bootstrapped).
- `skills/`: local Codex skills used for workflow/persona support (not runtime code).

When code is added, keep layout aligned to `ARCHITECTURE.MD` (e.g., a Rust workspace with focused crates such as runtime, io, decode, agent, coordinator).

## Build, Test, and Development Commands

This repo is currently doc-first; code scaffolding will land as part of `implementation.md` (M0+).

Once the Rust workspace exists:
- `cargo build`: build all crates.
- `cargo test`: run unit/integration tests.
- `cargo fmt --all`: format Rust code (rustfmt).
- `cargo clippy --all-targets --all-features`: lint for common Rust issues.
- `./scripts/smoke.sh`: run format/lint/tests plus internal demo gates (Demo 2 + Demo 3).
- `./scripts/py_smoke.sh`: build/install the PyO3 veneer with `maturin` and run the minimal Python example (M5 gate).

## Troubleshooting

- If you see `error: crate reqwest required to be available in rlib format, but was not found in this form` during `cargo run` or `./scripts/smoke.sh`, the incremental build artifacts can be in a bad state. Fix with:
  - `cargo clean -p reqwest`
  - then rerun the command (e.g., `./scripts/smoke.sh`).

## Coding Style & Naming Conventions

- Rust: run `cargo fmt` before commits; prefer explicit types at boundaries and clear error types.
- Rust should be **idiomatic**: clear ownership and error handling, avoid `unwrap()`/`expect()` in production paths, keep `unsafe` banned by default, and prefer small, composable modules with explicit boundaries.
- Naming:
  - crates/modules: kebab-case for crates, snake_case for modules/functions, CamelCase for types.
  - config/env: `MX8_*` prefix (e.g., `MX8_COORD_BIND_ADDR`, `MX8_COORD_URL`, `MX8_MANIFEST_STORE`).
- Keep “locked decisions” consistent; if you change behavior, update `ARCHITECTURE.MD`.

## Definition of Done (every implementation)

Every code change should end with:
- `cargo fmt --all`
- `cargo clippy --all-targets --all-features`
- `cargo test --workspace`
- A demo/test “gate” for the milestone (update an existing demo like `mx8-demo1`/`mx8-demo2` or add a new one) that proves the new invariant/behavior, plus the exact command to run it (prefer offline/deterministic when possible). Prefer “examples as docs”: make the demo/example show the safe/idiomatic usage pattern (don’t rely on warnings people won’t read).
- A brief note on hot-path time/memory complexity (big-O + growth drivers).
- Any new/changed invariants reflected in `ARCHITECTURE.MD` (if applicable).

## System Design Checklist (every new surface area)

Before adding a new crate/module/service endpoint, lock the following (in code + docs/tests):
- **Boundary**: smallest possible API (types, traits, RPCs); clear owner of state; no “leaky” abstractions.
- **Contracts/Invariants**: required vs optional fields, monotonicity rules, idempotency expectations, and what “valid” means (enforced by `validate()` + tests).
- **Resource limits**: explicit backpressure and hard caps (RAM, inflight bytes, queue sizes); no unbounded buffering.
- **Failure model**: timeouts, retries, cancellation, and partial-progress semantics (what is safe to replay).
- **Observability**: proof logs + metrics for the contract (so we can demo + debug by replaying logs).

## Optional Local Enforcement (no CI)

To enforce checks on every push without CI:
- `git config core.hooksPath .githooks`
- `chmod +x .githooks/pre-push scripts/check.sh`
- Then `git push` will run `./scripts/check.sh` (skip with `--no-verify`).

## Testing Guidelines

Testing harness is not yet established. When adding tests:
- Prefer integration tests for end-to-end invariants (leases/no-overlap, snapshot pinning).
- Name tests by behavior (e.g., `lease_expiry_requeues_remainder`).
- Add minimal “demo-replay” tests that validate logs/metrics contracts used in launch demos.

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, imperative, and descriptive (e.g., “Add .gitignore”, “Add David cofounder skill”).
- PRs (even for docs) should include:
  - what changed + why (1–3 bullets),
  - any updated invariants/decisions,
  - how to verify (commands or manual steps).

## Security & Configuration Tips

- Do not commit secrets (AWS keys, tokens). Use env vars or workload identity.
- macOS: `.DS_Store` should remain ignored via `.gitignore`.
