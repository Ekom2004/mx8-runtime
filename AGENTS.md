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
- A brief note on hot-path time/memory complexity (big-O + growth drivers).
- Any new/changed invariants reflected in `ARCHITECTURE.MD` (if applicable).

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
