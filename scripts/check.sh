#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/check.sh            # online (default)
#   MX8_CHECK_OFFLINE=1 scripts/check.sh
#
# This is the repo-local "Definition of Done" runner.

OFFLINE="${MX8_CHECK_OFFLINE:-0}"

echo "[mx8] fmt"
cargo fmt --all --check

if [[ "${OFFLINE}" == "1" ]]; then
  echo "[mx8] clippy (offline)"
  cargo lint_offline
  echo "[mx8] test (offline)"
  cargo test --workspace --offline
else
  echo "[mx8] clippy"
  cargo lint
  echo "[mx8] test"
  cargo test --workspace
fi

