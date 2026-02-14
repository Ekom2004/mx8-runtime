#!/usr/bin/env bash
set -euo pipefail

# Build a release wheel locally (requires Rust toolchain).
#
# Usage:
#   ./scripts/build_wheel.sh
#
# Output:
#   dist/*.whl

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

echo "[mx8] install maturin"
python3 -m pip install -U maturin >/dev/null

mkdir -p dist

echo "[mx8] maturin build (release -> dist/)"
python3 -m maturin build --release --out dist --manifest-path crates/mx8-py/Cargo.toml >/dev/null

echo "[mx8] built:"
ls -1 dist/*.whl
