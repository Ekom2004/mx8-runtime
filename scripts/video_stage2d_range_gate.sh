#!/usr/bin/env bash
set -euo pipefail

# v1.8 stage-2d planning gate:
# - deterministic range planner contract tests
# - sidecar schema parse/canonicalization tests
#
# Usage:
#   ./scripts/video_stage2d_range_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[mx8] video stage2d range gate: cargo test -p mx8-snapshot video_stage2d_"
cargo test -p mx8-snapshot video_stage2d_

echo "[mx8] video_stage2d_range_gate OK"
