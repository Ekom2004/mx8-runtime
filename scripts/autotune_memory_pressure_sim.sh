#!/usr/bin/env bash
set -euo pipefail

# Deterministic autotune control-law simulation:
# drives synthetic memory-pressure signals and prints knob reactions.
#
# Usage:
#   ./scripts/autotune_memory_pressure_sim.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

echo "[mx8] running deterministic autotune memory-pressure simulation"
python3 "${ROOT}/crates/mx8-py/python/m5_autotune_memory_pressure_sim.py"
echo "[mx8] autotune_memory_pressure_sim OK"
