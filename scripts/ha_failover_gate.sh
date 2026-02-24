#!/usr/bin/env bash
set -euo pipefail

# Deterministic coordinator HA failover gates.
# Validates: leader promotion continuity, no-overlap across failover, duplicate progress replay.

echo "[mx8] ha_failover_gate: continuity + no-overlap + duplicate-progress replay"
cargo test -p mx8-coordinator --test ha_failover
