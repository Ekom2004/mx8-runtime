#!/usr/bin/env bash
set -euo pipefail

# Coordinator crash/restart recovery gate.
# Verifies durable cursor replay resumes partially completed ranges after restart.

echo "[mx8] coordinator_restart_gate: durable cursor replay"
cargo test -p mx8-coordinator lease_log_restart_resumes_from_logged_cursor

