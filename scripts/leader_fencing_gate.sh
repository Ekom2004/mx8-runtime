#!/usr/bin/env bash
set -euo pipefail

# Leader election + term fencing gate.
# Verifies one leader holds lease and stale leader is fenced after term bump.

echo "[mx8] leader_fencing_gate: election + fencing semantics"
cargo test -p mx8-coordinator leader_lease_election_and_fencing

