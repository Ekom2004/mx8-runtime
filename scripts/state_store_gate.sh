#!/usr/bin/env bash
set -euo pipefail

# Durable shared coordinator state store gate.
# Verifies snapshot save/load roundtrip for coordinator durable state schema.

echo "[mx8] state_store_gate: durable snapshot roundtrip"
cargo test -p mx8-coordinator save_then_load_roundtrip

