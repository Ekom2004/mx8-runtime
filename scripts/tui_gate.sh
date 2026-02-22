#!/usr/bin/env bash
set -euo pipefail

# Read-only TUI gate:
# - starts demo2 to provide live coordinator/agent state
# - runs mx8-tui in headless mode and asserts non-empty panels

TMP_LOG="$(mktemp -t mx8-tui-gate.XXXXXX.log)"
DEMO_PID=""
BIN_DIR="${CARGO_TARGET_DIR:-target}/debug"
COORD_PORT="${MX8_TUI_GATE_COORD_PORT:-50061}"
COORD_BIND_ADDR="127.0.0.1:${COORD_PORT}"
COORD_URL="http://127.0.0.1:${COORD_PORT}"

cleanup() {
  if [[ -n "${DEMO_PID}" ]] && kill -0 "${DEMO_PID}" >/dev/null 2>&1; then
    kill "${DEMO_PID}" >/dev/null 2>&1 || true
    wait "${DEMO_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[mx8] tui_gate: starting demo2 background workload"

echo "[mx8] tui_gate: building binaries"
cargo build -p mx8-runtime --bin mx8-demo2 -p mx8-tui --bin mx8-tui >/dev/null

RUST_LOG=warn \
  MX8_COORD_BIND_ADDR="${COORD_BIND_ADDR}" \
  MX8_COORD_URL="${COORD_URL}" \
  MX8_TOTAL_SAMPLES=12000 \
  MX8_DEV_BLOCK_SIZE=1000 \
  MX8_SINK_SLEEP_MS=35 \
  MX8_KILL_AFTER_MS=40 \
  MX8_WAIT_REQUEUE_TIMEOUT_MS=15000 \
  MX8_WAIT_DRAIN_TIMEOUT_MS=30000 \
  "${BIN_DIR}/mx8-demo2" >"${TMP_LOG}" 2>&1 &
DEMO_PID=$!

# Wait until demo2 has actually granted at least one lease (avoids early empty snapshots).
ready=0
for _ in {1..40}; do
  if [[ -f "${TMP_LOG}" ]]; then
    log_text="$(<"${TMP_LOG}")"
    if [[ "${log_text}" == *"lease granted"* || "${log_text}" == *"waiting for range_requeued"* ]]; then
      ready=1
      break
    fi
  fi
  sleep 0.25
done

if [[ "${ready}" != "1" ]]; then
  echo "[mx8] tui_gate failed: demo2 did not reach lease-granted state"
  tail -n 120 "${TMP_LOG}" || true
  exit 1
fi

ok=0
for attempt in {1..8}; do
  echo "[mx8] tui_gate: headless probe attempt ${attempt}"
  if "${BIN_DIR}/mx8-tui" \
    --coord-url "${COORD_URL}" \
    --job-id demo2 \
    --poll-ms 300 \
    --headless-polls 6; then
    ok=1
    break
  fi
  sleep 1
done

if [[ "${ok}" != "1" ]]; then
  echo "[mx8] tui_gate failed; demo2 log tail:"
  tail -n 120 "${TMP_LOG}" || true
  exit 1
fi

wait "${DEMO_PID}"
echo "[mx8] tui_gate OK"
