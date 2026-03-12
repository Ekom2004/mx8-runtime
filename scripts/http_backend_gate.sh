#!/usr/bin/env bash
set -euo pipefail

# HTTP backend production gate:
# - validates HTTP range path throughput behavior on representative workloads
# - enforces prefetch speedup floors to catch performance regressions
# - validates transient retry path still completes with bounded caps
#
# Usage:
#   ./scripts/http_backend_gate.sh
#
# Optional threshold overrides:
#   MX8_HTTP_GATE_MIN_SPEEDUP_SMALL=1.8
#   MX8_HTTP_GATE_MIN_SPEEDUP_RETRY=1.4
#   MX8_HTTP_GATE_MIN_SPEEDUP_MIXED=1.2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[mx8] cargo not found on PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

MIN_SPEEDUP_SMALL="${MX8_HTTP_GATE_MIN_SPEEDUP_SMALL:-1.8}"
MIN_SPEEDUP_RETRY="${MX8_HTTP_GATE_MIN_SPEEDUP_RETRY:-1.4}"
MIN_SPEEDUP_MIXED="${MX8_HTTP_GATE_MIN_SPEEDUP_MIXED:-1.2}"
RUST_LOG_LEVEL="${MX8_HTTP_GATE_RUST_LOG:-warn}"

run_case() {
  local name="$1"
  local total_samples="$2"
  local bytes_per_sample="$3"
  local batch_size_samples="$4"
  local max_queue_batches="$5"
  local max_inflight_bytes="$6"
  local prefetch_compare="$7"
  local http_latency_ms="$8"
  local http_bandwidth_bps="$9"
  local http_fail_every_n="${10}"
  local min_speedup="${11}"

  echo "[mx8] http gate case=${name} total=${total_samples} bytes_per_sample=${bytes_per_sample} prefetch_compare=${prefetch_compare} latency_ms=${http_latency_ms} bandwidth_bps=${http_bandwidth_bps} fail_every_n=${http_fail_every_n}"

  local output
  output="$(
    RUST_LOG="${RUST_LOG_LEVEL}" \
      cargo run -q -p mx8-runtime --bin mx8-demo3 -- \
        --total-samples "${total_samples}" \
        --bytes-per-sample "${bytes_per_sample}" \
        --batch-size-samples "${batch_size_samples}" \
        --max-queue-batches "${max_queue_batches}" \
        --max-inflight-bytes "${max_inflight_bytes}" \
        --prefetch-compare "${prefetch_compare}" \
        --http-latency-ms "${http_latency_ms}" \
        --http-bandwidth-bps "${http_bandwidth_bps}" \
        --http-fail-every-n "${http_fail_every_n}"
  )"

  echo "${output}"

  MX8_HTTP_GATE_OUTPUT="${output}" python3 - \
    "${name}" \
    "${total_samples}" \
    "${prefetch_compare}" \
    "${max_inflight_bytes}" \
    "${min_speedup}" <<'PY'
import os
import re
import sys

if len(sys.argv) != 6:
    raise SystemExit("internal arg mismatch")

name = sys.argv[1]
expected_samples = int(sys.argv[2])
prefetch_compare = int(sys.argv[3])
max_inflight_bytes = int(sys.argv[4])
min_speedup = float(sys.argv[5])

line_re = re.compile(
    r"\[demo3\] prefetch=(?P<prefetch>\d+) elapsed_ms=(?P<elapsed>\d+) "
    r"samples=(?P<samples>\d+) bytes=(?P<bytes>\d+) "
    r"inflight_high_water=(?P<high>\d+) samples_per_sec=(?P<sps>[0-9.]+)"
)

rows = []
for line in os.environ.get("MX8_HTTP_GATE_OUTPUT", "").splitlines():
    m = line_re.search(line.strip())
    if not m:
        continue
    rows.append(
        {
            "prefetch": int(m.group("prefetch")),
            "elapsed_ms": int(m.group("elapsed")),
            "samples": int(m.group("samples")),
            "bytes": int(m.group("bytes")),
            "inflight_high_water": int(m.group("high")),
            "sps": float(m.group("sps")),
        }
    )

if len(rows) != 2:
    raise SystemExit(
        f"[mx8] {name}: expected 2 demo3 result lines, got {len(rows)}"
    )

by_prefetch = {row["prefetch"]: row for row in rows}
if 1 not in by_prefetch:
    raise SystemExit(f"[mx8] {name}: missing prefetch=1 baseline")
if prefetch_compare not in by_prefetch:
    raise SystemExit(
        f"[mx8] {name}: missing prefetch={prefetch_compare} comparison"
    )

baseline = by_prefetch[1]
compare = by_prefetch[prefetch_compare]

if baseline["samples"] != expected_samples:
    raise SystemExit(
        f"[mx8] {name}: baseline samples mismatch "
        f"(got {baseline['samples']} expected {expected_samples})"
    )
if compare["samples"] != expected_samples:
    raise SystemExit(
        f"[mx8] {name}: compare samples mismatch "
        f"(got {compare['samples']} expected {expected_samples})"
    )

if baseline["inflight_high_water"] > max_inflight_bytes:
    raise SystemExit(
        f"[mx8] {name}: baseline inflight_high_water "
        f"{baseline['inflight_high_water']} exceeds cap {max_inflight_bytes}"
    )
if compare["inflight_high_water"] > max_inflight_bytes:
    raise SystemExit(
        f"[mx8] {name}: compare inflight_high_water "
        f"{compare['inflight_high_water']} exceeds cap {max_inflight_bytes}"
    )

if baseline["sps"] <= 0.0:
    raise SystemExit(f"[mx8] {name}: baseline samples_per_sec must be > 0")
if compare["sps"] <= 0.0:
    raise SystemExit(f"[mx8] {name}: compare samples_per_sec must be > 0")

speedup = compare["sps"] / baseline["sps"]
if speedup < min_speedup:
    raise SystemExit(
        f"[mx8] {name}: speedup regression (got {speedup:.3f}, "
        f"required >= {min_speedup:.3f})"
    )

print(
    f"[mx8] {name}: speedup={speedup:.3f} "
    f"baseline_sps={baseline['sps']:.3f} compare_sps={compare['sps']:.3f} "
    f"baseline_ms={baseline['elapsed_ms']} compare_ms={compare['elapsed_ms']}"
)
PY
}

run_case \
  "small_objects_prefetch" \
  256 \
  4096 \
  16 \
  64 \
  $((64 * 1024 * 1024)) \
  8 \
  20 \
  0 \
  0 \
  "${MIN_SPEEDUP_SMALL}"

run_case \
  "small_objects_retry" \
  256 \
  4096 \
  16 \
  64 \
  $((64 * 1024 * 1024)) \
  8 \
  20 \
  0 \
  7 \
  "${MIN_SPEEDUP_RETRY}"

run_case \
  "mixed_large_bandwidth_limited" \
  96 \
  65536 \
  8 \
  64 \
  $((128 * 1024 * 1024)) \
  4 \
  10 \
  $((40 * 1024 * 1024)) \
  0 \
  "${MIN_SPEEDUP_MIXED}"

echo "[mx8] http_backend_gate OK"
