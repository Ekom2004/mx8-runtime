#!/usr/bin/env bash
set -euo pipefail

# MinIO gate for v1.8 Stage 1 S3 metadata extraction.
#
# Proves:
# - S3 prefix snapshot indexing emits Stage 1 video decode hints when enabled.
# - Manifest hash remains deterministic across unchanged runs.
#
# Usage:
#   ./scripts/minio_video_stage1_gate.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[mx8] docker not found" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[mx8] docker daemon not running" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[mx8] curl not found" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found" >&2
  exit 1
fi

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-video-stage1-gate-$$"

BUCKET="${MX8_MINIO_BUCKET:-mx8-video}"
PREFIX="${MX8_MINIO_PREFIX:-video/}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-minio-video-stage1-gate.XXXXXX")"
STORE_ROOT="${TMP_ROOT}/store"
SEED_FILE="${TMP_ROOT}/seed.bin"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${STORE_ROOT}"
python3 - "${SEED_FILE}" <<'PY'
import os, sys
path = sys.argv[1]
with open(path, "wb") as f:
    f.write(os.urandom(400_000))
PY

echo "[mx8] starting minio container (${IMAGE})"
docker run -d --name "${NAME}" \
  -p "127.0.0.1:${PORT}:9000" \
  -p "127.0.0.1:${CONSOLE_PORT}:9001" \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  "${IMAGE}" server /data --address ":9000" --console-address ":9001" >/dev/null

READY_URL="http://127.0.0.1:${PORT}/minio/health/ready"
echo "[mx8] waiting for minio ready (${READY_URL})"
for _ in $(seq 1 50); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "${READY_URL}" >/dev/null

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=1
export CARGO_PROFILE_DEV_DEBUG_ASSERTIONS="${CARGO_PROFILE_DEV_DEBUG_ASSERTIONS:-false}"

export MX8_S3_ENDPOINT_URL="http://127.0.0.1:${PORT}"
export MX8_S3_FORCE_PATH_STYLE=1
export MX8_S3_LABEL_MODE=none
export MX8_VIDEO_STAGE1_INDEX=1

echo "[mx8] seeding video-like keys under s3://${BUCKET}/${PREFIX}"
cargo build -p mx8-runtime --features s3 --bin mx8-seed-s3 >/dev/null
for key in \
  "${PREFIX}a.mp4" \
  "${PREFIX}nested/b.mp4" \
  "${PREFIX}clip.mov"; do
  MX8_MINIO_BUCKET="${BUCKET}" \
    MX8_MINIO_KEY="${key}" \
    MX8_SEED_FILE="${SEED_FILE}" \
    target/debug/mx8-seed-s3 >/dev/null
done

echo "[mx8] building mx8-snapshot-resolve (s3 feature enabled)"
cargo build -p mx8-snapshot --features s3 --bin mx8-snapshot-resolve >/dev/null

export MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}"
export MX8_DATASET_LINK="s3://${BUCKET}/${PREFIX}@refresh"
export MX8_NODE_ID="resolver"

echo "[mx8] resolving video snapshot from S3 prefix (run 1)"
h1="$(target/debug/mx8-snapshot-resolve | tee "${TMP_ROOT}/resolve1.log" | sed -n 's/^manifest_hash: //p')"
[[ -n "${h1}" ]] || { echo "[mx8] missing manifest hash (run 1)" >&2; exit 1; }

echo "[mx8] resolving video snapshot from S3 prefix (run 2)"
h2="$(target/debug/mx8-snapshot-resolve | tee "${TMP_ROOT}/resolve2.log" | sed -n 's/^manifest_hash: //p')"
[[ -n "${h2}" ]] || { echo "[mx8] missing manifest hash (run 2)" >&2; exit 1; }

if [[ "${h1}" != "${h2}" ]]; then
  echo "[mx8] expected deterministic manifest hash, got h1=${h1} h2=${h2}" >&2
  exit 1
fi

python3 - "${TMP_ROOT}/resolve1.log" "${TMP_ROOT}/resolve2.log" <<'PY'
import json
import re
import sys

logs = [open(p, "r", encoding="utf-8", errors="replace").read() for p in sys.argv[1:]]
event_re = re.compile(r'event\s*=\s*"snapshot_video_stage1_metadata_summary"')
hints_re = re.compile(r"video_hints_emitted_total\s*=\s*(\d+)")

for i, txt in enumerate(logs, start=1):
    if not event_re.search(txt):
        raise SystemExit(f"missing snapshot_video_stage1_metadata_summary in resolve{i}.log")
    matches = [int(m) for m in hints_re.findall(txt)]
    if not matches:
        raise SystemExit(f"missing video_hints_emitted_total field in resolve{i}.log")
    if max(matches) <= 0:
        raise SystemExit(
            f"video_hints_emitted_total did not increase in resolve{i}.log (values={matches})"
        )

print("[mx8] minio_video_stage1_gate proof event check OK")
PY

manifest_path="${STORE_ROOT}/by-hash/${h1}"
[[ -f "${manifest_path}" ]] || { echo "[mx8] missing manifest file ${manifest_path}" >&2; exit 1; }

python3 - "${manifest_path}" "${BUCKET}" "${PREFIX}" <<'PY'
import sys

path, bucket, prefix = sys.argv[1:4]
txt = open(path, "r", encoding="utf-8", errors="replace").read()

need = [
    f"s3://{bucket}/{prefix}a.mp4",
    f"s3://{bucket}/{prefix}nested/b.mp4",
    f"s3://{bucket}/{prefix}clip.mov",
]
for s in need:
    if s not in txt:
        raise SystemExit(f"manifest missing expected location: {s}")

lines = [l for l in txt.splitlines() if l.strip() and not l.startswith("schema_version=")]
by_loc = {}
for l in lines:
    cols = l.split("\t")
    if len(cols) < 5:
        continue
    by_loc[cols[1].strip()] = cols[4].strip()

for loc in need:
    hint = by_loc.get(loc, "")
    if "mx8:video;" not in hint:
        raise SystemExit(f"missing video decode_hint for {loc}: {hint!r}")
    if "frames=" not in hint:
        raise SystemExit(f"missing frames in decode_hint for {loc}: {hint!r}")
    if "stream_id=0" not in hint:
        raise SystemExit(f"missing stream_id in decode_hint for {loc}: {hint!r}")

print("[mx8] minio_video_stage1_gate OK")
PY

echo "[mx8] manifest_hash: ${h1}"
