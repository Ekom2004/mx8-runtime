#!/usr/bin/env bash
set -euo pipefail

# MinIO S3-compat gate (deterministic "S3" without AWS).
#
# Usage:
#   ./scripts/minio_gate.sh
#
# Prereqs:
# - docker
#
# Notes:
# - Starts a local MinIO container, runs `mx8-demo4-minio` against it, then tears it down.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "[mx8] docker not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[mx8] docker daemon not running (cannot connect to Docker socket)" >&2
  case "$(uname -s)" in
    Darwin)
      echo "[mx8] start Docker Desktop, then rerun (tip: \`open -a Docker\`)" >&2
      ;;
    *)
      echo "[mx8] start Docker engine/service, then rerun (e.g., systemctl start docker)" >&2
      ;;
  esac
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[mx8] curl not found (needed for MinIO health check)" >&2
  exit 1
fi

IMAGE="${MX8_MINIO_IMAGE:-minio/minio:latest}"
PORT="${MX8_MINIO_PORT:-9000}"
CONSOLE_PORT="${MX8_MINIO_CONSOLE_PORT:-9001}"
NAME="mx8-minio-gate-$$"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

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

if ! curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "[mx8] minio did not become ready" >&2
  docker logs "${NAME}" | tail -n 200 >&2 || true
  exit 1
fi

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=1

export MX8_S3_ENDPOINT_URL="http://127.0.0.1:${PORT}"
export MX8_S3_FORCE_PATH_STYLE=1

echo "[mx8] running demo4-minio (s3 feature enabled)"
(cd "${ROOT}" && RUST_LOG="${RUST_LOG:-warn}" cargo run -p mx8-runtime --features s3 --bin mx8-demo4-minio)

echo "[mx8] minio_gate OK"
