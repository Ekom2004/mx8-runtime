#!/usr/bin/env bash
set -euo pipefail

# Azurite gate (deterministic "Azure Blob" without cloud creds).
#
# Proves:
# - mx8-runtime azure feature can upload/read bytes via az:// locations
# - end-to-end delivery and inflight-byte cap invariants hold
#
# Usage:
#   ./scripts/azure_gate.sh
#
# Prereqs:
# - docker (daemon running)
# - curl
# - lsof

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep gate builds isolated from workspace target/ to avoid stale-lock stalls.
TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/mx8-target-azure-gate}"
if [[ "${TARGET_DIR}" = /* ]]; then
  BUILD_TARGET_DIR="${TARGET_DIR}"
else
  BUILD_TARGET_DIR="${ROOT}/${TARGET_DIR}"
fi
BIN_PATH="${BUILD_TARGET_DIR}/debug/mx8-demo5-azurite"

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
  echo "[mx8] curl not found (needed for Azurite readiness check)" >&2
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "[mx8] lsof not found (needed for local port checks)" >&2
  exit 1
fi

IMAGE="${MX8_AZURITE_IMAGE:-mcr.microsoft.com/azure-storage/azurite:3.35.0}"
DEFAULT_PORT=10000
PORT="${MX8_AZURITE_BLOB_PORT:-${DEFAULT_PORT}}"
NAME="mx8-azurite-gate-$$"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

port_is_listening() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

if [[ -z "${MX8_AZURITE_BLOB_PORT:-}" ]]; then
  if port_is_listening "${PORT}"; then
    for candidate in $(seq 10001 10100); do
      if ! port_is_listening "${candidate}"; then
        PORT="${candidate}"
        break
      fi
    done
    if [[ "${PORT}" == "${DEFAULT_PORT}" ]]; then
      echo "[mx8] no free local port found in 10001-10100 for Azurite" >&2
      exit 1
    fi
    echo "[mx8] port ${DEFAULT_PORT} is busy; using ${PORT} instead"
  fi
elif port_is_listening "${PORT}"; then
  echo "[mx8] requested MX8_AZURITE_BLOB_PORT=${PORT} is already in use" >&2
  echo "[mx8] choose a different port, or unset MX8_AZURITE_BLOB_PORT to auto-select" >&2
  exit 1
fi

echo "[mx8] starting azurite container (${IMAGE})"
docker run -d --name "${NAME}" \
  -p "127.0.0.1:${PORT}:10000" \
  "${IMAGE}" \
  azurite-blob --blobHost 0.0.0.0 --blobPort 10000 >/dev/null

READY_URL="http://127.0.0.1:${PORT}/"
echo "[mx8] waiting for azurite ready (${READY_URL})"
for _ in $(seq 1 80); do
  if curl -sS --max-time 1 "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! curl -sS --max-time 1 "${READY_URL}" >/dev/null 2>&1; then
  echo "[mx8] azurite did not become ready" >&2
  docker logs "${NAME}" | tail -n 200 >&2 || true
  exit 1
fi

export AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-devstoreaccount1}"
export AZURE_STORAGE_ACCESS_KEY="${AZURE_STORAGE_ACCESS_KEY:-Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==}"
export MX8_AZURE_ENDPOINT_URL="${MX8_AZURE_ENDPOINT_URL:-http://127.0.0.1:${PORT}/${AZURE_STORAGE_ACCOUNT}}"
export MX8_AZURE_ANONYMOUS=0

echo "[mx8] building demo5-azurite (azure feature enabled)"
(
  cd "${ROOT}" && \
    CARGO_TARGET_DIR="${BUILD_TARGET_DIR}" \
    CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" \
    cargo build -p mx8-runtime --features azure --bin mx8-demo5-azurite >/dev/null
)

echo "[mx8] running demo5-azurite (azure feature enabled)"
(
  cd "${ROOT}" && \
    RUST_LOG="${RUST_LOG:-warn}" \
    "${BIN_PATH}"
)

echo "[mx8] azure_gate OK"
