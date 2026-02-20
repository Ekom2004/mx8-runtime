#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VERSION_FILE="crates/mx8-py/Cargo.toml"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/release_stage2c.sh prep
  ./scripts/release_stage2c.sh bump <semver>
  ./scripts/release_stage2c.sh finalize <semver> [--push]

Commands:
  prep
      Runs release gates required for Stage2B/2C closeout:
      - scripts/check.sh
      - MX8_SMOKE_VIDEO_STAGE2B_CLEAN_ENV=1 MX8_SMOKE_VIDEO_STAGE2C_PERF=1 scripts/smoke.sh

  bump <semver>
      Updates mx8 Python package version in crates/mx8-py/Cargo.toml.
      Example: ./scripts/release_stage2c.sh bump 1.0.3

  finalize <semver> [--push]
      Validates version, commits the bump, and creates tag v<semver>.
      If --push is set, pushes main and the tag.
EOF
}

current_version() {
  sed -n 's/^version = "\(.*\)"/\1/p' "${VERSION_FILE}" | head -n 1
}

validate_semver() {
  local value="$1"
  if [[ ! "${value}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "[mx8] invalid semver: ${value}" >&2
    exit 1
  fi
}

set_version() {
  local new_version="$1"
  local tmp
  tmp="$(mktemp)"
  awk -v v="${new_version}" '
    BEGIN { updated = 0 }
    /^version = "/ && updated == 0 {
      print "version = \"" v "\""
      updated = 1
      next
    }
    { print }
    END {
      if (updated == 0) {
        exit 1
      }
    }
  ' "${VERSION_FILE}" > "${tmp}" || {
    rm -f "${tmp}"
    echo "[mx8] failed to update version in ${VERSION_FILE}" >&2
    exit 1
  }
  mv "${tmp}" "${VERSION_FILE}"
}

cmd="${1:-}"
case "${cmd}" in
  prep)
    echo "[mx8] release prep: check + stage2b/2c smoke gates"
    ./scripts/check.sh
    MX8_SMOKE_VIDEO_STAGE2B_CLEAN_ENV=1 \
    MX8_SMOKE_VIDEO_STAGE2C_PERF=1 \
      ./scripts/smoke.sh
    echo "[mx8] release prep OK"
    ;;

  bump)
    if [[ $# -ne 2 ]]; then
      usage
      exit 1
    fi
    new_version="$2"
    validate_semver "${new_version}"
    old_version="$(current_version)"
    if [[ "${new_version}" == "${old_version}" ]]; then
      echo "[mx8] version already ${new_version}; no changes"
      exit 0
    fi
    set_version "${new_version}"
    echo "[mx8] bumped mx8-py version ${old_version} -> ${new_version}"
    echo "[mx8] next: ./scripts/release_stage2c.sh finalize ${new_version}"
    ;;

  finalize)
    if [[ $# -lt 2 || $# -gt 3 ]]; then
      usage
      exit 1
    fi
    target="$2"
    push_flag="${3:-}"
    validate_semver "${target}"
    if [[ "${push_flag}" != "" && "${push_flag}" != "--push" ]]; then
      usage
      exit 1
    fi
    v_now="$(current_version)"
    if [[ "${v_now}" != "${target}" ]]; then
      echo "[mx8] version mismatch: Cargo=${v_now}, expected=${target}" >&2
      echo "[mx8] run: ./scripts/release_stage2c.sh bump ${target}" >&2
      exit 1
    fi
    tag="v${target}"
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
      echo "[mx8] tag already exists: ${tag}" >&2
      exit 1
    fi

    git add "${VERSION_FILE}"
    if git diff --cached --quiet; then
      echo "[mx8] nothing staged for version commit; did you bump already?"
      exit 1
    fi
    git commit -m "Bump mx8-py to ${target}"
    git tag "${tag}"
    echo "[mx8] created commit + tag ${tag}"

    if [[ "${push_flag}" == "--push" ]]; then
      git push origin main
      git push origin "${tag}"
      echo "[mx8] pushed main and ${tag}"
    else
      echo "[mx8] next: git push origin main && git push origin ${tag}"
    fi
    ;;

  ""|-h|--help|help)
    usage
    ;;

  *)
    usage
    exit 1
    ;;
esac
