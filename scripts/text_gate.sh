#!/usr/bin/env bash
set -euo pipefail

# Text loader gate for mx8.text.
#
# Proves:
# - end-to-end Python install path via maturin.
# - tokenizer loaded from local tokenizer.json.
# - int64 token_ids + bool attention_mask + int64 sample_ids output contract.
# - decode_error_policy behavior (`skip` succeeds, `error` fails on invalid UTF-8).
#
# Usage:
#   ./scripts/text_gate.sh
#
# Prereqs:
# - python3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[mx8] python3 not found on PATH" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mx8-text-gate.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

STORE_ROOT="${TMP_ROOT}/store"
RAW_DIR="${TMP_ROOT}/raw"
OUT_DIR="${TMP_ROOT}/mx8"
VENV_DIR="${TMP_ROOT}/venv"
TOKENIZER_JSON="${TMP_ROOT}/tokenizer.json"
DEV_MANIFEST_PATH="${TMP_ROOT}/dev_manifest.tsv"

mkdir -p "${STORE_ROOT}" "${RAW_DIR}" "${OUT_DIR}"

echo "[mx8] venv + deps (maturin + torch + tokenizers)"
python3 -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install -U pip >/dev/null
"${PYTHON_BIN}" -m pip install -U maturin >/dev/null
"${PYTHON_BIN}" -m pip install -U numpy torch tokenizers >/dev/null

echo "[mx8] generating text corpus (+ one invalid UTF-8 sample)"
"${PYTHON_BIN}" - "${RAW_DIR}" <<'PY'
import os
import sys

raw_dir = sys.argv[1]

samples = {
    "000.txt": "hello world mx8 runtime",
    "001.txt": "mx8 makes data loading deterministic",
    "002.txt": "tokenization in rust keeps python thin",
}

for name, text in samples.items():
    with open(os.path.join(raw_dir, name), "w", encoding="utf-8") as f:
        f.write(text)

# Invalid UTF-8 bytes on purpose for decode_error_policy contract check.
with open(os.path.join(raw_dir, "003.bin"), "wb") as f:
    f.write(b"\xff\xfe\xfd")
PY

echo "[mx8] building local tokenizer.json"
"${PYTHON_BIN}" - "${TOKENIZER_JSON}" <<'PY'
import sys
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

out = sys.argv[1]
vocab = {
    "[UNK]": 0,
    "hello": 1,
    "world": 2,
    "mx8": 3,
    "runtime": 4,
    "makes": 5,
    "data": 6,
    "loading": 7,
    "deterministic": 8,
    "tokenization": 9,
    "in": 10,
    "rust": 11,
    "keeps": 12,
    "python": 13,
    "thin": 14,
}
tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
tok.save(out)
PY

echo "[mx8] maturin develop"
(
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  unset CONDA_PREFIX
  "${PYTHON_BIN}" -m maturin develop --pip-path "${VENV_DIR}/bin/pip" --manifest-path crates/mx8-py/Cargo.toml >/dev/null
)

echo "[mx8] python: mx8.pack_dir"
"${PYTHON_BIN}" - "${RAW_DIR}" "${OUT_DIR}" <<'PY' >/dev/null
import sys
import mx8

raw_dir = sys.argv[1]
out_dir = sys.argv[2]

res = mx8.pack_dir(
    raw_dir,
    out=out_dir,
    shard_mb=1,
    label_mode="none",
)
if int(res["samples"]) != 4:
    raise SystemExit(f"unexpected pack_dir result: {res}")
PY

echo "[mx8] deriving dev manifest for local gate"
tail -n +2 "${OUT_DIR}/_mx8/manifest.tsv" > "${DEV_MANIFEST_PATH}"

echo "[mx8] running text loader contract gate"
MX8_MANIFEST_STORE_ROOT="${STORE_ROOT}" \
MX8_DATASET_LINK="${OUT_DIR}" \
MX8_DEV_MANIFEST_PATH="${DEV_MANIFEST_PATH}" \
MX8_TEXT_TOKENIZER_JSON="${TOKENIZER_JSON}" \
"${PYTHON_BIN}" "${ROOT}/crates/mx8-py/python/m6_text_gate.py" >/dev/null

echo "[mx8] text_gate OK"
