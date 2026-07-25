#!/usr/bin/env bash
# Bursa — model + tokenizer download for the ADTC 2026 audit.
# Idempotent and credential-free (D13). Downloads to model/ (git-ignored). The GGUF path
# MUST match metadata.json _runtime.model_path.
set -euo pipefail

MODEL_DIR="model"
mkdir -p "${MODEL_DIR}"

MODEL_FILE="bursa-recon-1.7b-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
# Phase 2 / C2 verification uses the ZERO-SHOT Qwen3-1.7B Q4_K_M GGUF (exists today).
# At Phase 3/4, repoint MODEL_URL to the fine-tuned Bursa-Recon GGUF (same output path).
# The 0.6B C2-pivot baseline lives at:
#   https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
MODEL_URL="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"
# Checksum: EMPTY = trust-on-first-use (see fetch_verify). Pin the real value after first run.
MODEL_SHA256=""

TOKENIZER_FILE="tokenizer.json"
TOKENIZER_PATH="${MODEL_DIR}/${TOKENIZER_FILE}"
# Qwen3-1.7B tokenizer (the fine-tuned model keeps the same tokenizer).
TOKENIZER_URL="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json"
TOKENIZER_SHA256=""

# fetch_verify <url> <dest> <expected_sha256> <label>
# Trust-on-first-use: with an EMPTY expected hash, download, print the computed digest, and
# use the file (verify it against the Hugging Face file page, then pin the value here and
# commit — subsequent runs verify and fail closed on mismatch). No placeholder hashes ever.
fetch_verify() {
  local url="$1" dest="$2" expected="$3" label="$4"
  if [ -f "${dest}" ]; then
    echo "${label} already present at ${dest}; skipping."
    return 0
  fi
  echo "Downloading ${label} to ${dest} ..."
  curl -fL "${url}" -o "${dest}.partial"
  local computed
  computed=$(shasum -a 256 "${dest}.partial" | awk '{print $1}')
  if [ -z "${expected}" ]; then
    echo "TOFU: computed ${label} SHA-256 = ${computed}"
    echo "      Verify against the Hugging Face file page, then pin it (set the *_SHA256"
    echo "      variable in this script) and commit the confirmed value."
    mv "${dest}.partial" "${dest}"
  elif [ "${computed}" = "${expected}" ]; then
    mv "${dest}.partial" "${dest}"
    echo "${label} checksum verified."
  else
    echo "ERROR: ${label} checksum mismatch (expected ${expected}, got ${computed})" >&2
    rm -f "${dest}.partial"
    exit 1
  fi
}

if [ -z "${MODEL_URL}" ]; then
  echo "ERROR: MODEL_URL is empty." >&2
  exit 1
fi
fetch_verify "${MODEL_URL}" "${MODEL_PATH}" "${MODEL_SHA256}" "GGUF"

if [ -n "${TOKENIZER_URL}" ]; then
  fetch_verify "${TOKENIZER_URL}" "${TOKENIZER_PATH}" "${TOKENIZER_SHA256}" "tokenizer.json"
else
  echo "WARN: TOKENIZER_URL not set; the app falls back to the heuristic token counter." >&2
fi

echo "Done."
