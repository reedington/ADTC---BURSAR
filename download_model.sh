#!/usr/bin/env bash
# Bursa — model + tokenizer download for the ADTC 2026 audit.
# Idempotent and credential-free (D13). Downloads to model/ (git-ignored). The GGUF path
# MUST match metadata.json _runtime.model_path.
set -euo pipefail

MODEL_DIR="model"
mkdir -p "${MODEL_DIR}"

MODEL_FILE="bursa-recon-1.7b-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
# TODO(Phase 3/4): public Hugging Face resolve URL of the released GGUF.
MODEL_URL="TODO_HUGGINGFACE_GGUF_URL"
# Checksum: leave EMPTY for trust-on-first-use (see fetch_verify). Never commit a placeholder.
MODEL_SHA256=""

TOKENIZER_FILE="tokenizer.json"
TOKENIZER_PATH="${MODEL_DIR}/${TOKENIZER_FILE}"
# TODO(Phase 3/4): tokenizer.json published ALONGSIDE the GGUF (the Qwen3 tokenizer).
TOKENIZER_URL="TODO_HUGGINGFACE_TOKENIZER_JSON_URL"
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

if [ "${MODEL_URL}" = "TODO_HUGGINGFACE_GGUF_URL" ]; then
  echo "ERROR: MODEL_URL is not set. The GGUF is published in Phase 3/4;" >&2
  echo "       fill MODEL_URL (and TOKENIZER_URL) with the public URLs before submission." >&2
  exit 1
fi
fetch_verify "${MODEL_URL}" "${MODEL_PATH}" "${MODEL_SHA256}" "GGUF"

if [ "${TOKENIZER_URL}" = "TODO_HUGGINGFACE_TOKENIZER_JSON_URL" ]; then
  echo "WARN: TOKENIZER_URL not set; the app falls back to the heuristic token counter." >&2
else
  fetch_verify "${TOKENIZER_URL}" "${TOKENIZER_PATH}" "${TOKENIZER_SHA256}" "tokenizer.json"
fi

echo "Done."
