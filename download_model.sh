#!/usr/bin/env bash
# Bursa — model + tokenizer download for the ADTC 2026 audit.
# Idempotent and credential-free (D13). By default this downloads only the primary model
# declared in metadata.json. Pass --baselines during C2 to fetch the 0.6B comparison model.
set -euo pipefail

MODEL_DIR="model"
mkdir -p "${MODEL_DIR}"

PRIMARY_MODEL_FILE="${BURSA_PRIMARY_MODEL_FILE:-qwen3-1.7b-q4_k_m.gguf}"
PRIMARY_MODEL_PATH="${MODEL_DIR}/${PRIMARY_MODEL_FILE}"
PRIMARY_MODEL_URL="${BURSA_PRIMARY_MODEL_URL:-https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf}"
# Empty hashes use trust-on-first-use and print the computed digest. Before Gate 1, set the
# BURSA_*_SHA256 values (or replace these defaults) with independently verified digests.
PRIMARY_MODEL_SHA256="${BURSA_PRIMARY_MODEL_SHA256:-}"

BASELINE_MODEL_FILE="${BURSA_BASELINE_MODEL_FILE:-qwen3-0.6b-q4_k_m.gguf}"
BASELINE_MODEL_PATH="${MODEL_DIR}/${BASELINE_MODEL_FILE}"
BASELINE_MODEL_URL="${BURSA_BASELINE_MODEL_URL:-https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf}"
BASELINE_MODEL_SHA256="${BURSA_BASELINE_MODEL_SHA256:-}"

TOKENIZER_FILE="tokenizer.json"
TOKENIZER_PATH="${MODEL_DIR}/${TOKENIZER_FILE}"
# Qwen3-1.7B tokenizer (the fine-tuned model keeps the same tokenizer).
TOKENIZER_URL="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json"
TOKENIZER_SHA256=""

WITH_BASELINES=false
case "${1:-}" in
  "")
    ;;
  --baselines)
    WITH_BASELINES=true
    ;;
  --help|-h)
    echo "Usage: bash download_model.sh [--baselines]"
    echo "  default      primary model + tokenizer (submission evaluator path)"
    echo "  --baselines  also download the Qwen3-0.6B C2 comparison model"
    exit 0
    ;;
  *)
    echo "ERROR: unknown argument: $1" >&2
    exit 2
    ;;
esac

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# fetch_verify <url> <dest> <expected_sha256> <label>
# Existing files are never skipped silently: they are verified when a hash is pinned, otherwise
# their current digest is printed so a hardware-session operator can compare and pin it.
fetch_verify() {
  local url="$1" dest="$2" expected="$3" label="$4"
  if [ -f "${dest}" ]; then
    local existing
    existing=$(sha256_file "${dest}")
    if [ -n "${expected}" ] && [ "${existing}" != "${expected}" ]; then
      echo "ERROR: existing ${label} checksum mismatch at ${dest}" >&2
      echo "       expected ${expected}, got ${existing}" >&2
      exit 1
    fi
    if [ -n "${expected}" ]; then
      echo "${label} already present; checksum verified (${existing})."
    else
      echo "${label} already present; unpinned SHA-256 = ${existing}"
      echo "WARN: verify this digest independently and pin it before Gate 1." >&2
    fi
    return 0
  fi
  echo "Downloading ${label} to ${dest} ..."
  curl --fail --location --retry 3 "${url}" -o "${dest}.partial"
  local computed
  computed=$(sha256_file "${dest}.partial")
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

if [ -z "${PRIMARY_MODEL_URL}" ]; then
  echo "ERROR: PRIMARY_MODEL_URL is empty." >&2
  exit 1
fi
fetch_verify "${PRIMARY_MODEL_URL}" "${PRIMARY_MODEL_PATH}" "${PRIMARY_MODEL_SHA256}" \
  "Qwen3-1.7B primary GGUF"

if [ "${WITH_BASELINES}" = true ]; then
  fetch_verify "${BASELINE_MODEL_URL}" "${BASELINE_MODEL_PATH}" "${BASELINE_MODEL_SHA256}" \
    "Qwen3-0.6B baseline GGUF"
fi

if [ -n "${TOKENIZER_URL}" ]; then
  fetch_verify "${TOKENIZER_URL}" "${TOKENIZER_PATH}" "${TOKENIZER_SHA256}" "tokenizer.json"
else
  echo "WARN: TOKENIZER_URL not set; the app falls back to the heuristic token counter." >&2
fi

echo "Done."
echo "Primary model: ${PRIMARY_MODEL_PATH}"
if [ "${WITH_BASELINES}" = true ]; then
  echo "C2 baseline:   ${BASELINE_MODEL_PATH}"
fi
