#!/usr/bin/env bash
# Bursa — model + tokenizer download for the ADTC 2026 audit.
# Idempotent and credential-free (D13). By default this downloads only the primary model
# declared in metadata.json. Pass --baselines during C2 to fetch the 0.6B comparison model.
set -euo pipefail

MODEL_DIR="model"
mkdir -p "${MODEL_DIR}"

PRIMARY_MODEL_FILE="qwen3-1.7b-q4_k_m.gguf"
PRIMARY_MODEL_PATH="${MODEL_DIR}/${PRIMARY_MODEL_FILE}"
PRIMARY_MODEL_URL="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/d7f544eead698dbd1f15126ef60b45a1e1933222/Qwen3-1.7B-Q4_K_M.gguf"
PRIMARY_MODEL_SHA256="b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"

BASELINE_MODEL_FILE="qwen3-0.6b-q4_k_m.gguf"
BASELINE_MODEL_PATH="${MODEL_DIR}/${BASELINE_MODEL_FILE}"
BASELINE_MODEL_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf"
BASELINE_MODEL_SHA256="ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"

TOKENIZER_FILE="tokenizer.json"
TOKENIZER_PATH="${MODEL_DIR}/${TOKENIZER_FILE}"
# Qwen3-1.7B tokenizer (the fine-tuned model keeps the same tokenizer).
TOKENIZER_URL="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json"
TOKENIZER_SHA256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

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
# Existing files are never skipped silently: every artifact is verified against its pinned hash.
fetch_verify() {
  local url="$1" dest="$2" expected="$3" label="$4"
  if [ -f "${dest}" ]; then
    local existing
    existing=$(sha256_file "${dest}")
    if [ "${existing}" != "${expected}" ]; then
      echo "ERROR: existing ${label} checksum mismatch at ${dest}" >&2
      echo "       expected ${expected}, got ${existing}" >&2
      exit 1
    fi
    echo "${label} already present; checksum verified (${existing})."
    return 0
  fi
  echo "Downloading ${label} to ${dest} ..."
  curl --fail --location --retry 3 "${url}" -o "${dest}.partial"
  local computed
  computed=$(sha256_file "${dest}.partial")
  if [ "${computed}" = "${expected}" ]; then
    mv "${dest}.partial" "${dest}"
    echo "${label} checksum verified."
  else
    echo "ERROR: ${label} checksum mismatch (expected ${expected}, got ${computed})" >&2
    rm -f "${dest}.partial"
    exit 1
  fi
}

fetch_verify "${PRIMARY_MODEL_URL}" "${PRIMARY_MODEL_PATH}" "${PRIMARY_MODEL_SHA256}" \
  "Qwen3-1.7B primary GGUF"

if [ "${WITH_BASELINES}" = true ]; then
  fetch_verify "${BASELINE_MODEL_URL}" "${BASELINE_MODEL_PATH}" "${BASELINE_MODEL_SHA256}" \
    "Qwen3-0.6B baseline GGUF"
fi

fetch_verify "${TOKENIZER_URL}" "${TOKENIZER_PATH}" "${TOKENIZER_SHA256}" "tokenizer.json"

echo "Done."
echo "Primary model: ${PRIMARY_MODEL_PATH}"
if [ "${WITH_BASELINES}" = true ]; then
  echo "C2 baseline:   ${BASELINE_MODEL_PATH}"
fi
