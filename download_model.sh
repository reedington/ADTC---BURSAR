#!/usr/bin/env bash
# Bursa — model download for the ADTC 2026 audit.
# Idempotent and credential-free (D13). Downloads the released GGUF weights to
# model/ (git-ignored). The output path MUST match metadata.json _runtime.model_path.
set -euo pipefail

MODEL_DIR="model"
MODEL_FILE="bursa-recon-1.7b-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

# TODO(Phase 3/4): set to the PUBLIC Hugging Face resolve URL of the released GGUF.
MODEL_URL="TODO_HUGGINGFACE_GGUF_URL"

mkdir -p "${MODEL_DIR}"

# Idempotent: if the weights are already present, do nothing.
if [ -f "${MODEL_PATH}" ]; then
  echo "Model already present at ${MODEL_PATH}; nothing to do."
  exit 0
fi

if [ "${MODEL_URL}" = "TODO_HUGGINGFACE_GGUF_URL" ]; then
  echo "ERROR: MODEL_URL is not set. The GGUF is published in Phase 3/4;" >&2
  echo "       fill MODEL_URL with the public Hugging Face URL before submission." >&2
  exit 1
fi

echo "Downloading model to ${MODEL_PATH} ..."
# Credential-free; -L follows redirects, -f fails on HTTP error. Atomic move on success.
curl -fL "${MODEL_URL}" -o "${MODEL_PATH}.partial"
mv "${MODEL_PATH}.partial" "${MODEL_PATH}"
echo "Done: ${MODEL_PATH}"
