#!/usr/bin/env bash
# Run only on the physical i5-class checkpoint laptop. Produces C2 scorecards, stress gates,
# llama-bench evidence, and ten participant-mode profiler reports.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -r /proc/cpuinfo ]; then
  CPU_NAME="$(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo)"
else
  CPU_NAME="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
fi
if [ -z "${CPU_NAME}" ]; then
  CPU_NAME="$(uname -m)"
fi
if [[ "${CPU_NAME}" != *"Intel"* || "${CPU_NAME}" != *"i5"* ]]; then
  echo "ERROR: target-hardware evidence requires an Intel i5-class laptop; found:${CPU_NAME}" >&2
  exit 2
fi

for tool in llama-server llama-bench curl; do
  command -v "${tool}" >/dev/null || {
    echo "ERROR: ${tool} is not on PATH" >&2
    exit 2
  }
done
test -x .venv/bin/adtc-profiler || {
  echo "ERROR: install requirements-profiler.txt into .venv first" >&2
  exit 2
}

test -f data/manifest.json || {
  echo "ERROR: freeze the reviewed dataset before the i5 checkpoint" >&2
  exit 2
}
.venv/bin/python -m bursa_eval.baseline fetch --accept-third-party-terms
mkdir -p runs/i5

.venv/bin/python -m bursa_eval.baseline run \
  --config configs/baseline/models.json \
  --dataset-manifest data/manifest.json \
  --enterprise data/adtc/proxy/mmlu_enterprise.jsonl \
  --bare data/bare/prompts.jsonl \
  --out runs/i5/baselines

SERVER_PID=""
stop_server() {
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT INT TERM

run_model() {
  local label="$1"
  local model_path="$2"
  llama-server --model "${model_path}" --ctx-size 2048 --threads 4 --temp 0 \
    --cache-type-k q8_0 --cache-type-v q8_0 --offline --port 8080 \
    >"runs/i5/${label}-server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for _attempt in $(seq 1 100); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null; then
      ready=1
      break
    fi
    sleep 0.2
  done
  if [ "${ready}" -ne 1 ]; then
    echo "ERROR: llama-server did not become ready for ${label}" >&2
    exit 1
  fi

  .venv/bin/python -m bursa_eval.stress --tokenizer model/tokenizer.json \
    --n 1000 --output "runs/i5/${label}-stress.json"
  stop_server
}

run_model "qwen3-1.7b" "model/qwen3-1.7b-q4_k_m.gguf"
run_model "qwen3-0.6b" "model/qwen3-0.6b-q4_k_m.gguf"

for run_number in $(seq 1 9); do
  .venv/bin/adtc-profiler run --submission . --mode participant --skip-accuracy \
    --output "runs/i5/submission-${run_number}.json"
done
.venv/bin/adtc-profiler run --submission . --mode participant \
  --output "runs/i5/submission-10.json"
cp "runs/i5/submission-10.json" submission.json

LATEST_BASELINE="$(find runs/i5/baselines -mindepth 1 -maxdepth 1 -type d \
  -name '*-four-model-zero-shot' | sort | tail -1)"
test -n "${LATEST_BASELINE}" || {
  echo "ERROR: four-model baseline directory was not produced" >&2
  exit 1
}
.venv/bin/python -m bursa_eval.baseline review-packet \
  --run-root "${LATEST_BASELINE}" \
  --output "runs/i5/bare-zero-shot-review.jsonl"

echo "i5 checkpoint complete. Fill the blind review packet, score it, then run bursa-baseline compare."
