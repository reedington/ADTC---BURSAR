# Training, calibration, and four-model runbook

This runbook deliberately separates code readiness from evidence. A command succeeding on an
Apple developer machine does not create i5 benchmark evidence, and a generated Yoruba draft does
not become reviewed data until the named reviewer promotes it.

## 1. Review and freeze the dataset

The repository contains 140 deterministic draft cases in `data/gold_drafts/`. Every draft has a
`REVIEW_DRAFT` marker. A reviewer must rewrite the narration/rationale, verify the financial
answer, and then promote it:

```bash
bursa-data promote \
  --draft data/gold_drafts/<case>.yaml \
  --gold-dir data/gold \
  --reviewer <reviewer-pseudonym>
```

Promotion removes the draft file, sets `provenance: team_authored`, and records a checksum,
reviewer, and timestamp. Yoruba cases cannot pass the final validation without this metadata.

```bash
bursa-data validate --gold-dir data/gold --require-final
```

Export the pinned OpenAssistant revision to local JSONL, then build without network access:

```bash
bursa-data build \
  --config configs/data/release.json \
  --seed 3407 \
  --n-synth 4000 \
  --n-calibration 600 \
  --oasst data/sources/oasst1-messages.jsonl \
  --oasst-revision fdf72ae0827c1cda404aff25b6603abec9e3399b \
  --out data/build

bursa-data freeze \
  --config configs/data/release.json \
  --build-manifest data/build/build_manifest.json \
  --output data/manifest.json
```

Use `--general-fraction 0.40` only for the single pre-committed forgetting retry.

## 2. Run and review the four zero-shot baselines

The fetch command stops below 12 GiB free and requires explicit acknowledgement of the Gemma and
Llama terms:

```bash
bursa-baseline fetch --accept-third-party-terms

bursa-baseline run \
  --config configs/baseline/models.json \
  --dataset-manifest data/manifest.json \
  --enterprise data/adtc/proxy/mmlu_enterprise.jsonl \
  --bare data/bare/prompts.jsonl \
  --out runs/baselines
```

Run this on the physical i5. The runner processes one GGUF at a time and records four threads,
context 2,048, q8 KV cache, llama-bench results, RSS, temperature when exposed by Linux, model
hash, Git commit, and environment. C2 refuses Apple/ARM or other non-i5 measurements.

Create a blind packet after all four model directories exist:

```bash
bursa-baseline review-packet \
  --run-root <timestamped-run-root> \
  --output reviews/bare-zero-shot.jsonl
```

The reviewer fills each 0–2 rubric cell. Keep the generated `.key.json` hidden until scoring.

```bash
bursa-baseline score-review \
  --packet reviews/bare-zero-shot.jsonl \
  --key reviews/bare-zero-shot.key.json \
  --output reviews/bare-zero-shot-scores.json

bursa-baseline compare \
  --config configs/baseline/models.json \
  --run-root <timestamped-run-root> \
  --review-scores reviews/bare-zero-shot-scores.json \
  --output artifacts/c2_decision.json
```

## 3. Fine-tune on the rented GPU

Build `Dockerfile.training` on an Ubuntu NVIDIA host and mount enough persistent storage. The
command refuses a non-Linux host, missing CUDA, less than 22 GiB GPU memory, less than 80 GiB free
disk, a dirty repository, an unfrozen or mismatched dataset, or a missing C2 decision.

```bash
docker build -f Dockerfile.training -t bursa-training:2026.7.5 .
docker run --gpus all --rm \
  -v "$PWD/artifacts:/workspace/artifacts" \
  -v "$PWD/.cache:/workspace/.cache" \
  bursa-training:2026.7.5 run \
  --model all \
  --c2-decision artifacts/c2_decision.json
```

Each model produces the adapter, merged 16-bit checkpoint, F16 GGUF, Q4_K_M GGUF, metrics,
embedded template, dependency freeze, and hashes. Q5_K_M remains absent unless the measured Q4
loss exceeds three exact-allocation points.

Verify fixed validation decisions before scorecard comparison:

```bash
bursa-train smoke-compare \
  --config configs/training/qwen3-lora.toml \
  --merged <merged-16bit-directory> \
  --gguf <candidate-q4_k_m.gguf> \
  --validation data/build/val.jsonl \
  --dataset-manifest data/manifest.json \
  --output artifacts/merged-vs-q4-smoke.json
```

After the merged, Q4, suite, stress, and blind-rubric results exist, apply C3 only to the
C2-selected family:

```bash
bursa-train compare \
  --model qwen3-0.6b \
  --c2-decision artifacts/c2_decision.json \
  --zero-shot <zero-shot-scorecard.json> \
  --candidate <q4-scorecard.json> \
  --merged <merged-scorecard.json> \
  --stress <stress-1000.json> \
  --zero-bare-score <score> \
  --candidate-bare-score <score> \
  --attempt primary \
  --output artifacts/c3_decision.json
```

Use `qwen3-1.7b` instead only when that is the C2 selection. The command rejects the
evidence-only family. If `q5_required` is true, and only then:

```bash
bursa-train quantize-q5 \
  --f16-gguf <unquantized.gguf> \
  --c3-decision artifacts/c3_decision.json \
  --output <candidate-q5_k_m.gguf>
```

If the primary C3 decision alone sets `retry_60_40: true`, rebuild with
`--general-fraction 0.40` and run one selected-family attempt:

```bash
bursa-train run \
  --attempt 60-40-retry \
  --model <C2-selected-family> \
  --retry-decision artifacts/c3_decision.json \
  --c2-decision artifacts/c2_decision.json \
  --dataset-manifest <frozen-60-40-manifest.json>
```

A completed retry writes a receipt and a second completed retry is refused. Compare it with
`--attempt 60-40-retry`; if it still fails, the decision marks the retry exhausted and retains
the best zero-shot artifact.

The run manifest deliberately leaves merged/GGUF semantic smoke, Ollama, LM Studio, development
scorecards, and stress evidence marked `pending` until those external checks are actually
performed. Do not edit a pending status to “passed” without attaching the corresponding output.

## 4. Collect and fit calibration

Use only the C2-frozen production family. Collection runs the isolated 400/200 calibration cases
through the real model path without posting:

```bash
bursa-calibrate collect \
  --config configs/calibration/logistic-v1.json \
  --cases data/build/calibration_cases.jsonl \
  --model <final-q4.gguf> \
  --tokenizer model/tokenizer.json \
  --output artifacts/calibration-observations

bursa-calibrate fit \
  --config configs/calibration/logistic-v1.json \
  --fit-data artifacts/calibration-observations/fit.jsonl \
  --threshold-data artifacts/calibration-observations/threshold.jsonl \
  --validation-data <authored-validation-observations.jsonl> \
  --model <final-q4.gguf> \
  --source-manifest data/manifest.json \
  --output artifacts/calibration.json
```

The portable JSON artifact is auto-enabled only with at least 25 positive and 25 negative fitting
observations and a threshold at or above 0.90 with zero false automatic matches. Otherwise the
runtime remains review-only.

Set `BURSA_CALIBRATION_PATH` only after verification:

```bash
bursa-calibrate verify \
  --config configs/calibration/logistic-v1.json \
  --artifact artifacts/calibration.json \
  --data artifacts/calibration-observations/threshold.jsonl \
  --model <final-q4.gguf>

python -m bursa_eval.stress \
  --tokenizer model/tokenizer.json \
  --model <final-q4.gguf> \
  --calibration artifacts/calibration.json \
  --output artifacts/stress-1000.json
```

Any frozen-test false automatic post disables model automatic posting. Do not retune on test.

## 5. Consume the sealed test once

Development baselines exclude every frozen test ID. After C3, run each final production artifact
once with `--split test`. The receipt name binds the dataset-manifest SHA to the model SHA and a
second run is refused:

```bash
bursa-baseline run \
  --config <single-production-artifact-model-config.json> \
  --model <artifact-id> \
  --split test \
  --gold-dir data/gold \
  --dataset-manifest data/manifest.json \
  --test-receipt-dir artifacts/frozen-test-receipts \
  --out runs/frozen-test
```

If either sealed run produces an incorrect automatic post, leave
`BURSA_CALIBRATION_PATH` unset and ship review-only behavior. Never change thresholds or train
again from sealed-test observations.

## 6. External evidence still requiring people or hardware

- The 140 draft cases require human review, and all 24 Yoruba cases require the recorded Yoruba
  reviewer sign-off.
- The physical i5 run requires at least 12 GiB free before model download. Disconnect external
  networking for inference; the runner also starts `llama-server` and `llama-bench` with their
  offline flag.
- Training requires the pinned Linux/CUDA container, a 24 GB-class GPU, and 80 GiB free.
- Ollama and LM Studio checks must be run from clean imports without a custom system prompt.
  Installing LM Studio or removing local files to make space remains an explicit user-approved
  action.
