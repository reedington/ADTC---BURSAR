# Eval-harness i5 session (single consolidated hardware sitting)

Cross-reference: [`agent-m-verification.md`](agent-m-verification.md). **This one session produces
ALL C2 inputs.** Benchmarks count only on i5-class hardware at 4 threads (Prime Directive 5).

## 0. Prereqs
- `bash download_model.sh` — fetches the zero-shot Qwen3-1.7B Q4_K_M + the 0.6B baseline + the
  tokenizer; verify the TOFU checksums it prints.
- Pin the llama.cpp build (see the Agent M runbook for the exact commit).

## 1. Agent M assertions
Run the Agent M verification runbook's assertions first (N=1000 distinct inputs, `/no_think`
present, overflow ladder 5→4→3). These gate the model path the Bursa-gold suite depends on.

## 2. Start llama-server
One server per model serves both endpoints the harness uses:
- `/completion` (grammar-constrained) — the Bursa-gold pipeline.
- `/v1/chat/completions` (embedded chat template) — the ADTC + bare-model suites via `backend.chat`.

## 3. Scorecard per model — C2 compares the two QWEN models
The Bursa-gold suite is **Qwen-template-bound by design** (the app self-applies the Qwen3 template,
locked in Agent M), so C2's pipeline comparison is 1.7B vs 0.6B:

    python -m bursa_eval.harness.scorecard --backend llama \
      --model-path model/qwen3-1.7b-q4.gguf --tokenizer model/tokenizer.json \
      --perf perf-1.7b.json --label qwen3-1.7b
    python -m bursa_eval.harness.scorecard --backend llama \
      --model-path model/qwen3-0.6b-q4.gguf --tokenizer model/tokenizer.json \
      --perf perf-0.6b.json --label qwen3-0.6b

Both MUST exit 0. A non-zero exit means `incorrect_auto_posts != 0` or `duplicate_blocked_rate < 100%`
— a wrong auto-post or an unblocked (money-doubling) duplicate. **STOP and investigate; do not merge.**

## 4. Bare-model + side-by-side (C3 quality-holds)
Author the two D14 visible prompts into `data/bare/prompts.jsonl` first (they are ours to write for
`metadata.json`; no hardware needed). The bare suite runs each GGUF's **embedded** chat template
(`backend.chat` → `/v1/chat/completions`), NOT the app's self-applied Qwen template.

Produce the human side-by-side by comparing the zero-shot run against the candidate run:

    python -m bursa_eval.harness.scorecard --sidebyside runs/zeroshot runs/qwen3-1.7b > sidebyside.md

Score `sidebyside.md` against the rubric (correct · coherent · no format leakage · appropriate
abstention) = C3's quality-holds clause.

## 5. ADTC / proxy (C3 forgetting-detector)
Replace `data/adtc/proxy/sample.jsonl` with a **license-verified** general-capability subset and
record it in `data/adtc/PROVENANCE.md`. Run pre- and post-fine-tune; report the **relative delta**
(`regression_delta`), never the absolute proxy score. Drop the judge-distributed official set into
`data/adtc/official/*.jsonl` when obtained — the adapter loads it unchanged.

## 6. Profiler perf
Per-model tps / RSS / peak-temp come from the ADTC profiler and feed the scorecard `perf` section
via `--perf`. Harness per-case timings are diagnostics only and are never used here.
