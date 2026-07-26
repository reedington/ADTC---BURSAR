# Real-GGUF functional smoke test — 2026-07-26

These runs exercised the pinned Qwen3 Q4_K_M artifacts through the real `llama-server`,
tokenizer, prompt, GBNF, schema validator, dry-run constraint engine, and scorecard. They ran on
an Apple ARM64 development machine, not the competition's i5 target. They are functional
regression evidence only; none of their timings are eligible for `REPORT.md`.

Configuration shared by both runs:

- llama.cpp build 9850, commit `4f31eedb0`;
- context 2048, four threads, temperature 0, q8_0 K/V cache;
- 14 authored gold cases, including the deterministic duplicate case;
- app prompt with `/no_think` and bounded GBNF output;
- no official ADTC validation data and no performance claims.

| Artifact | Valid JSON | Unsupported IDs | Incorrect auto-posts | Action accuracy | Top-1 student | Exact allocation |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | 100% | 0% | 0 | 46.2% | 91.7% | 83.3% |
| Qwen3-1.7B Q4_K_M | 100% | 0% | 0 | 23.1% | 58.3% | 58.3% |

The initial 0.6B run found two truncated JSON responses. The grammar allowed unbounded array
cardinality even though individual strings were bounded. After allocations, reason codes,
interpretation lists, and ambiguities were cardinality-bounded and the grammar-level explanation
limit was reduced to 240 characters, the identical 14-case run reached 100% schema-valid JSON.
The independent application validator still accepts explanations up to the locked 500-character
maximum.

Both candidates still score 0% on the single Pidgin case and 0% on sibling-split accuracy. The
bare-model smoke prompts also showed thinking-token exhaustion and weak answers on some prompts.
Accordingly, neither candidate has passed C2, and this small development set is not a model-choice
decision. The locked 1,000-prompt, bare-model, official/proxy, and i5 profiler runs remain required.
