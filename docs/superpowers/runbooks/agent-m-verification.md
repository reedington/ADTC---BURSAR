# Agent M — Verification Runbook (i5-class hardware)

These steps CANNOT run in the dev/CI environment (no model, no hardware). Run them on the
ADTC standard laptop profile and record results in REPORT.md.

0. **Pin the toolchain (required):** record the exact `llama.cpp` version / commit hash used.
   Grammar and `/no_think` behaviour vary across releases; C2 needs reproducible numbers.
   Tag every result below with this hash.

1. `bash download_model.sh` — fetches the verification GGUF + `tokenizer.json`.
   - **Verification/baseline artifacts are the ZERO-SHOT Qwen3 GGUFs (exist today):** the 1.7B
     Q4_K_M (`unsloth/Qwen3-1.7B-GGUF`, ~1.03 GB) is what `download_model.sh` fetches; the 0.6B
     Q4_K_M (`unsloth/Qwen3-0.6B-GGUF`, ~378 MB) is the C2-pivot baseline (URL in the script
     comments). The tokenizer is Qwen3-1.7B's (`Qwen/Qwen3-1.7B/tokenizer.json`). At Phase 3/4
     the GGUF is repointed to the fine-tuned Bursa-Recon.
   - **Checksums are trust-on-first-use.** On first download the script prints the computed
     SHA-256 for each file; verify each against its Hugging Face file page, then set
     `MODEL_SHA256` / `TOKENIZER_SHA256` and commit the confirmed values. Subsequent runs fail
     closed on mismatch. (No placeholder hashes are ever committed.)
   - Export `QWEN_TOKENIZER_JSON=model/tokenizer.json` so the app AND the token corpus
     over-count test (`tests/test_tokens.py::test_heuristic_overcounts_corpus`) use the exact
     tokenizer.

2. Start `llama-server` with the D3 config (see `bursa/inference/server.py::build_server_args`);
   confirm `GET /health`.

3. Run the 2 `metadata.json` test prompts through `LlamaServerBackend`; inspect JSON + reasoning.

4. **N=1000 over DISTINCT inputs (required).** At temp 0 an identical prompt yields identical
   output, so 1000 runs over the 13 demo fixtures = 13 effective samples and is invalid.
   - **Approach chosen: sequence after Agent D (Phase 2B).** Agent D's synthetic generator
     (deterministic, reproducibly-seeded name/amount/narration-noise permutations) is the source
     of the 1000 distinct inputs. Rationale: building a fixture-permutation harness now would
     duplicate Agent D's generator, which is being built next. (Carry-over recorded in the
     Agent D requirements.)
   - **Fallback if Agent D slips:** add a small deterministic seeded fixture-permutation script
     under `tests/fixtures/` that mutates the 13 demo cases into 1000 unique prompts.
   - Record valid-JSON pass/fail counts (**target ≥99.5%**) and unknown-ID count (**target 0%**).

5. **No-think assertion (required):** under `/no_think` + grammar, verify the output contains
   **no think-block content** (no `<think>…</think>`) and begins immediately with grammar-valid
   JSON. Thinking leakage would break parsing and consume the output budget.

6. Bare-GGUF chat-template check in clean **Ollama** AND **LM Studio** (no system prompt): the
   2 test prompts + generic enterprise prompts answer well.

7. Record tps / peak-RSS / temperature, **tagged with the pinned `llama.cpp` hash (step 0)** so
   C2's comparison across runs is reproducible.
