# Bursa Agent M — Inference Path Design Spec

**Phase:** 2A (Agent M — model & inference, app-side integration)
**Date:** 2026-07-25
**Status:** Approved design (brainstorming complete); implementation plan to follow
**Authoritative work order:** `BURSA_ADTC_EXECUTION_PLAN.md` §4 (Agent M), §3 (architecture), §6 (output contract, frozen), D3 (runtime config), D14 (test prompts)
**Reference baselines:** `docs/MODEL_ARCHITECTURE.md` (§5.4 candidate generation, §6 model, §8 confidence, §11 runtime budget, §12 metrics), `docs/PRD.md` (FR-005 AI reconciliation)
**Builds on:** Phase 1 Financial Core (`docs/superpowers/specs/2026-07-24-bursa-financial-core-design.md`), on `main`.

---

## 0. Status, scope & environment

- **Phase 1 is complete** (C1 met, 63 tests). This sub-project turns Phase 1's `ambiguous → review` stub into the real `candidate generator → prompt → GBNF → llama-server → schema validation → constraint engine (dry-run) → calibrator → review` path.
- **Done boundary:** the *entire* inference path built and unit/integration-tested against a `FakeBackend`, PLUS the real `LlamaServerBackend` adapter and a verification **runbook** for the hardware-bound checks. Nothing half-wired.
- **Environment reality:** this dev/CI environment has no llama.cpp, no GGUF, no Ollama/LM Studio, no i5 hardware. Everything here is tested against the fake backend; real-model behaviour, chat-template checks, and baselines are the runbook (§9), run on the i5 machine.
- **Safety posture:** the model's output is student-level `{student_id, amount}` per §6 (never `fee_id`), the constraint engine still owns all arithmetic, `ledger.py` remains the sole writer, and in v1 **nothing the model touches posts** — model-path results become validated, feature-scored proposals in the review queue.

### Non-goals (this sub-project)
Confidence *training* (Phase 3 — needs the gold set); Agent D data/eval; OCR; UI; the bursar-approves-then-posts review action (a later step); running a real model here.

## 1. Module architecture

```
bursa/
  inference/
    backend.py    # InferenceBackend protocol (+ declared exceptions); LlamaServerBackend; FakeBackend
    server.py     # LlamaServer lifecycle: start/stop/health managed subprocess (D3 config)
    tokens.py     # TokenCounter protocol; QwenTokenizer (real); HeuristicTokenCounter (fallback)
    grammar.py    # build_grammar(txn_id, surviving_candidate_ids, allowed_codes) -> GBNF str
    prompt.py     # PromptBuilder: chat template + static-prefix-first + overflow ladder (<=1300)
    schema.py     # validate(raw, txn_id, candidate_ids, allowed_codes) -> Result
    run.py        # run_inference(): transport retry-once (health/restart); returns raw content
  candidates.py   # generate(conn, txn) -> list[Candidate] (0..5, deterministic)
  features.py     # extract(...) -> dict (versioned, features_version=1)
  calibrator.py   # ModelConfidencePolicy.route(features)->action — model-path confidence seam; v1 -> review
```

**Additive Phase-1 extensions** (no behaviour change to existing code):
- `db.py`: add `student_aliases(student_id, alias, normalized_alias)` table; add `features` (JSON TEXT) column to `proposals`.
- `importers/students.py`: optional `aliases` CSV column → `student_aliases`.
- `projections.py`: add `payer_history(conn, normalized_payer)` — **live allocations only** (net of reversals).
- `reasoncodes.py`: add `PROMPT_BUDGET_EXCEEDED`, `INFERENCE_UNAVAILABLE`, `SCHEMA_INVALID`, `MODEL_RANKED`, `MODEL_ABSTAINED`.
- `pipeline.py`: extend `reconcile()` with the non-exact model branch (§6).

## 2. Candidate generator (`candidates.py`)

`generate(conn, txn) -> list[Candidate]` — length 0–5, deterministically ordered. `Candidate` carries: `student_id`, display name, `aliases`, guardian name(s), outstanding charges `(fee, balance)`, `is_prior_payer`, sibling-group members, and `fired_signals` (names + weights that contributed — recorded for reproducibility).

**Signals** (weights are named constants, recorded in features + decision path):

| Signal | Source | Weight | Pools? |
|---|---|---|---|
| exact name/alias token overlap | `normalized_name` + `student_aliases` | STRONG | yes |
| **fuzzy name similarity ≥ `FUZZY_NAME_THRESHOLD`** | Jaro-Winkler vs `normalized_name` + aliases | STRONG | **yes** |
| guardian name matches payer | `guardians.normalized_name` | STRONG | yes |
| guardian phone-suffix in narration | `guardians.phone_suffix` vs narration digit-runs | STRONG | yes |
| prior confirmed payer→student | `payer_history` (live allocations) | STRONG | yes |
| amount == an outstanding balance | `projections.charge_balance` | STRONG | yes |
| amount == sibling-group balance sum | shared-guardian group | MEDIUM | no (via sibling rider) |
| class/term active | `students.term_id`, `terms.is_active` | WEAK | no |
| narration keyword hit | `narration_tokens` | WEAK | no |

**Fuzzy signal (required):** hand-rolled Jaro-Winkler `similarity(a,b)∈[0,1]` (no dependency), used for **pooling** — token overlap alone cannot pool misspellings (demo scenario 3). Rosters are small, so an O(roster) scan per transaction is acceptable.

**Algorithm (deterministic):** (1) **pool** = students hit by any *strong* signal **+ their siblings** (shared guardian); (2) **score** = weighted sum of fired signals; (3) **rank** by `(score DESC, student_id ASC)`; (4) **cap at `MAX_CANDIDATES`=5**, `log()` dropped count + score-at-cut (no silent truncation).

**`payer_history(conn, normalized_payer)`** (added to `projections.py`): scans **live** `allocation` events, joins to `transactions.payer_name`, normalizes, groups by `student_id`; returns students with a live (non-reversed) allocation for this payer. A reversal removes the mapping — anti-evidence, never history.

## 3. Prompt builder & token budget

### 3.1 TokenCounter (`inference/tokens.py`)
Protocol `count(text) -> int`.
- **`QwenTokenizer`** (production): loads pinned `tokenizer.json` via `tokenizers`; exact counts. Used whenever the asset is present.
- **`HeuristicTokenCounter`** (fallback, asset-absent): conservative-by-construction — `ceil(len(text)/3) + nonascii_char_count + SPECIAL_MARGIN`. Documented bias: over-counts English (BPE ≈ 4 chars/tok) and surcharges multi-byte African names / `₦`.
- **Over-count guarantee:** a corpus test asserts `heuristic(x) >= real(x)` for every prompt in a representative Bursa corpus **whenever the asset is present**; failure ⇒ tighten the divisor. Production uses the exact tokenizer, so the heuristic's slack never wastes production budget — it only sheds a candidate sooner in asset-absent runs (the safe direction).

### 3.2 Chat template & counting (required)
The `PromptBuilder` applies the **full Qwen3 chat template itself** — `<|im_start|>{role}` markers, `/no_think` to disable thinking — producing the **final raw string**, sent to `/completion` (not `/chat/completions`). Token counting is performed on **exactly that final string**; nothing is appended after counting, so template overhead is inside the budget.

### 3.3 Static-prefix-first ordering (required)
Layout maximizes `cache_prompt` prefix reuse: the **system block** (reconciliation instructions, output-contract description, allowed reason codes — all constant across transactions) comes first and is identical every call; the **user block** (this transaction + its candidates) comes last. llama.cpp caches the shared prefix.

### 3.4 Overflow ladder (required, deterministic)
`PROMPT_TOKEN_BUDGET` = 1300. **Never-truncate:** the transaction (id, payer, narration, amount) and the output-contract instructions. If `count > budget`, apply in order, re-counting after each, stopping when it fits:
1. drop the payment-history section — **but keep each candidate's `is_prior_payer` marker**;
2. trim aliases — **but keep any alias that fired in pooling**;
3. reduce candidates 5→4→3 (drop lowest score; `log()` each);
4. still over at 3 → route to review with `PROMPT_BUDGET_EXCEEDED`; do **not** call the model.

**Invariant (required):** every rationale claim in the prompt must be verifiable from content still present — the ladder never sheds fired evidence.

### 3.5 Output-side budgeting (required)
- `n_predict = min(OUTPUT_MAX, CONTEXT_CAP − prompt_tokens − SAFETY_MARGIN)`, passed explicitly to `llama-server`.
- The validator enforces `MAX_EXPLANATION_CHARS`; the grammar bounds JSON structure. Together they hard-cap the output side so generation can't exhaust the 2048 window.

## 4. Grammar, backends & validation

### 4.1 Dynamic GBNF (`inference/grammar.py`)
`build_grammar(txn_id, surviving_candidate_ids, allowed_codes) -> str` — built from the **post-ladder surviving** candidates, never the original five. `transaction_id` → fixed literal; `student_id` → alternation of *only* surviving candidate id literals; `reason_codes` → array of *only* allowed-code literals; `amount_minor` → integer; `recommended_action` → `"auto"|"review"|"unmatched"`; strings bounded. Unknown-ID / invalid-action output is thus structurally un-generatable. **Test: grammar IDs == prompt IDs.**

### 4.2 Backends (`inference/backend.py`)
Protocol `generate(raw_prompt, grammar, n_predict) -> str`; **declares** `BackendTransportError` (timeout / connection / mid-request death).
- **`LlamaServerBackend`**: stdlib `urllib` POST to `127.0.0.1:<port>/completion` with `{prompt, grammar, n_predict, temperature: 0, cache_prompt: true}`; timeout; raises `BackendTransportError` on transport faults; returns `content`.
- **`FakeBackend`**: programmable canned string / callable, and **can raise `BackendTransportError` programmably** so the transport-retry path is testable here.

### 4.3 Server lifecycle (`inference/server.py`)
`LlamaServer` context manager: `start()` spawns `llama-server --ctx-size 2048 --threads 4 --temp 0 --cache-type-k q8_0 --cache-type-v q8_0 --port … + prompt cache`; `health()` polls `/health`; `stop()` terminates. Unit-tested via mocked subprocess (arg construction, health parsing); real launch is the runbook.

### 4.4 Schema validation + split retry (`inference/schema.py`, `inference/run.py`)
`validate(raw, txn_id, candidate_ids, allowed_codes)`: parse JSON → `transaction_id == txn_id` → every `student_id ∈ candidate_ids` → `amount_minor` non-neg int → `reason_codes ⊆ allowed` → `recommended_action` valid → `len(explanation) ≤ MAX_EXPLANATION_CHARS`.

**Split retry policy (required):**
- **Content-invalid → review immediately, NO retry** (temp 0 is deterministic; a re-run reproduces the same output).
- **Transport failure → retry once**: `run_inference` catches `BackendTransportError`, health-checks, restarts the server if unhealthy, retries once; still failing → raise → pipeline routes review with `INFERENCE_UNAVAILABLE`.

## 5. Features & calibrator

### 5.1 Frozen feature table (`features.py`, `features_version = 1`)
Recorded on every model-path proposal. Editing any row bumps the version.

| Feature | Type | Range | Exact computation |
|---|---|---|---|
| `name_alias_similarity` | float | 0..1 | max Jaro-Winkler between normalized payer/narration tokens and chosen student's `normalized_name` + aliases |
| `guardian_relationship` | int | {0,1} | 1 if chosen student's guardian `normalized_name` matches normalized payer |
| `amount_to_balance_agreement` | float | 0..1 | `1 − min(1, abs(allocated − outstanding)/max(outstanding,1))` for the chosen student |
| `historical_payer_consistency` | int | {0,1} | 1 if `is_prior_payer` (live `payer_history` hit) for chosen student |
| `candidate_separation` | float | 0..1 | `(top_score − second_score)/max(top_score,1)` from generator scores |
| `llm_ranking_consistency` | int | {0,1} | 1 if model's primary student == generator's top-ranked candidate |
| `constraint_validation_result` | int | {0,1} | 1 if dry-run `constraints.validate` ok |
| `budget_shed` | int | {0,1} | 1 if the overflow ladder dropped anything |
| `candidate_count` | int | 1..5 | number of surviving candidates |

### 5.2 Calibrator v1 (`calibrator.py`)
`ModelConfidencePolicy` is the **model-path** confidence component: `route(features) -> RecommendedAction`. It occupies the same architectural seam as Phase-1's `ConfidencePolicy` (which only ever governed the deterministic pass-through), but takes the extracted `features` because a features-blind `route(proposal)` cannot calibrate — this is a deliberate generalization of the seam, not a reuse of the identical signature. It computes + stores the feature-based score, but **v1 always returns `review`** (untrained ⇒ model never auto-posts; zero-false-auto-post holds trivially). Phase 3 fits the logistic model to the recorded features and replaces `route`'s internals behind this signature. Deterministic exact matches never reach the calibrator (they short-circuit in the pipeline, §6).

## 6. Pipeline wiring (`pipeline.py`)

```
reconcile(conn, txn_id, config, backend):
  p = matcher.match(txn)                                  # exact deterministic
  if p.action == AUTO:
      if config.auto_post_enabled: distribute+post -> "auto"
      else:                        store-review(p) -> "review"     # short-circuit; NO model, no reinterpret
      return
  cands = candidates.generate(txn)                         # [] -> set unmatched -> "unmatched"
  prompt, surviving = PromptBuilder.build(txn, cands)      # ladder; None -> review PROMPT_BUDGET_EXCEEDED
  grammar = build_grammar(txn_id, ids(surviving), ALLOWED_CODES)   # ids == prompt ids
  try:    raw = run_inference(backend, prompt, grammar, n_predict) # transport retry-once inside
  except BackendTransportError: return store-review(INFERENCE_UNAVAILABLE) -> "review"
  r = schema.validate(raw, txn_id, ids(surviving), ALLOWED_CODES)
  if not r.ok:  return store-review(SCHEMA_INVALID) -> "review"    # content-invalid: no retry
  dry   = constraints.validate(txn, distribute(r))         # dry-run = feature, NOT a post
  feats = features.extract(txn, surviving, r, dry)         # features_version=1
  calibrator.route(feats)                                  # v1 -> review
  return store-proposal(source="llm", r, feats) -> "review"
```

Model-path proposals are stored in `proposals` (+ `proposal_allocations`, `features` JSON), routed to review. Nothing model-touched posts in v1.

## 7. Named constants (frozen in code)

| Constant | Value |
|---|---|
| `PROMPT_TOKEN_BUDGET` | 1300 |
| `CONTEXT_CAP` | 2048 |
| `OUTPUT_MAX` | 512 |
| `SAFETY_MARGIN` | 64 |
| `MAX_EXPLANATION_CHARS` | 500 |
| `FUZZY_NAME_THRESHOLD` | 0.88 (Jaro-Winkler) |
| `MAX_CANDIDATES` | 5 |
| `TRANSPORT_RETRY` | 1 |
| signal weights | `STRONG=10, MEDIUM=5, WEAK=2` |
| `SPECIAL_MARGIN` (heuristic) | 8 |
| `features_version` | 1 |

## 8. Testing strategy (all against `FakeBackend`, offline)

- **Candidates:** exact + fuzzy pooling (misspelling reaches the set), guardian + phone-suffix signals, sibling riders, `payer_history` net-of-reversals, ≤5 cap with logged drops, deterministic order, recorded `fired_signals`.
- **Tokens:** heuristic determinism; **corpus over-count test** (`heuristic ≥ real` when asset present).
- **Prompt:** chat template applied + counted on final string; static-prefix-first; ladder steps in exact order; never-truncate narration/contract; **fired-evidence preserved** (kept alias, `is_prior_payer` marker); `PROMPT_BUDGET_EXCEEDED` at 3.
- **Grammar:** candidate-restricted; **grammar IDs == prompt IDs**; a conforming JSON parses; unknown id un-generatable.
- **Backend/lifecycle:** `FakeBackend` canned + raising `BackendTransportError`; `LlamaServerBackend` request construction (mocked urllib); server arg construction + health parse (mocked subprocess).
- **Validation/retry:** unknown-ID / bad-action / over-long-explanation → review; content-invalid → review **no retry**; transport error → health/restart/retry-once → success or `INFERENCE_UNAVAILABLE`.
- **Features/calibrator:** each feature computed per the frozen table + `features_version=1`; calibrator always `review`.
- **Pipeline end-to-end (FakeBackend):** exact→auto; exact + flag-off→review-no-model; no-candidates→unmatched; valid model result→review with stored proposal + features; budget-exceeded→review; transport-fail→review; content-invalid→review.

## 9. Verification runbook (`docs/superpowers/runbooks/agent-m-verification.md`)

Hardware-bound steps, run by the team on i5-class hardware (I cannot run them here):
0. **Pin the toolchain (required):** record the exact `llama.cpp` version / commit hash used. Grammar and `/no_think` behaviour vary across releases, and C2 depends on reproducible numbers — so this hash is recorded alongside *every* result below.
1. `bash download_model.sh` — fetches the GGUF **and `tokenizer.json` with a pinned SHA-256 checksum** (script updated: `TOKENIZER_URL` + `TOKENIZER_SHA256`, verified post-download).
2. Start `llama-server` with the D3 config; confirm `/health`.
3. Run the **2 D14 test prompts** through `LlamaServerBackend`; inspect JSON + reasoning.
4. **N=1000** consecutive app inferences over **distinct inputs** (required): record valid-JSON pass/fail counts (**target ≥99.5%**) and unknown-ID count (**target 0%**). At temp 0 an identical prompt yields identical output, so the 1000 prompts MUST differ — **1000 runs over the 13 demo fixtures is only 13 effective samples and does NOT satisfy this.** Either sequence this check *after* Agent D's synthetic generator exists, or drive it with deterministic, reproducibly-seeded fixture permutations (name / amount / narration-noise variants) so all 1000 prompts are unique.
5. **No-think assertion:** under `/no_think` + grammar, verify output contains **no think-block content** and begins immediately with grammar-valid JSON (thinking leakage would break parsing and consume output budget).
6. Bare-GGUF chat-template check in clean **Ollama** and **LM Studio** (no system prompt) — the 2 test prompts + generic enterprise prompts answer well.
7. Record tps / peak-RSS / temperature on i5-class — **tagged with the pinned `llama.cpp` hash (step 0)** so C2's comparison across runs is reproducible.

## 10. Acceptance criteria (Phase 2A)

- All §8 tests green, offline, against `FakeBackend`.
- Grammar IDs always equal prompt (surviving) IDs; unknown-ID output un-generatable and, if ever present, rejected by the validator.
- Prompt always ≤ `PROMPT_TOKEN_BUDGET` on the exact final string, or routed `PROMPT_BUDGET_EXCEEDED`; narration/contract never truncated; fired evidence never shed.
- Content-invalid → review with no retry; transport-fault retried once then review.
- Model-path results never post (v1); every model-path proposal carries features + `features_version=1`.
- Real `LlamaServerBackend` + `LlamaServer` lifecycle exist and are unit-tested via mocks; the runbook (§9) documents the hardware verification incl. N=1000 and the no-think assertion.

## 11. Deferred (later phases)
Calibrator *training* → Phase 3 (needs gold set); real-model valid-JSON / tps / RSS / thermal + chat-template → runbook on i5 hardware; Agent D data/eval; OCR; UI; bursar-approves-then-posts review action.

**Carry-over requirement into Agent D:** the evaluation harness must measure **candidate-pool recall on the gold set** — the fraction of gold cases whose correct student(s) appear in the generated ≤5 candidates. This is the metric that validates `FUZZY_NAME_THRESHOLD` and the pooling signals (a threshold too high silently drops correct students before the model ever sees them). Design Agent D's harness to report it.
