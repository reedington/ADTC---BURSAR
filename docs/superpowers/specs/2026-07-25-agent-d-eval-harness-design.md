# Bursa Agent D — Eval Harness Design

**Status:** approved (brainstorming) · **Date:** 2026-07-25 · **Sub-project:** Agent D #2 (evaluation)
**Depends on:** Agent F financial core (`bursa/`), Agent M inference path (`bursa/inference/`, `bursa/pipeline.py`), Agent D data foundation (`bursa_eval/` schema, loader, synth, dataset).

## 1. Purpose & scope

One cohesive evaluation harness that scores any model, through the **real serving pipeline**, and emits a single scorecard. It is the measurement instrument for every baseline, every fine-tune candidate, and the Phase-3 calibrator gate. It is *not* the official ADTC profiler (that is Agent P, run separately on i5 hardware); this harness is Bursa's internal, offline-testable scoring machine, and it is deliberately built so its numbers can never diverge from what the app would actually produce.

Three suites, one seam, one command:

- **Bursa-gold** — the gold cases run through the real pipeline; owns the reconciliation metrics and the two hard safety gates.
- **ADTC-validation** — an lm-eval-compatible adapter; official judge set when obtained, permissively-licensed proxy general-capability set as interim C3 regression guard.
- **Bare-model** — generic-enterprise + the two D14 visible prompts, run with each model's embedded chat template, for format-leakage / abstention / quality checks.

### 1.1 The seam (locked)

Every suite drives models through the existing `bursa.inference.InferenceBackend` protocol. `FakeBackend` runs everything offline in CI (deterministic canned outputs → the scoring math is fully testable without weights or network); `LlamaServerBackend` produces real numbers on the i5. Harness code is identical across both — only the backend and the ingested perf file differ.

## 2. Package layout

```
bursa_eval/harness/
  runner.py      # evaluate_case() + the three suite drivers; emits per-case records
  metrics.py     # pure (records) -> value functions; the §12 family + carry-ins
  scorecard.py   # aggregation, provenance, hard-gate exit codes, --diff, CLI
data/adtc/
  *.jsonl        # lm-eval-shaped placeholder: {id, prompt, expected, choices?}
  PROVENANCE.md  # per-proxy-dataset license + source (training-30% rule)
tests/
  test_harness_metrics.py   # metric math on hand-authored record fixtures
  test_harness_runner.py    # FakeBackend drives the real pipeline end-to-end
  test_scorecard.py         # smoke: json + records + sidebyside + gates + --diff
docs/superpowers/runbooks/eval-harness-i5.md   # the single consolidated hardware session
```

## 3. What the harness scores (the central decision)

In v1 the calibrator **always** routes the model path to `review`. Scoring the *calibrated routing* would make the action metric ~100% "review" and meaningless. The harness therefore scores the **model's raw validated output** — `candidate_allocations`, `recommended_action`, `explanation`, `interpretation` — against `expected`, independent of the v1 safety routing. It measures reconciliation *quality*; the safety layer is measured separately by the auto-post gate.

## 4. Per-case evaluation — `evaluate_case(case, backend)`

Mirrors the real pipeline but captures the raw model output:

1. `loader.materialize(case)` → real in-memory DB (history replayed through the ledger).
2. `matcher.match`. If it **exact-matches** (deterministic short-circuit): score the deterministic outcome and mark `would_auto_post=True`; **exclude from `pool_recall`** (no pool was built).
3. Else `candidates.generate` → record `pool_recall_hit = case.pool_truth() ⊆ pool` (non-exact cases only).
4. `prompt.build` → `run_inference(backend)` → `schema.validate_output` → raw `data` (or a validity failure recorded as `valid_json=False`).
5. `distribute` + `constraints.validate` as a **dry-run** → financial-validity signal (`dry_run_ok`).
6. `duplicate_blocked` cases: **do not run the model.** Assert the import/dedup layer blocks the case transaction (reference present from replayed history). Scored at pipeline level → `dup_blocked`.
7. Return the per-case record (§6).

## 5. The three suite-runners

### 5.1 Bursa-gold runner
Iterates gold cases through `evaluate_case`, aggregating the §7 metrics. Owns **`incorrect_auto_posts`**: for every case the pipeline *would auto-post* (today: deterministic exact matches; Phase 3: calibrated high-confidence), check the auto-posted allocation against `expected`; any mismatch counts, required value **0**. The gold suite is **Qwen-template-bound by design** (the app path self-applies the Qwen3 chat template, locked in Agent M); therefore C2's pipeline comparison is the **two Qwen models** (1.7B vs 0.6B), not a cross-architecture comparison.

### 5.2 ADTC-validation runner
An **lm-eval-compatible adapter** over the seam:
- **Format:** documented placeholder `data/adtc/*.jsonl` (`{id, prompt, expected, choices?}`) that maps onto an lm-eval task, so the **judge-distributed set drops in unchanged** (bare prompt → backend → exact/choice match).
- **Interim proxy (C3 amendment):** a permissively-licensed general-capability set (e.g. MMLU/ARC subset) runs through the same adapter as the regression guard. The proxy is a **forgetting detector, not a domain-fit measure**: C3 compares **relative deltas** (post − pre fine-tune), **never absolute proxy scores**. Every proxy dataset's real license is verified and recorded in `data/adtc/PROVENANCE.md` before its files are committed (training-30% provenance rule).
- The scorecard labels results `official` vs `proxy` so the interim number is never mistaken for the real audit.

**Access status:** the official Corporate/Enterprise validation set is **access-gated — judge-distributed** (confirmed: the ADTC profiler's `accuracy.py` states the real audit uses the hidden 30% validation subset distributed by judges; the public ADTC repos contain only template + profiler code). This is a human dependency to resolve; the proxy guards C3 until it arrives.

### 5.3 Bare-model runner
Generic-enterprise prompts (summarize / draft / analyze / reconcile-conversationally) **plus the two D14 visible test prompts verbatim**, as named, individually-tracked cases (the only prompts judges are guaranteed to run). **Chat template:** the bare-model runner applies each model's **GGUF-embedded chat template** (llama-server chat endpoint or equivalent) — **never** the app's self-applied Qwen template — because the judges' tools read the embedded template and that embedded-template behavior is the artifact under test. (The app path keeps self-application; that remains locked.)

Automated scoring is structural: valid output, the **D6 tripwire** (a generic prompt must not produce reconciliation-JSON spew), and calibrated abstention on the noisy prompt. Open-ended quality is not faked into a number — it is routed to the human rubric via the side-by-side file (§8).

## 6. Per-case record schema

One row per case, JSONL, keyed by `case_id`:

```
suite, case_id, family, language, difficulty,
would_auto_post, correct_action, top1_hit, exact_alloc_hit,
pool_recall_hit|null, valid_json, dry_run_ok, dup_blocked|null,
timings{...}   # diagnostics ONLY — never enter the scorecard perf section
```

Records are written per run so two runs diff by a plain `case_id` key-join → "which cases regressed," which is the question Phase-3 debugging actually asks.

## 7. Metrics (`metrics.py`, pure)

Consumes records; never re-runs a model. Each metric is `(records) -> value`, unit-testable against hand-built records.

| Metric | Definition |
|---|---|
| `incorrect_auto_posts` | **HARD-ZERO GATE** — over would-auto-post cases, count where posted allocation ≠ expected |
| `duplicate_blocked_rate` | **HARD-100% GATE** — fraction of `duplicate_blocked` cases the import layer blocked; below 100% is a money-doubling regression |
| `action_accuracy` | aggregate of `correct_action` (model `recommended_action` == expected outcome) |
| `valid_json_rate` | fraction of outputs passing `schema.validate_output` |
| `top1_student_accuracy` | primary student == expected primary |
| `exact_allocation_accuracy` | full allocation set (student+charge+minor units) == expected |
| `sibling_split_accuracy` | exact-allocation restricted to multi-student cases |
| `abstention_precision` / `_recall` | model abstains ⇔ `case.is_abstention()` |
| `pool_recall` | over **non-exact** cases only, `case.pool_truth() ⊆ pool` |
| `unsupported_id_rate` | model names a student not in the pool |
| `language_subset_accuracy` | top-1 accuracy sliced by `language` (en/pcm/yo/ha/ig) |

**Model-side abstention is defined explicitly:** `candidate_allocations == []` **or** `recommended_action == "unmatched"`, matched against `case.is_abstention()`. Review-*with*-allocations cases (e.g. sibling splits routed to review) are therefore **not** miscounted as abstentions.

ADTC suite → `{accuracy, per_task}` labeled `official|proxy`, plus `regression_delta` (post − pre) — the only C3-reported number. Bare-model suite → structural pass/fail (valid, D6-tripwire-clean, abstention) + pointer to the side-by-side file.

## 8. The scorecard (`scorecard.py`) — one command

`python -m bursa_eval.harness.scorecard --backend fake|llama [--perf perf.json] [--label NAME]`

- Runs the three suites through the chosen backend; writes `runs/<label>/{records.jsonl, scorecard.json}`.
- `scorecard.json` sections: `bursa_gold` (all §7 metrics), `adtc` (official/proxy + `regression_delta`), `bare_model` (structural + pointer), `perf` (**ingests profiler/runbook numbers only** via `--perf`; never harness timings), and a **`provenance`** block: git commit, model path + sha256, backend config, seeds — so any cross-candidate diff is self-describing.
- Prints a human table with the two gates at the top; **exits non-zero if `incorrect_auto_posts != 0` or `duplicate_blocked_rate < 100%`** — CI- and gate-enforceable.
- **`--diff runs/A runs/B`** emits the per-case regression list (key-join on `case_id`).
- **`sidebyside.md` is generated by comparing two run directories** (zero-shot vs candidate) — `--sidebyside runs/zeroshot runs/candidate` — not by a single run. It lays out prompt / zero-shot output / candidate output for the human rubric (correct · coherent · no format leakage · appropriate abstention), which is what C3's quality-holds clause means concretely.

## 9. Testing strategy

All three layers run on dev hardware / CI with `FakeBackend`, deterministically, no weights, no network.

1. **Metrics unit tests** (`test_harness_metrics.py`) — pure, against hand-authored record fixtures: an `incorrect_auto_posts` case (→ 1, non-zero exit), a review-with-allocations case (asserts *not* counted as abstention), a `pool_recall` fixture over non-exact cases, an unblocked-duplicate fixture (asserts non-zero exit). Proves the gate math. Never weaken an assertion to make code pass.
2. **Runner integration tests** (`test_harness_runner.py`) — a handful of gold cases through `evaluate_case`; `FakeBackend` returns a canned validated output per case, so the *pipeline* runs for real while the *model* is deterministic. Asserts record schema completeness + `case_id` keying, and that a known-good case scores perfectly / a known-bad case misses as expected.
3. **Scorecard smoke** (`test_scorecard.py`) — `--backend fake` over the committed gold examples emits json + records; asserts the provenance block is populated, both gates present, and `--diff` / `--sidebyside` on two run dirs yield the expected deltas.

## 10. Offline vs. i5 — the consolidated hardware session

The **`eval-harness-i5.md` runbook is the single consolidated hardware session**, cross-referencing the Agent M verification runbook. In one sitting it covers: model downloads (via `download_model.sh`), the Agent M assertions (from that runbook), scorecard runs (`--backend llama`) for **Qwen3-1.7B and Qwen3-0.6B**, and per-model profiler perf — producing **all C2 inputs in one sitting**. The profiler `perf.json` from this session is what the scorecard's `perf` section ingests; the harness never self-reports tps/RSS. The bare-model side-by-side + human rubric pass is also part of this session.

## 11. Acceptance (maps to the execution plan's Agent D criteria)

- One command emits the full scorecard across the three suites.
- `incorrect_auto_posts == 0` and `duplicate_blocked_rate == 100%` enforced as non-zero exit codes.
- `pool_recall` computed over non-exact cases via `case.pool_truth()`; `duplicate_blocked` evaluated at pipeline level (import layer blocks them).
- ADTC adapter is lm-eval-shaped with the official set droppable in unchanged; proxy guards C3 by relative delta until then.
- Full suite is FakeBackend-deterministic in CI; real numbers + perf come from the consolidated i5 runbook.
- README status table updated on merge.
