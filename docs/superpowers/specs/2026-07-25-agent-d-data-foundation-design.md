# Bursa Agent D — Data Foundation Design Spec

**Phase:** 2B (Agent D — data & evaluation), sub-project 1 of 2: **data foundation**
**Date:** 2026-07-25
**Status:** Approved design. Schema + validator + scaffold already SHIPPED (branch `phase-2b-agent-d`); generator, splits, and assembly are the implementation plan.
**Work order:** `BURSA_ADTC_EXECUTION_PLAN.md` §4 (Agent D), D6 (dual-format training mix), D10 (languages), MODEL_ARCHITECTURE §9 (dataset), §9.3–§9.4 (splits, quality controls).
**Builds on:** Phase 1 Financial Core + Phase 2A Agent M (both on `main`). Reuses their constraint engine, ledger, candidate generator, and prompt builder.

---

## 0. Status & scope

**Already shipped** (this sub-project's early deliverable, committed on branch):
- `bursa_eval/models.py` — the gold-case schema (Pydantic, v3).
- `bursa_eval/loader.py` — materializes a case into a real in-memory DB, replaying `setup.history` through the ledger.
- `bursa_eval/goldcheck.py` — the validator (constraint-engine-gated) + CLI + coverage report.
- `bursa_eval/goldnew.py` — the scaffold command.
- `data/gold/` — 3 worked examples (easy/medium/hard); `tests/test_goldset.py` validates them in CI.

**Designed here, to be built:** the synthetic generator, the D6 dual-format renderers, near-dup detection, the split algorithm, and dataset assembly + manifest.

**Non-goals:** the evaluation harness (sub-project 2); the general-enterprise 30% training data (a separate licensed source, §9); actually authoring the 150 gold cases (human work, in parallel).

## 1. Gold-case schema (v3) — SHIPPED

A case is a self-contained mini-world + the correct answer, authored in **YAML, one file per case** (`data/gold/<id>-<lang>.yaml`), amounts in **naira**. Full field set is in `bursa_eval/models.py`; the load-bearing rules:

- **Money:** `amount_naira` is **int or quoted string only**; a YAML float node is rejected (`_reject_float`) — INV-03 extended to the data. Converted to minor units via `money.py`.
- **`setup.history`:** prior transactions + allocations the loader **replays through the real ledger** (`ledger.post`) before the case transaction — makes instalment, duplicate-reference, and payer-history families expressible.
- **`expected.outcome`** ∈ `auto | review | unmatched | duplicate_blocked` (the last is an import-layer block, not a routing state).
- **`expected.allocations`** (the correct split), **`expected.credits`** (holder, amount — asserts overpayment surplus), **`expected.pool_must_include`** (pool-recall truth; defaults to the allocation students; set explicitly for empty-allocation abstention cases).
- **Split keys:** `guardian_family`, `template_family`. **Coverage keys:** `scenario_family` (one of 14 validated families), `language` (`en|pcm|yo|ha|ig`), `difficulty`.
- Abstention is **derived** (`allocations == []`), not a stored field.

## 2. Validator (`goldcheck`) — SHIPPED

`check_case(case) -> list[str]` (empty = valid). Reuses the **production constraint engine as the validity oracle**: it materializes the case, then posts `expected.allocations + expected.credits` through `ledger.post` — an answer that violates any invariant (over-allocation, charge overfill, credit imbalance) is rejected *at authoring time*. `duplicate_blocked` cases assert the reference already exists in replayed history. Runs in **CI** (`tests/test_goldset.py`), so no invalid case merges. The CLI reports per-file status, family/language coverage, gaps, and the abstention/non-auto ratios (warns if non-auto < 25%).

## 3. Scaffold (`goldnew`) — SHIPPED

`python -m bursa_eval.goldnew --family <f> --lang <l> --difficulty <d>` writes a commented template YAML. `render()` is pure (testable without file IO).

## 4. Synthetic generator (`bursa_eval/synth/`)

Produces ~4,000 synthetic reconciliation examples + the deterministic distinct-input source for the runbook's N=1000 check. **Hybrid:** per-family template generators produce structurally-correct base cases; deterministic perturbation layers add realism.

- `namepools.py` — Nigerian first/last name pools per language, alias/abbreviation rules.
- `templates.py` — per-family `gen(rng) -> GoldCase`, computing the correct `expected` by construction, in a **disjoint `synth-*` `guardian_family`/`template_family` namespace** (so synthetic components never straddle into the gold hidden set).
- `perturb.py` — deterministic layers: `ocr_corrupt` (O↔0, I↔1, S↔5 on narration/reference), `to_pidgin` (Pidgin phrasing templates), `inject` (append a malicious payload; the **expected answer is unchanged** — narration is data), `name_variant` (initials, abbreviations, edit-distance misspellings).
- `generate.py` — `generate(base_seed, n, config) -> list[GoldCase]`, a **pure function** of its inputs.

**Determinism (required fix):** per-case seeds via a **stable hash**, never builtin `hash()` (which is PYTHONHASHSEED-salted per process):
```python
def _seed(base_seed, template_id, index) -> int:
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{template_id}:{index}".encode()).digest()[:8], "big")
```
Proven by a **two-subprocess byte-identical test** (generation spawned twice via `subprocess`; outputs asserted equal) — same-process tests cannot catch process-salt regressions.

**Validation gate:** every generated case runs through `goldcheck.check_case` (financial validity by construction). **A synthetic case that near-dups a gold case (§7 signature) is dropped and regenerated with a new seed** — gold is scarce and must never be pulled toward train by a synthetic collision; regeneration is free.

**Abstention:** the family mix in `config` includes ambiguous/no-candidate/unmatched templates; `generate` enforces the ≥25% floor by construction.

**Process gate (required):** the **Pidgin phrasing templates receive native-speaker review before the first mass generation** — template flaws replicate at 4,000× scale, and the D10 Alpha-language claim rests on authenticity. Blocks mass Pidgin generation until reviewed.

## 5. Dual-format rendering (`bursa_eval/synth/render.py`, D6)

Each reconciliation example is emitted in **both** formats.

**`to_app_format` — via the REAL Agent M path (required, no train/serve skew).** It does not reimplement any prompt string. It **materializes the case** (`loader.materialize`), runs the **real `candidates.generate`** and the **real `PromptBuilder.build`** (real Qwen3 chat template, static-prefix-first, overflow ladder) to produce the prompt, then builds the §6 JSON target. Training and serving prompt distributions are therefore identical by construction.

**§6 target-field derivation** (the completion the model is trained to produce):

| §6 field | derived from |
|---|---|
| `transaction_id` | the case transaction id |
| `interpretation.payer_name` | `transaction.payer_name` |
| `interpretation.student_mentions` | narration tokens matching candidate names/aliases |
| `interpretation.term` / `fee_types` | `setup.term`; the `fee_id`s in `expected.allocations` |
| `interpretation.payment_intent` | `scenario_family` → intent mapping (documented) |
| `candidate_allocations[].student_id / amount_minor` | `expected.allocations` |
| `candidate_allocations[].reason_codes` | `FAMILY_REASON_CODES[scenario_family]` (synthetic templates may specify directly) |
| `recommended_action` | `expected.outcome` |
| `explanation` | `expected.rationale` |
| `ambiguities` | scenario-derived (empty unless the template specifies) |

`duplicate_blocked` cases produce **no app-format prompt** (they are blocked at import and never reach the model) — they train the import/dedup layer, not the model, and are excluded from `to_app_format`.

**`to_chat_format`** — a self-contained scenario in a single user turn, **no system prompt**, with a worked reasoning answer (the bare-model conversational format the hidden prompts hit).

## 6. Near-duplicate detection + splits (`bursa_eval/dataset.py`)

**Single graph enforces all constraints.** Nodes = all cases (gold + synthetic). Edges connect any two cases sharing a `guardian_family`, a `template_family`, **or** a near-dup signature. Connected components are the atomic split unit — so no family and no near-dup pair can straddle a split (leakage impossible by construction).

**Near-dup signature:** `(scenario_family, sorted normalized-narration tokens, amount bucket, sorted normalized student names)`; near-dups (equal signature or narration-token Jaccard above a threshold) become edges.

**Split assignment (deterministic greedy, required):** components sorted by size descending; each assigned to the split furthest below its target ratio; seeded tiebreaks. Realized ratios (not expected) are what matter at n≈150. **Warn if any component exceeds ~10% of gold.** Structural rules:
- A component containing **any synthetic case → `train`** (`synthetic` is training-only).
- Gold-only components → `train / val / test` at **70/15/15** — the ratios (and the manifest's `realized_split`) are computed **over the gold set only**; synthetic is added to train on top and excluded from the ratio.
- **Test (frozen hidden) = gold-only components in test.**

**Coverage report (required):** per-split `scenario_family × language` matrix, warning on empty val/test cells and oversized components — so authoring can be steered before the freeze.

## 7. Assembly + manifest (`data/manifest.json`)

`dataset.build()` = assemble gold + synthetic → near-dup edges → components → greedy split → render (D6) → write `data/build/{train,val,test}.jsonl` (git-ignored; regenerable from seed + committed gold).

**Manifest** — the reproducibility record and freeze artifact:
- `base_seed`, `synth_config_hash` (**sha256 over canonical (sorted-key) JSON**, same stability class as the seed fix), counts, realized split ratios, coverage.
- **`test_case_ids` AND `val_case_ids` are pinned** (immovable). Val is pinned too because Phase-3 calibration tunes on it — a silent val→train migration would contaminate calibration.
- **Freeze trigger:** the manifest freezes **when gold authoring completes, immediately before Phase 3 training**. Pre-freeze re-splits are expected and normal (the split re-runs as gold accrues).
- **Post-freeze enforcement:** `split()` keeps pinned val/test IDs in place and **raises loudly** if any new edge links a pinned val/test case toward a train-bound component — the human then drops/quarantines the offending new case rather than silently corrupting a frozen set.

## 8. Dependencies & handoffs

- **General-enterprise 30% (D6):** the reconciliation 70% is produced here (both formats); the general 30% (summarization/drafting/analysis) is a **separate curated source consumed at training time (Agent T)**. It **must come from permissively-licensed sources (e.g. Apache-2.0 / MIT / CC-BY), with provenance recorded in the dataset report** — it ships in the public repo and Gate 2 audits provenance.
- **Eval harness (sub-project 2):** consumes this schema. Carry-over: **pool-recall is computed over non-exact cases only** (deterministic short-circuits build no pool); pool truth = `case.pool_truth()`.
- **Pidgin reviewer (D10):** a confirmed native speaker reviews Pidgin templates before mass generation.

## 9. Testing

- Schema/validator/scaffold — SHIPPED (`test_gold_models.py`, `test_goldcheck.py`, `test_goldset.py`).
- Generator: **two-subprocess byte-identical purity test**; validator-gate (every generated case valid); synthetic-near-dups-gold → dropped; abstention floor met.
- Renderers: `to_app_format` prompt equals the live `PromptBuilder` output for the materialized case (identity test); `duplicate_blocked` excluded; §6 target fields match the derivation table.
- Splits: no `guardian_family`/`template_family`/near-dup straddles a boundary; synthetic never in val/test; realized ratios within tolerance; oversized-component warning fires.
- Manifest: pinned val/test IDs immovable; a new edge toward a pinned case raises.

## 10. Acceptance criteria

`goldcheck` green in CI on all gold; generator pure (subprocess test) and validator-gated; renderers reuse the real Agent M path with the §6 derivation; splits provably leak-free with synthetic train-only and a gold-only frozen hidden set; manifest pins + enforces val/test; one command builds `train/val/test.jsonl` reproducibly from `base_seed` + committed gold; coverage report emitted. (The 150 gold cases and the general-30% source are tracked separately.)

## 11. Module map

```
bursa_eval/
  models.py loader.py goldcheck.py goldnew.py        # SHIPPED
  synth/
    __init__.py namepools.py templates.py perturb.py generate.py render.py
  dataset.py                                         # near-dup + split + assembly + manifest
data/gold/*.yaml                                     # team-authored (in parallel)
data/build/{train,val,test}.jsonl                    # git-ignored, regenerable
data/manifest.json                                   # reproducibility + freeze artifact
```
