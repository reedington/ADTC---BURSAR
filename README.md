# Bursa

**Offline AI reconciliation copilot for African school fees** — Africa Deep Tech Challenge 2026, Corporate/Enterprise track.

Bursa turns messy bank statements and payment evidence into an accurate, auditable student-fee ledger on an ordinary school laptop — fully offline. It pairs a locally-run language model (interprets ambiguous narrations, ranks candidate students) with deterministic matching and a financial constraint engine that owns all arithmetic. The model **never** writes to the ledger.

> This README tracks project status and is updated after each major milestone.

## Status

| Phase | Component | State |
|---|---|---|
| Phase 1 | **Financial Core** (data model, imports, dedup, matcher, INV-01..10, append-only ledger) | ✅ code complete; C1 hardware prerequisites still pending |
| Phase 2A | **Inference path** (candidate generator, prompt, GBNF, validation, fail-safe routing) | ✅ code complete; real-GGUF 1,000-run acceptance pending |
| Phase 2B | **Data & Evaluation** | ✅ harness complete; 14/14 authored scenario families, English + Pidgin, both hard gates exercised in CI |
| Demo | **Offline local web workflow** | ✅ setup imports, bank mapping, queues, review/edit/split/reject, explicit credit, audit and reversals; visually verified |
| — | Baselines / C2 pivot (1.7B vs 0.6B), Phase 3 fine-tune + calibrator training | ⏳ upcoming |

**Test suite:** 193 passing, 1 skipped (the tokenizer/model-bound check runs during the target
hardware session). Passing fake-backend tests validate mechanics and safety seams, not model
quality or performance.

## Architecture

```
CSV → Normalize + dedupe → Deterministic matcher
  → unambiguous → Constraint engine
  → ambiguous  → Candidate generator (≤5) → local LLM (GBNF JSON) → Schema validation → Constraint engine
Constraint engine → Calibrated confidence → auto-match / review / unmatched
All approved postings → append-only ledger (source, evidence, actor, decision path)
```

Inviolable boundaries: OCR transcribes (never reconciles); the LLM ranks and explains (never posts); the constraint engine owns all money math; the bursar owns uncertainty; the bank statement is the source of truth for received money.

OCR is deliberately **not included in the current Gate 1 build**. It will ship only if the
single printed-transfer workflow passes the plan's reliability review; otherwise it remains
future work.

## Repository layout

```
bursa/            # the application
  money.py, models.py, db.py, repository.py, projections.py   # foundation
  constraints.py, ledger.py, distribute.py                    # money core (sole ledger writer)
  importers/, normalize.py, matcher.py, candidates.py, pipeline.py
  inference/      # llama-server backend, GBNF grammar, prompt builder, token budget, validation
  features.py, calibrator.py                                  # confidence seam (v1: model → review)
bursa_eval/       # evaluation & data tooling (Agent D)
  models.py, loader.py, goldcheck.py, goldnew.py              # gold-case schema, validator, scaffold
  synth/          # deterministic synthetic generator (templates, perturbation, D6 renderers)
  harness/        # gold, ADTC/proxy, bare-model suites + scorecard
  dataset.py      # near-dup + leak-free splits, assembly, freeze manifest
data/gold/        # team-authored gold cases (YAML, one per file)
data/build/       # generated {train,val,test}.jsonl (git-ignored, regenerable from seed + gold)
docs/             # PRD.md, MODEL_ARCHITECTURE.md, PROJECT_CONTEXT.md
  superpowers/specs/, superpowers/plans/, superpowers/runbooks/
tests/            # pytest + hypothesis
demo/             # fictional CSV fixtures
media/            # verified application screenshots
BURSA_ADTC_EXECUTION_PLAN.md   # the locked work order (decisions, phases, checkpoints)
metadata.json, download_model.sh, REPORT.md                   # ADTC submission artifacts
```

## Setup & test

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q          # run the full suite
```

## Run the offline demonstration

```bash
.venv/bin/bursa-web
# open http://127.0.0.1:8000
```

The app seeds fictional records on first launch. CSV fixtures are also available in `demo/`.
The interface binds to loopback only, uses a local SQLite database, and has no analytics or
external telemetry.

To enable local-model review, point Bursa at the pinned files:

```bash
export BURSA_MODEL_PATH=model/qwen3-1.7b-q4_k_m.gguf
export BURSA_TOKENIZER_PATH=model/tokenizer.json
export BURSA_ACTOR=bursar
.venv/bin/bursa-web
```

If the model is missing or unhealthy, Bursa continues serving imports, deterministic exact
matches, review evidence, and the ledger. Candidate-bearing ambiguous payments route to
`INFERENCE_UNAVAILABLE` review; only genuine no-candidate payments become unmatched.

## Key commands

| Command | Purpose |
|---|---|
| `.venv/bin/pytest -q` | run the test suite |
| `python -m bursa_eval.goldcheck data/gold` | validate gold cases (schema + financial invariants) + coverage report |
| `python -m bursa_eval.goldnew --family sibling_split --lang en --difficulty hard` | scaffold a new gold case |
| `bash download_model.sh` | fetch the revision- and checksum-pinned zero-shot Qwen3-1.7B Q4_K_M + tokenizer |
| `bash download_model.sh --baselines` | additionally fetch the Qwen3-0.6B C2 comparison model |
| `python -m bursa_eval.harness.scorecard --backend fake` | run the eval harness offline; emits the scorecard + per-case records (exits non-zero if a safety gate fails) |
| `bash scripts/run_i5_checkpoint.sh` | on the physical Intel i5 only: run both real scorecards, 1,000-input safety gates, llama-bench, and ten profiler passes |
| `bursa-demo-case data/gold/<case>.yaml --output data/local/<demo>.db` | materialize one fictional authored scenario as a persistent browser-test database |
| `python -m bursa_eval.submission_check --gate1` | fail fast on missing identity, profiler, report, prompt, license, screenshot, or video artifacts |

## Safety invariants (execution plan §3.1)

All money is **integer minor units** (no floats); allocations can't exceed the transaction; the ledger is **append-only** (reversals, never edits — enforced by DB triggers); the LLM never posts and, in v1, the model path never auto-posts (everything it touches → review); every posting records source, evidence, actor, timestamp, and decision path.

## Documentation

- Financial Core: [design](docs/superpowers/specs/2026-07-24-bursa-financial-core-design.md) · [plan](docs/superpowers/plans/2026-07-24-bursa-financial-core.md)
- Agent M inference path: [design](docs/superpowers/specs/2026-07-25-agent-m-inference-design.md) · [plan](docs/superpowers/plans/2026-07-25-agent-m-inference.md) · [verification runbook](docs/superpowers/runbooks/agent-m-verification.md)
- Agent D eval harness: [design](docs/superpowers/specs/2026-07-25-agent-d-eval-harness-design.md) · [plan](docs/superpowers/plans/2026-07-25-agent-d-eval-harness.md) · [i5 runbook](docs/superpowers/runbooks/eval-harness-i5.md)
- Submission: [technical report](REPORT.md) · [demo script](docs/DEMO_SCRIPT.md) · [privacy](PRIVACY.md) · [security](SECURITY.md)
- Product baselines: [PRD](docs/PRD.md) · [Model architecture](docs/MODEL_ARCHITECTURE.md) · [Project context](docs/PROJECT_CONTEXT.md)
