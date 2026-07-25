# Bursa

**Offline AI reconciliation copilot for African school fees** — Africa Deep Tech Challenge 2026, Corporate/Enterprise track.

Bursa turns messy bank statements and payment evidence into an accurate, auditable student-fee ledger on an ordinary school laptop — fully offline. It pairs a locally-run language model (interprets ambiguous narrations, ranks candidate students) with deterministic matching and a financial constraint engine that owns all arithmetic. The model **never** writes to the ledger.

> This README tracks project status and is updated after each major milestone.

## Status

| Phase | Component | State |
|---|---|---|
| Phase 1 | **Agent F — Financial Core** (data model, imports, dedup, matcher, constraint engine INV-01..10, append-only ledger) | ✅ complete on `main` (checkpoint C1) |
| Phase 2A | **Agent M — Inference path** (candidate generator, prompt builder, GBNF grammar, schema validation, calibrator v1 — model never auto-posts) | ✅ complete on `main` |
| Phase 2B | **Agent D — Data & Evaluation** | 🚧 in progress on branch `phase-2b-agent-d` — gold-case schema, validator, and scaffold shipped; synthetic generator + eval harness next |
| — | Baselines / C2 pivot (1.7B vs 0.6B), Phase 3 fine-tune + calibrator training | ⏳ upcoming |

**Test suite:** 116 passing (1 skipped — a tokenizer-asset check that runs on i5-class hardware).

## Architecture

```
CSV / screenshot → Import + OCR → Normalize + dedupe → Deterministic matcher
  → unambiguous → Constraint engine
  → ambiguous  → Candidate generator (≤5) → local LLM (GBNF JSON) → Schema validation → Constraint engine
Constraint engine → Calibrated confidence → auto-match / review / unmatched
All approved postings → append-only ledger (source, evidence, actor, decision path)
```

Inviolable boundaries: OCR transcribes (never reconciles); the LLM ranks and explains (never posts); the constraint engine owns all money math; the bursar owns uncertainty; the bank statement is the source of truth for received money.

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
data/gold/        # team-authored gold cases (YAML, one per file)
docs/             # PRD.md, MODEL_ARCHITECTURE.md, PROJECT_CONTEXT.md
  superpowers/specs/, superpowers/plans/, superpowers/runbooks/
tests/            # pytest + hypothesis
BURSA_ADTC_EXECUTION_PLAN.md   # the locked work order (decisions, phases, checkpoints)
metadata.json, download_model.sh, REPORT.md                   # ADTC submission artifacts
```

## Setup & test

```bash
uv venv .venv
uv pip install --python .venv/bin/python "pydantic>=2.6" pytest hypothesis pyyaml
.venv/bin/pytest -q          # run the full suite
```

## Key commands

| Command | Purpose |
|---|---|
| `.venv/bin/pytest -q` | run the test suite |
| `python -m bursa_eval.goldcheck data/gold` | validate gold cases (schema + financial invariants) + coverage report |
| `python -m bursa_eval.goldnew --family sibling_split --lang en --difficulty hard` | scaffold a new gold case |
| `bash download_model.sh` | fetch the GGUF (zero-shot Qwen3-1.7B Q4_K_M for now) + tokenizer, TOFU checksums |

## Safety invariants (execution plan §3.1)

All money is **integer minor units** (no floats); allocations can't exceed the transaction; the ledger is **append-only** (reversals, never edits — enforced by DB triggers); the LLM never posts and, in v1, the model path never auto-posts (everything it touches → review); every posting records source, evidence, actor, timestamp, and decision path.

## Documentation

- Financial Core: [design](docs/superpowers/specs/2026-07-24-bursa-financial-core-design.md) · [plan](docs/superpowers/plans/2026-07-24-bursa-financial-core.md)
- Agent M inference path: [design](docs/superpowers/specs/2026-07-25-agent-m-inference-design.md) · [plan](docs/superpowers/plans/2026-07-25-agent-m-inference.md) · [verification runbook](docs/superpowers/runbooks/agent-m-verification.md)
- Product baselines: [PRD](docs/PRD.md) · [Model architecture](docs/MODEL_ARCHITECTURE.md) · [Project context](docs/PROJECT_CONTEXT.md)
