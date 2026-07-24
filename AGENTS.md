# Bursa — Agent Instructions

Read `BURSA_ADTC_EXECUTION_PLAN.md` in full before doing anything. It is the
authoritative work order: locked decisions, work packages, phases,
checkpoints, and tripwires. Decisions are final unless their pre-committed
pivot trigger fires.

Reference documents live in `docs/`:

- `docs/PROJECT_CONTEXT.md` — shared product context and why the product pivoted
- `docs/PRD.md` — full product requirements; FR-001 to FR-015 are the
  acceptance criteria for imports, reconciliation, review, ledger, and reports
- `docs/MODEL_ARCHITECTURE.md` — approved model and pipeline design baseline

Hard rules:

1. The financial invariants in plan §3.1 (INV-01..INV-10, BR-01..BR-05) are
   law. If §3.1 ever conflicts with `docs/`, stop and flag both versions for
   human sign-off. Never silently pick one.
2. The LLM never writes to the ledger. All money math uses integer minor
   units. No floats near money.
3. Nothing that can crash or OOM ships. An OOM during evaluation is
   disqualification.
4. Every model change is evaluated on the Bursa gold set, the ADTC validation
   set, and the bare-model prompt suite before merge.
5. Benchmarks count only when measured on i5-class hardware at 4 threads.
