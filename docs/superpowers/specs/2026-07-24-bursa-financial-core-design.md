# Bursa Financial Core — Design Spec

**Phase:** 1 (Agent F — Financial Core)
**Date:** 2026-07-24
**Status:** Approved design (brainstorming complete); implementation plan to follow
**Authoritative work order:** `BURSA_ADTC_EXECUTION_PLAN.md` §4 (Agent F), §3.1 (invariants), §6 (output contract), D7 (stack), D12 (money)
**Reference baselines:** `docs/PRD.md` (FR-001..015, §11), `docs/MODEL_ARCHITECTURE.md` (§5–§7), `docs/PROJECT_CONTEXT.md` (§8–§9)

---

## 0. Status & gate

- **§3.1 reconciliation gate: PASSED** (no blocking conflict; Hard Rule #1 HALT not triggered). §3.1 is safe to implement against as-is. Provenance wording notes were applied to the plan by the human; invariants are near-verbatim from MODEL_ARCHITECTURE §7, BR-01..04 condense PRD §11, BR-05 sources from PRD §14 / PROJECT_CONTEXT §9.
- **Phase 1 scope (locked):** **backend financial core + test suite only.** No web UI, no LLM, no OCR, no confidence calibrator this phase — those are Phase 2/3. This matches execution-plan §4 Agent F and C1 ("financial core tests green").

## 1. Scope & non-goals

### In scope (Phase 1)
Canonical data model; CSV importers (students/guardians/siblings, fee schedule, bank statement) with column mapping and row-level errors; idempotent re-import; duplicate detection; deterministic matcher with reason codes; constraint engine enforcing INV-01..INV-10; append-only ledger with reversals and credit; derived projections (balances, fee status, dashboard totals); the pytest + hypothesis suite that C1 gates on.

### Deferred (later phases; seams built now, implementations later)
- **LLM reconciliation path** (Phase 2): the "ambiguous → candidate generator → LLM → schema validation" branch. Phase 1 **stubs it to route ambiguous cases to `review`** as data. Nothing fabricates a match.
- **Confidence calibrator** (Phase 3, D11): Phase 1 ships a thin rule-based `ConfidencePolicy` behind the interface the logistic calibrator will later replace.
- **OCR** (Phase 3, D8), **web UI / HTMX** (D7, later), **natural-language reporting** (D9, CUT).

### Not building (execution-plan §11)
Payment APIs, virtual accounts, refunds, parent portal, SMS/email, natural-language reporting, unrestricted SQL, handwriting OCR, cloud anything, general chatbot, full ERP.

## 2. Architecture

### 2.1 Module map (`bursa/` core, `tests/` alongside)

```
bursa/
  money.py          # SOLE money converter, both directions: parse_naira(str)->int minor; format_naira(int)->"₦…". Decimal only at this boundary; never stored/computed as float.
  models.py         # Pydantic v2 domain types (Student, Guardian, Term, FeeItem, Charge, Transaction, Proposal, LedgerEvent, …)
  db.py             # sqlite3 connection factory: WAL + PRAGMA foreign_keys=ON on EVERY connection; schema DDL; triggers; transaction() helper (BEGIN IMMEDIATE)
  repository.py     # Typed data access — parameterized SQL only; no business logic
  importers/
    students.py     # student/guardian/sibling CSV -> validated rows -> repo
    fees.py         # fee schedule CSV -> fee_items + charges
    statement.py    # bank CSV -> canonical transactions (column mapping, dedup, idempotent)
  normalize.py      # pure name/narration normalization, alias handling, reference canonicalization
  reasoncodes.py    # canonical reason-code enum (shared by matcher; later the LLM path)
  matcher.py        # deterministic rules -> proposals + reason_codes (or ambiguous->review / none->unmatched)
  distribute.py     # student+amount proposal line -> charge-level allocation events (documented deterministic policy)
  confidence.py     # ConfidencePolicy interface + Phase-1 rule-based impl (Phase-3 calibrator swaps in here)
  constraints.py    # the constraint engine: INV-01..10 evaluated against cumulative net-of-reversals state
  ledger.py         # SOLE ledger writer: atomic post(), reverse(), credit ops; refuses anything the engine hasn't passed
  projections.py    # balances, fee status, credit, unapplied, dashboard totals — all derived from ledger_events
  pipeline.py       # orchestrates import -> normalize -> match -> distribute -> validate -> post/route
  config.py         # runtime config (e.g. auto_post_enabled: bool = True)
  errors.py         # typed errors (ImportRowError, InvariantViolation, …)
tests/              # pytest + hypothesis; INV matrix, BR tests, property/stateful tests, structural/concurrency tests
```

### 2.2 Tech choices (inside locked D7 stack)

| Choice | Decision | Why |
|---|---|---|
| DB access | stdlib `sqlite3` + thin `repository.py`, WAL | zero extra deps, lowest RAM, full control over the atomic posting transaction; parameterized queries native |
| Validation | Pydantic v2 for import rows & DTOs | row-level field/reason errors (FR-001/002); clean boundary types |
| Money | `int` minor units everywhere; `money.py` sole converter | INV-03 / D12; Decimal only at parse boundary, never stored |
| Tests | pytest + hypothesis (incl. `RuleBasedStateMachine`) | §4 mandates property tests on allocation sums; stateful testing over real operations |

### 2.3 Inviolable boundaries (structural, not conventional)
1. **`constraints.validate()` is called on every path before any post** and is the only money/ID validator.
2. **`ledger.py` is the only module that appends `ledger_events`**, and it refuses anything the engine hasn't passed.
3. **`money.py` is the only module that parses or formats money.**
4. **`ledger_events` is physically append-only** (DB triggers, §4.4).

## 3. Data model

Approach: **charge-level ledger** (allocations target a `(student, charge)` pair). This satisfies INV-04 ("student and fee IDs") and FR-010 (credit applied to a specific fee item) literally. The LLM/deterministic path proposes `{student_id, amount}` (matching the §6 contract, which carries **no** `fee_id` and must not be extended); the app's `distribute.py` maps that to charge-level allocation events. Fee assignment is money math and therefore lives on the deterministic side of the boundary.

### 3.1 Tables

**Setup (imported, read-mostly):**

| Table | Purpose | Key columns |
|---|---|---|
| `terms` | academic session + term | `term_id`, `session`, `term_name`, `is_active` |
| `students` | roster | `student_id` (PK), `name`, `normalized_name`, `class`, `term_id` |
| `guardians` | payers | `guardian_id`, `name`, `normalized_name`, `phone_suffix?` |
| `student_guardians` | M:N; **siblings = shared guardian** | `student_id`, `guardian_id` |
| `fee_items` | billable catalog | `fee_id` (PK), `name`, `term_id`, `priority` |
| `charges` | per-student billed line *identity* (student × fee × term); **the billed amount is a `charge_created` ledger event, not a column here** | `charge_id` (PK), `student_id`, `fee_id`, `term_id` |

**Ingestion:**

| Table | Purpose | Key columns |
|---|---|---|
| `import_batches` | idempotency + import summary | `batch_id`, `source_file`, `imported_at`, `accepted`, `rejected`, `duplicate` |
| `transactions` | canonical bank credits | `transaction_id` (PK), `source`, `reference?`, `raw_reference?`, `posted_at`, `payer_name?`, `narration?`, `amount_minor`, `direction`, `dedup_hash`, `batch_id`, `routing_state` |

**Reconciliation:**

| Table | Purpose | Key columns |
|---|---|---|
| `proposals` | a reconciliation attempt (retained for audit, FR-007) | `proposal_id`, `transaction_id`, `source` (`deterministic`/`llm`), `recommended_action` (`auto`/`review`/`unmatched`), `confidence`, `explanation`, `status`, `created_at` |
| `proposal_allocations` | student-level proposal lines (pre-distribution, matches §6) | `proposal_id`, `student_id`, `amount_minor`, `reason_codes` |

**Ledger (single source of truth):**

| Table | Purpose | Key columns |
|---|---|---|
| `ledger_events` | **append-only**, only mutable financial record | `event_id` (monotonic PK), `event_type` (`charge_created`/`allocation`/`reversal`/`credit_grant`/`credit_application`), `transaction_id?`, `charge_id?`, `holder?` (student/guardian, for credit), `amount_minor`, `actor`, `source`, `evidence_ref`, `decision_path`, `reverses_event_id?`, `created_at` |

`amount_minor`, `actor`, `source`, `evidence_ref`, `decision_path` are **NOT NULL** (INV-09). FKs from `charge_id`→`charges`, `student_id`→`students`, `fee_id`→`fee_items`, `transaction_id`→`transactions` (INV-04). For setup events (`charge_created`), `source`/`evidence_ref` reference the originating import batch and `transaction_id` is null; for payment events (`allocation`, `credit_grant`, `credit_application`) they reference the transaction.

### 3.2 Allocations are events, not a side table
An allocation is a `ledger_events` row of type `allocation` linking `transaction → charge`. There is no separate mutable allocations table to drift. Likewise the **billed amount of each charge is a `charge_created` event** (the `charges` table holds only identity), so the debit side of every balance also lives in the event log. Balances, fee status, credit, and unapplied are computed from events (§5.5) — so "balances reconstruct purely from ledger events" (§4 acceptance) is true by construction, with `charges`/`students`/`fee_items` supplying identity only.

### 3.3 Charge-distribution policy (`distribute.py`) — documented & deterministic
Input: a proposal line `{student_id, amount_minor}`. Output: charge-level allocation events + any remainder.
1. Select the student's charges with remaining balance > 0.
2. Order deterministically: **(a) `fee_items.priority`** (imported, data-driven), then **(b) term due order**, then **(c) `charge_id` ascending** (final tiebreak).
3. Fill each charge up to its remaining balance until the line amount is exhausted.
4. **Surplus after all the student's charges are cleared → `credit_grant` for that student** (INV-05/07).
5. **Money on the transaction assigned to no student → `unapplied` remainder** (INV-06).
The default is deterministic and testable; a **bursar override at review** may re-distribute across charges, convert a remainder to guardian credit (FR-009), or reject — each recorded as its own event.

### 3.4 Unapplied is first-class (INV-06 / BR-02)
Every transaction the pipeline touches yields an explicit `unapplied_minor = amount − Σ allocations − Σ credit_grants(from txn)`, always surfaced, never hidden, recomputed on read (never a stored field).

### 3.5 Queues are data, not logic
There is no queue engine. "Review" and "unmatched" queues are **filtered reads** over `transactions.routing_state` (`SELECT … WHERE routing_state = 'review'`). Routing is a value set by the matcher + confidence policy + constraint engine; the queues are views over that value.

## 4. Ingestion pipeline

### 4.1 Import & idempotency
Each importer: validate row (Pydantic) → collect row-level `ImportRowError(row_number, field, reason)` without aborting the file → write accepted rows under an `import_batch` recording `accepted/rejected/duplicate` counts (FR-001/002). `money.py` parses naira cells to `int` minor units at the boundary. **The fee importer writes each `charges` identity row and its originating `charge_created` ledger event in one atomic transaction** (`BEGIN IMMEDIATE` … `COMMIT`) — a charge can never exist without its billing event (tested in §6.4).

### 4.2 Normalization (`normalize.py`)
Pure functions: `normalize_name` (casefold, strip titles/punctuation, collapse whitespace, alias expansion); `canonicalize_reference` (uppercase, strip separators); narration tokenization. Raw values are always preserved alongside normalized.

### 4.3 Duplicate handling — two distinct concerns, kept separate

| Concern | Mechanism | Enforced at |
|---|---|---|
| Idempotent re-import (same statement twice = no-op) | `dedup_hash` **UNIQUE** | schema |
| A reference maps to at most one row | partial index `UNIQUE(reference) WHERE reference IS NOT NULL` | schema |
| Possible duplicate, no exact reference (FR-011) | content-similarity check → `routing_state='review'` | matcher (not schema) |

`dedup_hash` is defined to **collide only on genuine re-imports**:
- With a reference present: `dedup_hash = hash(reference)` → true duplicates collide; re-import is idempotent.
- With no reference: `dedup_hash = hash(source_file, canonical_row, occurrence_index_within_file)` → re-importing the *same file* collides (idempotent); a genuinely separate payment in a *different* file does not.
- Separately, a new reference-less transaction whose `(posted_at, amount_minor, normalized_payer, normalized_narration)` matches an existing transaction is **inserted and routed to review** as a possible duplicate — never rejected. **`reference IS NULL` is legitimate and never blocks import.**

### 4.4 Structural enforcement (immutability is physical)
```sql
CREATE TRIGGER ledger_events_no_update BEFORE UPDATE ON ledger_events
BEGIN SELECT RAISE(ABORT, 'ledger_events is append-only: UPDATE forbidden'); END;
CREATE TRIGGER ledger_events_no_delete BEFORE DELETE ON ledger_events
BEGIN SELECT RAISE(ABORT, 'ledger_events is append-only: DELETE forbidden'); END;
```
Inserts (including reversal events) pass; any UPDATE/DELETE aborts. Deliberate schema migrations drop/recreate triggers explicitly — the only sanctioned path (documented). `db.py` sets `PRAGMA journal_mode=WAL` **and** `PRAGMA foreign_keys=ON` on every connection (SQLite defaults FKs off per-connection; without this INV-04's FK layer is decorative).

### 4.5 Deterministic matcher (`matcher.py`) + reason codes
Resolves safe cases before any model (MODEL_ARCH §5.3), emitting a `proposal` with `reason_codes` from the `reasoncodes.py` enum:

| Rule | Reason code(s) | Routes to |
|---|---|---|
| Exact `student_id` in narration | `EXACT_STUDENT_ID` | auto |
| Single guardian → single student, amount = one exact outstanding charge | `SINGLE_GUARDIAN_STUDENT` + `EXACT_OUTSTANDING_BALANCE` | auto |
| Known prior payer→student mapping **corroborated by** exact outstanding-balance match | `KNOWN_PAYER_MAPPING` + `EXACT_OUTSTANDING_BALANCE` | auto |
| Known prior payer→student mapping **alone** (no exact-balance corroboration) | `KNOWN_PAYER_MAPPING` | **review** |
| Duplicate reference | `DUPLICATE_REFERENCE` | blocked (idempotent) |
| Multiple plausible candidates / fuzzy only | `AMBIGUOUS_CANDIDATES` | **review** (Phase-2 LLM path stub) |
| No candidate | `NO_CANDIDATE` | **unmatched** |

**A prior payer mapping is a hint, not exact evidence** — it auto-posts only with exact-balance corroboration.

### 4.6 Confidence & auto-post
Phase-1 `ConfidencePolicy` is rule-based: deterministic exact rules → high (auto-eligible); ambiguous → review; none → unmatched. The **D11 logistic calibrator (Phase 3) swaps in behind the same interface.**

**Auto-posting sits behind one config flag** `auto_post_enabled` (default **on**) — the concrete realization of BR-04's "subject to configured policy." Flag off ⇒ exact matches become auto-eligible but route to review instead of posting. **Auto-post is not a bypass**: it calls the same atomic `post()` and is still fully validated by the engine (§5). Only exact deterministic rules ever reach `auto`; an "auto" proposal the engine rejects drops to review (INV-10). This is how "zero false auto-posts" (D11) and BR-05 hold by construction.

## 5. Constraint engine, atomic posting, reversals & projections

### 5.1 The engine is stateful and net of reversals
`constraints.validate(proposal, cumulative_state) → ValidationResult(ok, violations[])`. It computes the **resulting cumulative state** — all prior posted events for the affected transaction/charges/holder **plus** the proposed new events — and checks every invariant against that result. Never judges a proposal in isolation.

**Unifying rule: a reversal is the negation of its target; every cumulative check and projection is a signed sum over `ledger_events`.** "Net of reversals" is therefore automatic, not a special case. Three aggregates:

| Aggregate | Signed sum (net of reversals) | Bound |
|---|---|---|
| Transaction capacity used (INV-01) | Σ `allocation(txn)` + Σ `credit_grant(txn)` | ≤ `txn.amount_minor` |
| Charge balance (INV-05) | Σ `charge_created(charge)` − [Σ `allocation→charge` + Σ `credit_application→charge`] (all net of reversals) | ≥ 0 |
| Holder credit (INV-05/07 + credit sufficiency) | Σ `credit_grant(holder)` − Σ `credit_application(holder)` | ≥ 0 |

**Overpayment credit is funded by the transaction**, so `credit_grant(txn)` counts against the transaction's INV-01 capacity — required for conservation (allocations + credit + unapplied = amount).

### 5.2 INV enforcement map (spine of the C1 suite)

| INV | Concrete check | Enforced at | On fail |
|---|---|---|---|
| 01 | Σ(posted allocations for txn, net) + Σ(proposed) ≤ `txn.amount` | engine (cumulative) | → review |
| 02 | txn reference/dedup identity hasn't already produced this posting | schema `UNIQUE(ref)`,`UNIQUE(dedup_hash)` + engine | → review / blocked |
| 03 | amounts are `int` minor units; no float anywhere | types + `money.py` + engine assert | reject |
| 04 | every `student_id`/`charge_id`/`fee_id` ∈ imported set | FKs (`foreign_keys=ON`) + engine membership | → review |
| 05 | no charge balance < 0; student surplus emitted as credit | engine | → review |
| 06 | `unapplied = amount − Σalloc − Σcredit(txn)` always computed & visible | projection | n/a (visible) |
| 07 | overpayment beyond all student charges → `credit_grant`, never dropped | engine + distribute | → review |
| 08 | reversals are new compensating events; no edit/delete; **no reversal-of-a-reversal** | triggers + `ledger.reverse()` guard | ABORT / reject |
| 09 | event carries source, evidence, actor, timestamp, decision_path | `NOT NULL` cols + engine | reject |
| 10 | any failed invariant prevents posting, routes to review | engine top-level = atomic rollback | → review |

### 5.3 Atomic posting (`ledger.py`, sole writer)
```
post(proposal, actor, source, evidence_ref, decision_path):
    with db.transaction():                       # BEGIN IMMEDIATE
        state  = load_cumulative_state(txn_id, affected_charges, holder)   # signed, net of reversals
        result = constraints.validate(proposal, state)
        if not result.ok:
            raise InvariantViolation(result.violations)   # rollback -> routing_state='review' (INV-10)
        for ev in proposal.to_events():          # allocation(s) + credit_grant; unapplied is derived
            repo.insert_ledger_event(ev, actor, source, evidence_ref, decision_path)
        # COMMIT
```
All-or-nothing. On `InvariantViolation` the caller sets `routing_state='review'`, records the failed proposal (audit), never posts. `BEGIN IMMEDIATE` serializes concurrent posts to the same transaction so the cumulative INV-01 check is race-safe.

### 5.4 Reversals & credit — all as events, through the same gate
- **Reversal / correction:** `ledger.reverse(event_id, actor, reason)` inserts a compensating event referencing `reverses_event_id`; the original stays forever (INV-08, FR-007). Accept/edit/split/reject at review = reversal + replacement allocation. **A reversal event can never be the target of another reversal** — `ledger.reverse()` rejects it; correcting a mistaken reversal is a fresh allocation through `post()`. (Protects the signed-sum model from double-negation.)
- **Credit:** `credit_grant` (overpayment surplus, or a bursar-directed transaction remainder — both **funded by the transaction**, so `transaction_id` is set and the grant counts against that transaction's INV-01 capacity, keeping conservation sound), `credit_application` (apply existing credit to another approved fee item, FR-010). **`credit_application` is posted via the same atomic `post()` + engine**, with the cumulative sufficiency check **Σ applications ≤ Σ grants per holder (net of reversals)** and the charge-balance ceiling. Credit is not a side door around the invariants.

### 5.5 Projections (`projections.py`) — read-side, derived from events
Fee status (BR-03/FR-008) is computed, never stored: `outstanding` (paid 0) · `part_paid` (0<paid<billed) · `cleared` (paid≥billed) · `credit` (credit>0). Charge/student balances, transaction unapplied, and FR-013 dashboard totals (billed, received, allocated, outstanding, credited, unapplied) all reconstruct purely from `ledger_events`.

## 6. Testing strategy (what makes C1 objective)

**Framework & fixtures:** pytest + hypothesis. A fixture builds a temp-file SQLite with the full schema, **triggers installed, `foreign_keys=ON`, WAL** — tests run against the real structural guarantees.

### 6.1 INV matrix — one posting-blocked test per invariant, each with a passing twin
For each INV-01..10: a proposal engineered to violate it asserts `post()` raises `InvariantViolation`, **no `ledger_event` is written**, and `routing_state` becomes `review` (INV-10). The passing twin proves the valid case still posts (guards against an over-strict check that blocks everything). INV-08 test asserts both append-only triggers fire on UPDATE and DELETE.

### 6.2 BR behavior tests
BR-01 (credit=money vs image=evidence) · BR-02 (0/1/many allocations, partial unapplied) · BR-03 (status derived; no status field exists to mutate) · BR-04 (auto=reversible proposal; flag off ⇒ review; med/low never post) · BR-05 (simulated matcher/OCR failure posts nothing).

### 6.3 Property & stateful tests (hypothesis)
- **Conservation:** `Σ allocations + Σ credit_grants + unapplied == txn.amount` exactly (integer).
- **No charge overfill:** `Σ (allocations + credit_applications) → charge ≤ Σ charge_created(charge)` (billed amount as a projection, net of reversals).
- **Credit non-negativity:** random grant/application/reversal sequences never drive holder credit < 0.
- **Reconstruction:** balances/status from events == running projection.
- **Re-import idempotency:** importing any generated batch twice ⇒ identical state / zero new transactions.
- **Stateful machine:** a `RuleBasedStateMachine` whose rules are the real operations (import, post, reverse, credit-grant, credit-apply, flag-flip), asserting **conservation, charge non-negativity, credit non-negativity** as machine invariants after every step.

### 6.4 Closure, dedup, structural, security & concurrency tests
- **Reversal/credit closures:** post→reverse→repost same amount **succeeds**; post→reverse→over-post **fails**; reversal-of-a-reversal **blocked**; credit sufficiency (apply ≤ grants passes, apply > grants net fails); credit routed through `post()`.
- **Dedup:** same statement twice → 0 new txns; reference-less rows never blocked; near-duplicate (no ref) → review.
- **Structural:** both append-only triggers fire; FK enforcement fires (bad `student_id`/`charge_id`/`fee_id` rejected); `money.py` rejects floats and round-trips `parse_naira ↔ format_naira`.
- **Charge/billing atomicity:** the fee importer writes each `charges` row and its `charge_created` event in one atomic transaction; a test asserts **no charge exists without its billing event**, and that a failed fee import rolls back both together.
- **Untrusted narration (Prime directive 6):** a narration naming a non-imported student ID never yields an allocation to it (matcher treats narration as data; INV-04 backstop).
- **Concurrency:** two threads posting a combined over-capacity amount to one transaction under `BEGIN IMMEDIATE` — exactly one commits, the other routes to review.

### 6.5 Acceptance → test mapping (C1 is checkable, not vibes)
| §4 Agent F acceptance clause | Proving test(s) |
|---|---|
| re-import same statement twice → zero new transactions | 6.3 idempotency, 6.4 dedup |
| every INV-01..10 has a failing test that blocks posting | 6.1 INV matrix |
| BR-01..05 each have behavior tests | 6.2 |
| balances reconstruct purely from ledger events | 6.3 reconstruction |
| property-based tests pass on allocation sums | 6.3 conservation / overfill / stateful machine |

## 7. Acceptance criteria (Phase 1 / C1)
All of: §6.1 INV matrix green (block + twin), §6.2 BR tests green, §6.3 property + stateful machine green, §6.4 closure/dedup/structural/security/concurrency green, §6.5 mapping complete. Plus: full test suite runs offline; no floats anywhere near money; `ledger.py` is the only ledger writer; append-only triggers and `foreign_keys=ON` verified by test.

## 8. Open items / deferred
- LLM reconciliation path, GBNF grammar, prompt builder → Phase 2 (Agent M). Phase-1 ambiguous cases route to review.
- Confidence calibrator (D11) → Phase 3; `ConfidencePolicy` seam built now.
- OCR (D8), web UI (D7), evidence table population → later phases; `evidence_ref` on ledger events references the source transaction in Phase 1.
- Fee-item `priority` default ordering is data-driven from `fee_items.priority` (imported); confirm the seed ordering with domain input when demo fixtures are authored.
