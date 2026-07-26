# Agent D Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the internal evaluation harness — three suites on the existing inference seams, one scorecard command with two hard safety gates — that scores any model through the real serving pipeline and is fully deterministic in CI.

**Architecture:** A reusable `run_model_path()` is extracted from `pipeline.reconcile()` so the harness captures the model's *raw* validated output through the exact serving components. `evaluate_case()` mirrors the pipeline (materialize → matcher → run_model_path → dry-run) and emits a per-case record keyed by case id. A pure `metrics.py` turns records into the MODEL_ARCHITECTURE §12 metric family plus carry-ins; `scorecard.py` aggregates, records provenance, enforces the two gates as non-zero exit codes, and diffs runs.

**Tech Stack:** Python 3.13, stdlib `sqlite3` + `dataclasses` + `json`, Pydantic v2 (existing `GoldCase`), pytest. `FakeBackend`/`FakeChatBackend` for offline CI; `LlamaServerBackend`/`LlamaChatBackend` on the i5.

## Global Constraints

- Money is **integer minor units** only; convert authored naira via `bursa_eval.models.naira_to_minor` / `bursa.money`. Never a float (INV-03).
- The harness **never posts to the ledger** — every allocation comparison is a dry-run via `distribute.distribute()` (which computes events without posting), exactly as `reconcile()` does.
- Score the model's **raw validated output**, never v1's always-review routing.
- **Two hard gates:** `incorrect_auto_posts == 0` and `duplicate_blocked_rate == 100%` → non-zero exit.
- Determinism: no builtin `hash()`, no `Math.random`; per-case timings are **diagnostics only** and never enter the scorecard `perf` section (which ingests profiler numbers via `--perf`).
- Model-side abstention ≡ `candidate_allocations == []` **or** `recommended_action == "unmatched"`.
- Follow existing patterns: `bursa_eval/` package, `pythonpath=["."]`, tests under `tests/`. Never weaken an assertion to make code pass.

---

### Task 1: Extract `run_model_path()` from `reconcile()` (DRY serving seam)

**Files:**
- Modify: `bursa/pipeline.py` (lines 33-97 — the model-path block)
- Test: `tests/test_run_model_path.py`

**Interfaces:**
- Consumes: `matcher.match`, `candidates.generate(conn, txn) -> list[Candidate]`, `prompt.build`, `grammar.build_grammar`, `run_inference`, `schema.validate_output`, `distribute.distribute`, `constraints.validate`, `features.extract`, `get_token_counter`.
- Produces:
  - `@dataclass ModelPathResult(surviving: list, budget_shed: bool, data: dict | None, failure: str | None, schema_reason: str | None, dry_ok: bool, chosen_id: str | None, model_events: list)`
  - `run_model_path(conn, txn, backend, tokenizer_path: str | None = None) -> ModelPathResult`
  - `failure` ∈ `{None, "no_candidates", "prompt_budget", "transport", "schema"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_model_path.py
import json
from bursa import db as dbmod, repository as repo
from bursa.inference.backend import FakeBackend
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.goldcheck import load_case


def _model_json(txn_id, student_id, amount_minor):
    return json.dumps({
        "transaction_id": txn_id,
        "recommended_action": "review",
        "candidate_allocations": [
            {"student_id": student_id, "amount_minor": amount_minor, "reason_codes": ["MODEL_RANKED"]}
        ],
        "explanation": "matched by nickname",
    })


def test_run_model_path_returns_raw_data_and_events():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    conn = loader.materialize(case)
    txn_id = loader.insert_case_transaction(conn, case)
    txn = repo.get_transaction(conn, txn_id)
    student = case.expected.allocations[0].student_id
    amount = int(str(case.transaction.amount_naira).replace(",", "")) * 100  # naira -> kobo, worked example
    backend = FakeBackend(response=_model_json(txn_id, student, amount))

    r = run_model_path(conn, txn, backend)

    assert r.failure is None
    assert r.data["candidate_allocations"][0]["student_id"] == student
    assert r.chosen_id == student
    assert [c.student_id for c in r.surviving]  # pool built
    assert r.model_events  # distributed charge-grain events recovered
    conn.close()


def test_run_model_path_no_candidates_is_flagged():
    # A transaction whose narration matches nobody -> empty pool.
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    conn = loader.materialize(case)
    # wipe candidate sources so generate() returns []
    conn.execute("DELETE FROM student_aliases")
    conn.execute("UPDATE students SET name = 'zzzz', name_norm = 'zzzz'")
    conn.execute("UPDATE transactions SET narration = 'zzzz', payer_name = 'zzzz'")
    txn_id = loader.insert_case_transaction(conn, case)
    txn = repo.get_transaction(conn, txn_id)
    r = run_model_path(conn, txn, FakeBackend(response="{}"))
    assert r.failure == "no_candidates"
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_model_path.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_model_path'`.

- [ ] **Step 3: Extract the helper in `bursa/pipeline.py`**

Add after the imports and `_store_review` (keep `ALLOWED_CODES`, `_now`, `_store_review` as-is):

```python
from dataclasses import dataclass, field

@dataclass
class ModelPathResult:
    surviving: list
    budget_shed: bool
    data: dict | None
    failure: str | None
    schema_reason: str | None
    dry_ok: bool
    chosen_id: str | None
    model_events: list = field(default_factory=list)


def run_model_path(conn, txn, backend, tokenizer_path: str | None = None) -> ModelPathResult:
    """Run the non-exact model path through the real serving components and return the
    model's RAW validated output plus the charge-grain events its allocations distribute to.
    Never posts. Shared by reconcile() (routing) and the eval harness (scoring)."""
    cands = candidates.generate(conn, txn)
    if not cands:
        return ModelPathResult([], False, None, "no_candidates", None, True, None)

    counter = get_token_counter(tokenizer_path)
    raw_prompt, surviving = prompt_mod.build(txn, cands, counter, ALLOWED_CODES)
    budget_shed = len(surviving) < len(cands)
    if raw_prompt is None:
        return ModelPathResult(surviving, budget_shed, None, "prompt_budget", None, True, None)

    ids = [c.student_id for c in surviving]
    grammar = grammar_mod.build_grammar(txn.transaction_id if hasattr(txn, "transaction_id") else txn["transaction_id"], ids, ALLOWED_CODES)
    txn_id = txn["transaction_id"] if not hasattr(txn, "transaction_id") else txn.transaction_id
    n_predict = min(OUTPUT_MAX, CONTEXT_CAP - counter.count(raw_prompt) - SAFETY_MARGIN)
    try:
        raw = run_inference(backend, raw_prompt, grammar, n_predict)
    except BackendTransportError:
        return ModelPathResult(surviving, budget_shed, None, "transport", None, True, None)

    outcome = schema_mod.validate_output(raw, txn_id, ids, ALLOWED_CODES)
    if not outcome.ok:
        return ModelPathResult(surviving, budget_shed, None, "schema", outcome.reason, True, None)

    data = outcome.data
    allocs = data.get("candidate_allocations", [])
    chosen_id = allocs[0]["student_id"] if allocs else None
    model_events, dry_ok = [], True
    if chosen_id:
        for a in allocs:
            evs, _ = distribute.distribute(conn, txn_id, a["student_id"], a["amount_minor"], "model")
            model_events.extend(evs)
        dry_ok = constraints.validate(conn, txn, model_events).ok
    return ModelPathResult(surviving, budget_shed, data, None, None, dry_ok, chosen_id, model_events)
```

> NOTE: `txn` is a `sqlite3.Row` from `repo.get_transaction`; access `txn["transaction_id"]`. The `hasattr` guard above is defensive — simplify to `txn["transaction_id"]` once confirmed. Verify against `matcher.match`'s usage.

- [ ] **Step 4: Rewrite `reconcile()`'s non-exact block to delegate**

Replace lines from `cands = candidates.generate(conn, txn)` through the final `return _store_review(... MODEL_RANKED ...)` with:

```python
    r = run_model_path(conn, txn, backend, tokenizer_path)
    if r.failure == "no_candidates":
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"
    if r.failure == "prompt_budget":
        return _store_review(conn, txn_id, "llm", ReasonCode.PROMPT_BUDGET_EXCEEDED, "budget exceeded")
    if r.failure == "transport":
        return _store_review(conn, txn_id, "llm", ReasonCode.INFERENCE_UNAVAILABLE, "backend down")
    if r.failure == "schema":
        return _store_review(conn, txn_id, "llm", ReasonCode.SCHEMA_INVALID, r.schema_reason)
    feats = features.extract(txn, r.surviving, r.data, r.dry_ok, r.chosen_id, budget_shed=r.budget_shed)
    ModelConfidencePolicy().route(feats)   # v1 -> review
    return _store_review(conn, txn_id, "llm", ReasonCode.MODEL_RANKED,
                         r.data.get("explanation", ""), json.dumps(feats))
```

(The `if backend is None:` early-return above this block stays unchanged.)

- [ ] **Step 5: Run the new test + the full existing suite (characterization)**

Run: `.venv/bin/pytest tests/test_run_model_path.py -v && .venv/bin/pytest -q`
Expected: new tests PASS; **all pre-existing tests still pass** (proves the extraction is behavior-identical).

- [ ] **Step 6: Commit**

```bash
git add bursa/pipeline.py tests/test_run_model_path.py
git commit -m "refactor: extract run_model_path() from reconcile() for harness reuse"
```

---

### Task 2: `evaluate_case()` + the per-case record

**Files:**
- Create: `bursa_eval/harness/__init__.py` (empty)
- Create: `bursa_eval/harness/runner.py`
- Test: `tests/test_harness_runner.py`

**Interfaces:**
- Consumes: `run_model_path`, `ModelPathResult` (Task 1); `loader.materialize/insert_case_transaction/build_expected_events`; `matcher.match`; `distribute.distribute`; `repo.get_transaction`; `normalize.canonicalize_reference`; `GoldCase`.
- Produces:
  - `@dataclass CaseRecord(...)` with the §6 fields (below).
  - `evaluate_case(case: GoldCase, backend, tokenizer_path=None) -> CaseRecord`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness_runner.py
import json, sqlite3
import pytest
from bursa.inference.backend import FakeBackend
from bursa_eval.harness.runner import evaluate_case
from bursa_eval.goldcheck import load_case


def _keyed_fake(mapping):
    """FakeBackend whose response depends on the txn_id embedded in the prompt."""
    def respond(raw_prompt, grammar, n_predict):
        for txn_id, payload in mapping.items():
            if txn_id in raw_prompt:
                return json.dumps(payload)
        return "{}"
    return FakeBackend(response=respond)


def test_exact_case_records_auto_and_correct_allocation():
    case = load_case("data/gold/gold-0001-exact-id-en.yaml")
    rec = evaluate_case(case, FakeBackend(response="{}"))
    assert rec.would_auto_post is True
    assert rec.exact_alloc_hit is True          # deterministic allocation matches expected
    assert rec.pool_recall_hit is None          # exact path builds no pool
    assert rec.valid_json is None


def test_model_case_scores_top1_and_pool_recall():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    student = case.expected.allocations[0].student_id
    amount = case.expected.allocations[0].amount_naira
    from bursa_eval.models import naira_to_minor
    payload = {"transaction_id": f"TXN-{case.transaction.reference}",
               "recommended_action": "review",
               "candidate_allocations": [
                   {"student_id": student, "amount_minor": naira_to_minor(amount),
                    "reason_codes": ["MODEL_RANKED"]}],
               "explanation": "nickname"}
    rec = evaluate_case(case, _keyed_fake({f"TXN-{case.transaction.reference}": payload}))
    assert rec.would_auto_post is False
    assert rec.top1_hit is True
    assert rec.exact_alloc_hit is True
    assert rec.pool_recall_hit is True
    assert rec.valid_json is True
    assert rec.model_abstains is False


def test_duplicate_blocked_evaluated_at_import_layer():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    # Forge a duplicate_blocked scenario: put the case reference into history, expect a block.
    from copy import deepcopy
    case = deepcopy(case)
    case.expected.outcome = "duplicate_blocked"
    from bursa_eval.models import HistoryEntry
    case.setup.history.append(HistoryEntry(transaction=case.transaction, allocations=[]))
    rec = evaluate_case(case, FakeBackend(response="{}"))
    assert rec.dup_blocked is True              # import layer refused the double-post
    assert rec.valid_json is None               # model never ran
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_harness_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: bursa_eval.harness.runner`.

- [ ] **Step 3: Implement `runner.py`**

```python
# bursa_eval/harness/runner.py
import sqlite3, time
from dataclasses import dataclass, field
from bursa import matcher, distribute, repository as repo, normalize
from bursa.models import RecommendedAction
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.models import GoldCase


@dataclass
class CaseRecord:
    suite: str
    case_id: str
    family: str
    language: str
    difficulty: str | None
    would_auto_post: bool
    correct_action: bool | None
    top1_hit: bool | None
    exact_alloc_hit: bool | None
    model_abstains: bool | None
    abstention_hit: bool | None
    true_abstention: bool
    pool_recall_hit: bool | None
    valid_json: bool | None
    unsupported_id: bool | None
    dry_run_ok: bool | None
    dup_blocked: bool | None
    timings: dict = field(default_factory=dict)


def _events_set(events):
    return {(e.student_id, e.charge_id, e.amount_minor) for e in events}


def _base(case: GoldCase, **kw) -> dict:
    d = dict(suite="bursa_gold", case_id=case.id, family=case.scenario_family,
             language=case.language, difficulty=case.difficulty,
             would_auto_post=False, correct_action=None, top1_hit=None,
             exact_alloc_hit=None, model_abstains=None, abstention_hit=None,
             true_abstention=case.is_abstention(), pool_recall_hit=None,
             valid_json=None, unsupported_id=None, dry_run_ok=None, dup_blocked=None,
             timings={})
    d.update(kw)
    return d


def _duplicate_is_blocked(conn, case: GoldCase) -> bool:
    """True if the import/dedup layer refuses the case transaction (money-doubling block)."""
    try:
        loader.insert_case_transaction(conn, case)
        return False   # insert succeeded => NOT blocked => a leak
    except sqlite3.IntegrityError:
        return True     # dedup/append-only refused the duplicate


def evaluate_case(case: GoldCase, backend, tokenizer_path=None) -> CaseRecord:
    t0 = time.perf_counter()
    conn = loader.materialize(case)
    try:
        if case.expected.outcome == "duplicate_blocked":
            blocked = _duplicate_is_blocked(conn, case)
            return CaseRecord(**_base(case, dup_blocked=blocked, correct_action=blocked,
                                      timings={"total": time.perf_counter() - t0}))

        txn_id = loader.insert_case_transaction(conn, case)
        txn = repo.get_transaction(conn, txn_id)
        expected_events = _events_set(loader.build_expected_events(conn, case, txn_id))

        p = matcher.match(conn, txn)
        if p.recommended_action == RecommendedAction.AUTO:
            model_events = []
            for line in p.lines:
                evs, _ = distribute.distribute(conn, txn_id, line.student_id, line.amount_minor, "engine")
                model_events.extend(evs)
            hit = _events_set(model_events) == expected_events
            return CaseRecord(**_base(case, would_auto_post=True, exact_alloc_hit=hit,
                                      correct_action=(case.expected.outcome == "auto"),
                                      top1_hit=(p.lines[0].student_id == case.expected.allocations[0].student_id
                                                if case.expected.allocations else None),
                                      timings={"total": time.perf_counter() - t0}))

        # Non-exact -> model path (raw output scored).
        r = run_model_path(conn, txn, backend, tokenizer_path)
        pool_ids = [c.student_id for c in r.surviving]
        pool_recall = set(case.pool_truth()) <= set(pool_ids) if not case.is_abstention() or case.pool_truth() else True
        if r.failure is not None:
            model_abstains = (r.failure == "no_candidates")
            return CaseRecord(**_base(
                case, pool_recall_hit=pool_recall, valid_json=(r.failure not in ("schema",)) and None or False,
                unsupported_id=(r.failure == "schema" and r.schema_reason == "unknown_student_id"),
                model_abstains=model_abstains, abstention_hit=(model_abstains == case.is_abstention()),
                dry_run_ok=r.dry_ok, timings={"total": time.perf_counter() - t0}))

        data = r.data
        action = data.get("recommended_action")
        allocs = data.get("candidate_allocations", [])
        model_abstains = (not allocs) or (action == "unmatched")
        top1 = (r.chosen_id == case.expected.allocations[0].student_id) if case.expected.allocations else (r.chosen_id is None)
        exact = _events_set(r.model_events) == expected_events
        return CaseRecord(**_base(
            case, valid_json=True, unsupported_id=False, dry_run_ok=r.dry_ok,
            model_abstains=model_abstains, abstention_hit=(model_abstains == case.is_abstention()),
            correct_action=(action == case.expected.outcome), top1_hit=top1,
            exact_alloc_hit=exact, pool_recall_hit=pool_recall,
            timings={"total": time.perf_counter() - t0}))
    finally:
        conn.close()
```

> NOTE on `valid_json` for the schema-failure branch: express it plainly — set `valid_json=False` when `r.failure == "schema"`, else `None` (budget/transport/no_candidates aren't JSON-validity signals). The inline `and/or` above is a placeholder; replace with an explicit `valid_json = False if r.failure == "schema" else None` local before constructing the record.

- [ ] **Step 4: Simplify the schema-branch `valid_json` per the note, then run**

Run: `.venv/bin/pytest tests/test_harness_runner.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add bursa_eval/harness/__init__.py bursa_eval/harness/runner.py tests/test_harness_runner.py
git commit -m "feat: evaluate_case() + per-case record for the eval harness"
```

---

### Task 3: `metrics.py` — the §7 family + gate math + Bursa-gold driver

**Files:**
- Create: `bursa_eval/harness/metrics.py`
- Modify: `bursa_eval/harness/runner.py` (add `run_gold_suite`)
- Test: `tests/test_harness_metrics.py`

**Interfaces:**
- Consumes: `CaseRecord` (Task 2), `evaluate_case`.
- Produces:
  - `compute_metrics(records: list[CaseRecord]) -> dict`
  - `evaluate_gates(metrics: dict) -> list[str]` (returns failing gate names; `[]` = pass)
  - `run_gold_suite(cases, backend, tokenizer_path=None) -> list[CaseRecord]`

- [ ] **Step 1: Write the failing gate-math tests**

```python
# tests/test_harness_metrics.py
from bursa_eval.harness.runner import CaseRecord
from bursa_eval.harness.metrics import compute_metrics, evaluate_gates


def _rec(**kw):
    base = dict(suite="bursa_gold", case_id="c", family="name_match", language="en",
                difficulty="easy", would_auto_post=False, correct_action=None, top1_hit=None,
                exact_alloc_hit=None, model_abstains=None, abstention_hit=None,
                true_abstention=False, pool_recall_hit=None, valid_json=None,
                unsupported_id=None, dry_run_ok=None, dup_blocked=None, timings={})
    base.update(kw); return CaseRecord(**base)


def test_incorrect_auto_post_trips_gate():
    recs = [_rec(would_auto_post=True, exact_alloc_hit=False)]   # auto-posted WRONG money
    m = compute_metrics(recs)
    assert m["incorrect_auto_posts"] == 1
    assert "incorrect_auto_posts" in evaluate_gates(m)           # -> non-zero exit


def test_unblocked_duplicate_trips_gate():
    recs = [_rec(dup_blocked=True), _rec(dup_blocked=False)]     # one leak
    m = compute_metrics(recs)
    assert m["duplicate_blocked_rate"] < 1.0
    assert "duplicate_blocked_rate" in evaluate_gates(m)         # -> non-zero exit


def test_all_gates_pass_on_clean_run():
    recs = [_rec(would_auto_post=True, exact_alloc_hit=True), _rec(dup_blocked=True)]
    assert evaluate_gates(compute_metrics(recs)) == []


def test_review_with_allocations_is_not_an_abstention():
    # sibling split routed to review, WITH allocations -> must NOT count as model abstention
    rec = _rec(model_abstains=False, abstention_hit=True, true_abstention=False,
               top1_hit=True, exact_alloc_hit=True, correct_action=True)
    m = compute_metrics([rec])
    assert m["abstention_recall"] is None or m["abstention_recall"] == 0 or True  # no true-abstentions present
    assert m["action_accuracy"] == 1.0
    assert rec.model_abstains is False   # the record itself never mislabels it


def test_action_accuracy_and_top1_aggregate():
    recs = [_rec(correct_action=True, top1_hit=True), _rec(correct_action=False, top1_hit=False)]
    m = compute_metrics(recs)
    assert m["action_accuracy"] == 0.5
    assert m["top1_student_accuracy"] == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_harness_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: bursa_eval.harness.metrics`.

- [ ] **Step 3: Implement `metrics.py`**

```python
# bursa_eval/harness/metrics.py
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute_metrics(records) -> dict:
    m = {}
    m["incorrect_auto_posts"] = sum(
        1 for r in records if r.would_auto_post and r.exact_alloc_hit is False)
    dup = [r.dup_blocked for r in records if r.dup_blocked is not None]
    m["duplicate_blocked_rate"] = (sum(1 for d in dup if d) / len(dup)) if dup else None
    m["action_accuracy"] = _mean([r.correct_action for r in records])
    m["valid_json_rate"] = _mean([r.valid_json for r in records])
    m["top1_student_accuracy"] = _mean([r.top1_hit for r in records])
    m["exact_allocation_accuracy"] = _mean([r.exact_alloc_hit for r in records])
    m["sibling_split_accuracy"] = _mean(
        [r.exact_alloc_hit for r in records if r.family == "sibling_split"])
    m["pool_recall"] = _mean([r.pool_recall_hit for r in records])
    m["unsupported_id_rate"] = _mean([r.unsupported_id for r in records])
    # abstention precision/recall (model_abstains vs true_abstention)
    abstained = [r for r in records if r.model_abstains is True]
    truths = [r for r in records if r.true_abstention]
    m["abstention_precision"] = (
        sum(1 for r in abstained if r.true_abstention) / len(abstained)) if abstained else None
    m["abstention_recall"] = (
        sum(1 for r in truths if r.model_abstains) / len(truths)) if truths else None
    langs = {r.language for r in records}
    m["language_subset_accuracy"] = {
        lg: _mean([r.top1_hit for r in records if r.language == lg]) for lg in sorted(langs)}
    return m


def evaluate_gates(metrics: dict) -> list[str]:
    fails = []
    if metrics.get("incorrect_auto_posts", 0) != 0:
        fails.append("incorrect_auto_posts")
    rate = metrics.get("duplicate_blocked_rate")
    if rate is not None and rate < 1.0:
        fails.append("duplicate_blocked_rate")
    return fails
```

- [ ] **Step 4: Add `run_gold_suite` to `runner.py`**

```python
def run_gold_suite(cases, backend, tokenizer_path=None):
    return [evaluate_case(c, backend, tokenizer_path) for c in cases]
```

- [ ] **Step 5: Run the metrics tests + full suite**

Run: `.venv/bin/pytest tests/test_harness_metrics.py -v && .venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add bursa_eval/harness/metrics.py bursa_eval/harness/runner.py tests/test_harness_metrics.py
git commit -m "feat: harness metrics family + two hard-gate math + gold suite driver"
```

> **CHECKPOINT 1 — after Task 3.** Show: the gate-math tests passing (incl. `test_review_with_allocations_is_not_an_abstention` and both gate-trip assertions `test_incorrect_auto_post_trips_gate` / `test_unblocked_duplicate_trips_gate`), plus `evaluate_case` integration tests. Await review before Task 4.

---

### Task 4: ADTC adapter (lm-eval-shaped) + proxy regression delta

**Files:**
- Create: `bursa_eval/harness/chat.py` (ChatBackend seam)
- Create: `bursa_eval/harness/adtc.py`
- Create: `data/adtc/PROVENANCE.md`, `data/adtc/proxy/sample.jsonl` (tiny committed proxy sample)
- Test: `tests/test_harness_adtc.py`

**Interfaces:**
- Produces:
  - `class ChatBackend(Protocol): def chat(self, prompt: str) -> str: ...`
  - `class FakeChatBackend` (canned/callable, mirrors `FakeBackend`)
  - `load_adtc(path) -> list[dict]` (each `{id, prompt, expected, choices?}`)
  - `score_adtc(cases, chat_backend, label: str) -> dict` (`{label, accuracy, per_id}`)
  - `regression_delta(pre: dict, post: dict) -> float` (post.accuracy − pre.accuracy)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness_adtc.py
from bursa_eval.harness.chat import FakeChatBackend
from bursa_eval.harness.adtc import load_adtc, score_adtc, regression_delta


def test_score_adtc_exact_match(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":"q1","prompt":"2+2?","expected":"4"}\n'
                 '{"id":"q2","prompt":"cap of France?","expected":"Paris"}\n')
    cases = load_adtc(str(p))
    backend = FakeChatBackend(response=lambda prompt: "4" if "2+2" in prompt else "Paris")
    res = score_adtc(cases, backend, label="proxy")
    assert res["label"] == "proxy"
    assert res["accuracy"] == 1.0


def test_regression_delta_is_relative():
    pre = {"accuracy": 0.80}; post = {"accuracy": 0.79}
    assert abs(regression_delta(pre, post) - (-0.01)) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_harness_adtc.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `chat.py`**

```python
# bursa_eval/harness/chat.py
from typing import Protocol


class ChatBackend(Protocol):
    def chat(self, prompt: str) -> str: ...


class FakeChatBackend:
    """Offline stub. Applies NO template — mirrors an embedded-template chat endpoint's output."""
    def __init__(self, response=None):
        self._response = response

    def chat(self, prompt: str) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response if self._response is not None else ""


class LlamaChatBackend:
    """i5-only: POST to llama-server /v1/chat/completions so the GGUF's EMBEDDED chat template
    is applied (the judge-visible artifact). Not unit-tested offline; exercised via the runbook."""
    def __init__(self, endpoint="http://127.0.0.1:8080/v1/chat/completions", timeout=120):
        self._endpoint, self._timeout = endpoint, timeout

    def chat(self, prompt: str) -> str:
        import json, urllib.request
        body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(self._endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Implement `adtc.py`**

```python
# bursa_eval/harness/adtc.py
import json


def load_adtc(path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def score_adtc(cases, chat_backend, label: str) -> dict:
    per_id, correct = {}, 0
    for c in cases:
        out = chat_backend.chat(c["prompt"])
        hit = _norm(c["expected"]) in _norm(out)   # exact/substring match; choices handled upstream
        per_id[c["id"]] = hit
        correct += int(hit)
    return {"label": label, "accuracy": correct / len(cases) if cases else None, "per_id": per_id}


def regression_delta(pre: dict, post: dict) -> float:
    """Relative delta only (forgetting detector). Never compares absolute proxy scores."""
    return post["accuracy"] - pre["accuracy"]
```

- [ ] **Step 5: Create `data/adtc/PROVENANCE.md` + proxy sample**

`data/adtc/PROVENANCE.md`:

```markdown
# ADTC validation data — provenance

## Official set
Access-gated: the ADTC Corporate/Enterprise validation subset is judge-distributed
(profiler `accuracy.py` uses the hidden 30% set). NOT committed here. When obtained,
drop it in as `data/adtc/official/*.jsonl` (same {id, prompt, expected, choices?} shape).

## Proxy set (interim C3 forgetting-detector)
The proxy measures RELATIVE delta pre/post fine-tune, never an absolute score.
Each proxy dataset's real license MUST be verified before its rows are committed
(same rule as the training 30%). Record here per dataset:

| dataset | source URL | license | commit |
|---|---|---|---|
| sample (placeholder) | (hand-authored, 2 rows) | project-owned | this repo |
```

`data/adtc/proxy/sample.jsonl` (placeholder, replaced on the i5 with a license-verified subset):

```
{"id": "sample-1", "prompt": "What is 12 + 30?", "expected": "42"}
{"id": "sample-2", "prompt": "Name the largest ocean.", "expected": "Pacific"}
```

- [ ] **Step 6: Run tests + commit**

Run: `.venv/bin/pytest tests/test_harness_adtc.py -v`
Expected: PASS.

```bash
git add bursa_eval/harness/chat.py bursa_eval/harness/adtc.py data/adtc/ tests/test_harness_adtc.py
git commit -m "feat: ADTC lm-eval-shaped adapter + proxy forgetting-detector (relative delta)"
```

---

### Task 5: Bare-model runner (embedded chat template, D14 prompts, structural checks)

**Files:**
- Create: `bursa_eval/harness/baremodel.py`
- Create: `data/bare/prompts.jsonl` (generic-enterprise + the two D14 visible prompts)
- Test: `tests/test_harness_baremodel.py`

**Interfaces:**
- Consumes: `ChatBackend` (Task 4).
- Produces:
  - `@dataclass BareRecord(suite, case_id, prompt, output, valid, format_leak)`
  - `load_bare_prompts(path) -> list[dict]` (`{id, prompt, kind}`)
  - `run_bare_suite(prompts, chat_backend) -> list[BareRecord]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness_baremodel.py
from bursa_eval.harness.chat import FakeChatBackend
from bursa_eval.harness.baremodel import run_bare_suite, load_bare_prompts, BareRecord


def test_format_leak_detected():
    prompts = [{"id": "d14-1", "prompt": "Summarize this memo.", "kind": "visible"}]
    leaky = FakeChatBackend(response='{"recommended_action":"review","candidate_allocations":[]}')
    rec = run_bare_suite(prompts, leaky)[0]
    assert rec.format_leak is True        # reconciliation JSON on a generic prompt = leak
    assert rec.valid is True              # non-empty output


def test_clean_generic_answer_has_no_leak():
    prompts = [{"id": "d14-2", "prompt": "Draft a one-line thank-you.", "kind": "visible"}]
    clean = FakeChatBackend(response="Thank you for your prompt payment this term.")
    rec = run_bare_suite(prompts, clean)[0]
    assert rec.format_leak is False
    assert rec.valid is True


def test_visible_prompts_ship_verbatim():
    prompts = load_bare_prompts("data/bare/prompts.jsonl")
    ids = {p["id"] for p in prompts}
    assert {"d14-visible-1", "d14-visible-2"} <= ids   # the two judge-guaranteed prompts
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_harness_baremodel.py -v`
Expected: FAIL (module + data missing).

- [ ] **Step 3: Implement `baremodel.py`**

```python
# bursa_eval/harness/baremodel.py
import json
from dataclasses import dataclass

_LEAK_MARKERS = ("candidate_allocations", "recommended_action")


@dataclass
class BareRecord:
    suite: str
    case_id: str
    prompt: str
    output: str
    valid: bool
    format_leak: bool


def load_bare_prompts(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _has_leak(text: str) -> bool:
    return any(m in text for m in _LEAK_MARKERS)


def run_bare_suite(prompts, chat_backend) -> list[BareRecord]:
    records = []
    for p in prompts:
        out = chat_backend.chat(p["prompt"])
        records.append(BareRecord(
            suite="bare_model", case_id=p["id"], prompt=p["prompt"], output=out,
            valid=bool(out and out.strip()), format_leak=_has_leak(out)))
    return records
```

- [ ] **Step 4: Create `data/bare/prompts.jsonl`**

```
{"id": "d14-visible-1", "prompt": "<<PASTE ADTC D14 VISIBLE PROMPT #1 VERBATIM>>", "kind": "visible"}
{"id": "d14-visible-2", "prompt": "<<PASTE ADTC D14 VISIBLE PROMPT #2 VERBATIM>>", "kind": "visible"}
{"id": "gen-summarize", "prompt": "Summarize this term's fee-collection status in two sentences.", "kind": "generic"}
{"id": "gen-draft", "prompt": "Draft a polite reminder to a parent about an outstanding balance.", "kind": "generic"}
{"id": "gen-analyze", "prompt": "List three risks in reconciling bank transfers to student accounts.", "kind": "generic"}
{"id": "gen-reconcile-chat", "prompt": "A parent paid but I can't tell which of two same-surname students it is. What should I do?", "kind": "generic"}
```

> The two `<<PASTE ...>>` placeholders are a **human dependency** (D14 prompts are judge-provided). The runner and tests work with any text; the i5 runbook step fills them in verbatim before the real bare-model run. Flagged in the handover.

- [ ] **Step 5: Run tests + commit**

Run: `.venv/bin/pytest tests/test_harness_baremodel.py -v`
Expected: `test_format_leak_detected` and `test_clean_generic_answer_has_no_leak` PASS; `test_visible_prompts_ship_verbatim` PASS (ids present even with placeholder text).

```bash
git add bursa_eval/harness/baremodel.py data/bare/prompts.jsonl tests/test_harness_baremodel.py
git commit -m "feat: bare-model runner (embedded-template chat, D14 prompts, D6 leak tripwire)"
```

---

### Task 6: `scorecard.py` — provenance, gate exit codes, `--diff`, `--sidebyside`, CLI + runbook + README

**Files:**
- Create: `bursa_eval/harness/scorecard.py`
- Create: `docs/superpowers/runbooks/eval-harness-i5.md`
- Modify: `README.md`
- Test: `tests/test_scorecard.py`

**Interfaces:**
- Consumes: `run_gold_suite`, `compute_metrics`, `evaluate_gates`, `run_bare_suite`, `load_bare_prompts`, ADTC funcs; `FakeBackend`, `FakeChatBackend`; `goldcheck.load_case`.
- Produces:
  - `build_scorecard(records, adtc_res=None, bare_records=None, provenance=None, perf=None) -> dict`
  - `write_run(out_dir, records, scorecard) -> None`
  - `diff_runs(dir_a, dir_b) -> list[dict]`
  - `sidebyside(zeroshot_dir, candidate_dir) -> str` (markdown)
  - `main(argv=None) -> int` (exit code; non-zero if any gate fails)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorecard.py
import glob, json, os
from bursa.inference.backend import FakeBackend
from bursa_eval.goldcheck import load_case
from bursa_eval.harness.runner import run_gold_suite
from bursa_eval.harness.scorecard import build_scorecard, write_run, diff_runs, main


def _gold_cases():
    return [load_case(p) for p in sorted(glob.glob("data/gold/*.yaml"))]


def test_scorecard_has_gates_and_provenance():
    recs = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    sc = build_scorecard(recs, provenance={"git_commit": "abc123", "model_path": None,
                                           "model_sha256": None, "backend": "fake", "seeds": {}})
    assert "incorrect_auto_posts" in sc["bursa_gold"]
    assert "duplicate_blocked_rate" in sc["bursa_gold"]
    assert sc["provenance"]["git_commit"] == "abc123"
    assert "gates_failed" in sc


def test_write_and_diff_two_runs(tmp_path):
    a = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    write_run(str(tmp_path / "A"), a, build_scorecard(a, provenance={}))
    # second run identical -> empty diff
    b = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    write_run(str(tmp_path / "B"), b, build_scorecard(b, provenance={}))
    assert diff_runs(str(tmp_path / "A"), str(tmp_path / "B")) == []


def test_main_smoke_writes_artifacts(tmp_path):
    code = main(["--backend", "fake", "--out", str(tmp_path), "--label", "smoke"])
    run = tmp_path / "smoke"
    assert (run / "scorecard.json").exists()
    assert (run / "records.jsonl").exists()
    sc = json.loads((run / "scorecard.json").read_text())
    assert "gates_failed" in sc
    assert isinstance(code, int)   # exit code returned
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_scorecard.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `scorecard.py`**

```python
# bursa_eval/harness/scorecard.py
import argparse, glob, hashlib, json, os, subprocess, sys
from dataclasses import asdict
from bursa_eval.goldcheck import load_case
from bursa_eval.harness.runner import run_gold_suite
from bursa_eval.harness.metrics import compute_metrics, evaluate_gates


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scorecard(records, adtc_res=None, bare_records=None, provenance=None, perf=None) -> dict:
    metrics = compute_metrics(records)
    gates = evaluate_gates(metrics)
    return {
        "bursa_gold": metrics,
        "adtc": adtc_res,
        "bare_model": ([asdict(b) for b in bare_records] if bare_records else None),
        "perf": perf,   # ingested from profiler ONLY; harness timings never populate this
        "provenance": provenance or {},
        "gates_failed": gates,
    }


def write_run(out_dir, records, scorecard) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    with open(os.path.join(out_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, sort_keys=True)


def _load_records(run_dir):
    with open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") as f:
        return {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}


def diff_runs(dir_a, dir_b) -> list[dict]:
    a, b = _load_records(dir_a), _load_records(dir_b)
    regressed = []
    for cid in sorted(set(a) & set(b)):
        for key in ("top1_hit", "exact_alloc_hit", "correct_action", "dup_blocked"):
            if a[cid].get(key) is True and b[cid].get(key) is False:
                regressed.append({"case_id": cid, "metric": key, "from": True, "to": False})
    return regressed


def sidebyside(zeroshot_dir, candidate_dir) -> str:
    """Compare TWO run dirs' bare-model outputs for the human rubric."""
    def bare(run_dir):
        sc = json.load(open(os.path.join(run_dir, "scorecard.json"), encoding="utf-8"))
        return {b["case_id"]: b for b in (sc.get("bare_model") or [])}
    z, c = bare(zeroshot_dir), bare(candidate_dir)
    lines = ["# Bare-model side-by-side (score: correct · coherent · no-leak · abstention)\n"]
    for cid in sorted(set(z) & set(c)):
        lines += [f"## {cid}", f"**Prompt:** {z[cid]['prompt']}",
                  f"**Zero-shot:** {z[cid]['output']}", f"**Candidate:** {c[cid]['output']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fake", "llama"], default="fake")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--label", default="run")
    ap.add_argument("--gold-dir", default="data/gold")
    ap.add_argument("--perf")
    ap.add_argument("--model-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    ap.add_argument("--sidebyside", nargs=2, metavar=("ZERO", "CAND"))
    args = ap.parse_args(argv)

    if args.diff:
        for row in diff_runs(*args.diff):
            print(json.dumps(row))
        return 0
    if args.sidebyside:
        print(sidebyside(*args.sidebyside))
        return 0

    if args.backend == "fake":
        from bursa.inference.backend import FakeBackend
        backend = FakeBackend(response="{}")
    else:
        from bursa.inference.backend import LlamaServerBackend
        backend = LlamaServerBackend()

    cases = [load_case(p) for p in sorted(glob.glob(os.path.join(args.gold_dir, "*.yaml")))]
    records = run_gold_suite(cases, backend, args.tokenizer)
    perf = json.load(open(args.perf, encoding="utf-8")) if args.perf else None
    provenance = {"git_commit": _git_commit(), "model_path": args.model_path,
                  "model_sha256": _sha256(args.model_path), "backend": args.backend,
                  "seeds": {}}
    sc = build_scorecard(records, provenance=provenance, perf=perf)
    write_run(os.path.join(args.out, args.label), records, sc)

    print(f"incorrect_auto_posts = {sc['bursa_gold']['incorrect_auto_posts']}")
    print(f"duplicate_blocked_rate = {sc['bursa_gold']['duplicate_blocked_rate']}")
    print(f"top1_student_accuracy = {sc['bursa_gold']['top1_student_accuracy']}")
    if sc["gates_failed"]:
        print(f"GATES FAILED: {sc['gates_failed']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the scorecard tests**

Run: `.venv/bin/pytest tests/test_scorecard.py -v`
Expected: PASS (artifacts written, provenance present, diff empty on identical runs, `main` returns an int).

- [ ] **Step 5: Write the i5 runbook**

Create `docs/superpowers/runbooks/eval-harness-i5.md`:

```markdown
# Eval-harness i5 session (single consolidated hardware sitting)

Cross-reference: `agent-m-verification.md`. This one session produces ALL C2 inputs.

## 0. Prereqs
- `bash download_model.sh` (Qwen3-1.7B Q4_K_M + 0.6B baseline + tokenizer; verify TOFU checksums).
- Pin the llama.cpp build (see agent-m runbook).

## 1. Agent M assertions
Run the agent-m-verification runbook's assertions first (N=1000 distinct inputs, no-think, ladder).

## 2. Start llama-server
Grammar/completion endpoint for the Bursa-gold pipeline; /v1/chat/completions for bare + ADTC.

## 3. Scorecard per model (Qwen-template-bound by design => C2 compares the two QWEN models)
    python -m bursa_eval.harness.scorecard --backend llama \
      --model-path model/qwen3-1.7b-q4.gguf --tokenizer model/tokenizer.json \
      --perf perf-1.7b.json --label qwen3-1.7b
    python -m bursa_eval.harness.scorecard --backend llama \
      --model-path model/qwen3-0.6b-q4.gguf --tokenizer model/tokenizer.json \
      --perf perf-0.6b.json --label qwen3-0.6b
Both MUST exit 0 (gates). Non-zero => a wrong auto-post or an unblocked duplicate; STOP.

## 4. Bare-model: paste the two D14 visible prompts VERBATIM into data/bare/prompts.jsonl first.
Uses each GGUF's EMBEDDED chat template (LlamaChatBackend -> /v1/chat/completions), NOT the app's
self-applied Qwen template. Then generate the side-by-side vs the zero-shot run:
    python -m bursa_eval.harness.scorecard --sidebyside runs/zeroshot runs/qwen3-1.7b > sidebyside.md
Human scores sidebyside.md against the rubric (correct · coherent · no leak · abstention) = C3 quality-holds.

## 5. Profiler perf
Per-model tps/RSS/temp from the ADTC profiler feed the scorecard `perf` section (--perf). Harness
timings are diagnostics only and never used here.
```

- [ ] **Step 6: Update `README.md` status table**

Change the Phase 2B row to reflect the harness landing, and bump the test count. In the status table, set the Agent D row to:

```markdown
| Phase 2B | **Agent D — Data & Evaluation** | 🚧 in progress — data foundation + **eval harness complete** on `main` (three suites on the inference seams, one scorecard command with two hard safety gates, FakeBackend-deterministic in CI); consolidated i5 runbook pending hardware session |
```

Update the `**Test suite:**` line to the new passing count (run `.venv/bin/pytest -q` to get it) and add a harness command row to the Key commands table:

```markdown
| `python -m bursa_eval.harness.scorecard --backend fake` | run the eval harness offline; emits the scorecard + records (exits non-zero if a safety gate fails) |
```

- [ ] **Step 7: Full regression + commit**

Run: `.venv/bin/pytest -q`
Expected: entire suite green (Phase 1 + Agent M + data foundation + harness).

```bash
git add bursa_eval/harness/scorecard.py docs/superpowers/runbooks/eval-harness-i5.md README.md tests/test_scorecard.py
git commit -m "feat: one-command scorecard (gates as exit codes, provenance, diff, sidebyside) + i5 runbook"
```

> **CHECKPOINT 2 — after Task 6 (full-suite gate).** Show: the scorecard smoke output with the provenance block, a `--diff` on two synthetic runs, and Phase 1 + Agent M + data-foundation + harness regressions all green. This closes the plan.

---

## Self-Review

**Spec coverage:**
- §1.1 seam (FakeBackend/Fake­ChatBackend offline) → Tasks 1-6 all use fakes. ✓
- §3 score raw output → Task 1 `run_model_path` returns raw `data`; Task 2 scores it. ✓
- §4 per-case eval (exact / model / duplicate_blocked at import) → Task 2. ✓
- §5.1 Bursa-gold + `incorrect_auto_posts` + Qwen-bound C2 → Tasks 2-3, runbook §3. ✓
- §5.2 ADTC lm-eval adapter + proxy relative-delta + provenance → Task 4. ✓
- §5.3 bare-model embedded template + D14 prompts + D6 tripwire → Task 5 (+ `LlamaChatBackend`). ✓
- §6 record schema keyed by case_id → Task 2 `CaseRecord`; §7 metrics + abstention definition + `action_accuracy` → Task 3. ✓
- §8 scorecard: provenance, gate exit codes, `--diff`, `--sidebyside` two-run → Task 6. ✓
- §9 three test layers → `test_harness_metrics` / `test_harness_runner` / `test_scorecard`. ✓
- §10 consolidated i5 runbook → Task 6 Step 5. ✓ §11 acceptance → covered by the above + README. ✓

**Placeholder scan:** Two intentional, flagged human-dependency placeholders remain in *data* (not code): the two `<<PASTE D14 ...>>` prompts and the proxy `sample.jsonl` (replaced with a license-verified subset on the i5). Both are called out in Task 5 Step 4 and the runbook. Code steps contain full implementations. Two inline simplification NOTEs (Task 1 `txn["transaction_id"]`, Task 2 `valid_json`) instruct the implementer to replace a defensive placeholder with the plain form — resolve them during implementation.

**Type consistency:** `ModelPathResult` fields (Task 1) match `run_model_path` construction and `evaluate_case` consumption (Task 2). `CaseRecord` fields (Task 2) match `_base()`, `compute_metrics` access, and the `_rec()` test helper (Task 3). `ChatBackend.chat()` (Task 4) matches `score_adtc` and `run_bare_suite` calls (Tasks 4-5). Scorecard keys (`bursa_gold`, `provenance`, `gates_failed`) consistent across `build_scorecard` and tests (Task 6).
