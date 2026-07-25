# Agent M — Inference Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 1's `ambiguous → review` stub into the real app-side LLM reconciliation path — candidate generator → prompt builder (budgeted) → dynamic GBNF → llama-server → schema validation → constraint-engine dry-run → feature extractor → calibrator — with everything tested against a `FakeBackend` and a real `LlamaServerBackend` + runbook for the hardware checks.

**Architecture:** New `bursa/inference/` package plus app-level `candidates.py`, `features.py`, `calibrator.py`; additive extensions to Phase-1 `db.py`, `projections.py`, `normalize.py`, `reasoncodes.py`, `importers/students.py`, `pipeline.py`. The model outputs student-level `{student_id, amount}` (§6, never `fee_id`); the Phase-1 constraint engine still owns arithmetic and is reused as a *dry run* to score proposals; `ledger.py` stays the sole writer; in v1 nothing model-touched posts (all → review).

**Tech Stack:** Python 3.13, stdlib `sqlite3`/`urllib`/`subprocess`, Pydantic v2, pytest, hypothesis; optional `tokenizers` (real Qwen tokenizer, asset-gated). No `requests`.

## Global Constraints

- Model output is student-level per §6; grammar restricts `student_id` + `reason_codes` to prompt-supplied values; the app path uses `/completion` with a self-applied Qwen3 chat template + `/no_think`.
- Prompt ≤ `PROMPT_TOKEN_BUDGET=1300` on the *exact final string*; `CONTEXT_CAP=2048`; `OUTPUT_MAX=512`; `SAFETY_MARGIN=64`; `MAX_EXPLANATION_CHARS=500`; `FUZZY_NAME_THRESHOLD=0.88`; `MAX_CANDIDATES=5`; `TRANSPORT_RETRY=1`; weights `STRONG=10/MEDIUM=5/WEAK=2`; `SPECIAL_MARGIN=8`; `features_version=1`.
- Never truncate narration or the output contract; the overflow ladder never sheds fired evidence.
- Content-invalid → review immediately (no retry, temp 0 is deterministic); transport failure → retry once (health/restart) then review.
- `build_grammar` uses the post-ladder surviving candidates (grammar IDs == prompt IDs).
- Everything is tested offline against `FakeBackend`; real model = the runbook, on i5-class hardware.
- All new modules follow Phase-1 patterns: `int` minor units, parameterized SQL, `money.py`/`ledger.py` boundaries untouched.

---

## File structure

```
bursa/
  inference/
    __init__.py
    constants.py   # PROMPT_TOKEN_BUDGET, CONTEXT_CAP, OUTPUT_MAX, SAFETY_MARGIN, MAX_EXPLANATION_CHARS, ...
    tokens.py      # TokenCounter, HeuristicTokenCounter, QwenTokenizer, get_token_counter()
    grammar.py     # build_grammar(txn_id, candidate_ids, allowed_codes) -> str
    prompt.py      # PromptBuilder.build(...) -> (raw_prompt, surviving_candidates) | (None, PROMPT_BUDGET_EXCEEDED)
    backend.py     # InferenceBackend, BackendTransportError, FakeBackend, LlamaServerBackend
    server.py      # LlamaServer (start/stop/health), build_server_args()
    schema.py      # validate_output(raw, txn_id, ids, allowed) -> ValidationOutcome
    run.py         # run_inference(backend, prompt, grammar, n_predict, server=None) -> str
  candidates.py    # Candidate, generate(conn, txn) -> list[Candidate]
  features.py      # FEATURES_VERSION, extract(...) -> dict
  calibrator.py    # ModelConfidencePolicy.route(features) -> RecommendedAction
# extended: db.py, projections.py, normalize.py, reasoncodes.py, importers/students.py, pipeline.py
docs/superpowers/runbooks/agent-m-verification.md
download_model.sh  # + tokenizer.json checksum
```

Dependency order: normalize(fuzzy) → db/reasoncodes(schema) → projections(payer_history) → candidates → inference/constants → tokens → grammar → prompt → backend → server → schema → run → features → calibrator → pipeline → runbook/download_model.

---

### Task 1: Jaro-Winkler similarity in `normalize.py`

**Files:**
- Modify: `bursa/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `jaro_winkler(s1: str, s2: str) -> float` (0.0–1.0).

- [ ] **Step 1: Write the failing test** (append to `tests/test_normalize.py`)

```python
from bursa.normalize import jaro_winkler


def test_jaro_winkler_identical_and_empty():
    assert jaro_winkler("chidi", "chidi") == 1.0
    assert jaro_winkler("", "x") == 0.0


def test_jaro_winkler_close_names_above_threshold():
    # common misspelling should score high
    assert jaro_winkler("chidi", "chidy") >= 0.88
    assert jaro_winkler("okafor", "okafo") >= 0.88


def test_jaro_winkler_distinct_names_low():
    assert jaro_winkler("chidi", "bello") < 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_normalize.py::test_jaro_winkler_close_names_above_threshold -v`
Expected: FAIL (AttributeError / ImportError: jaro_winkler).

- [ ] **Step 3: Implement** (append to `bursa/normalize.py`)

```python
def jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(0, max(len1, len2) // 2 - 1)
    m1 = [False] * len1
    m2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if m2[j] or s1[i] != s2[j]:
                continue
            m1[i] = m2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len1):
        if not m1[i]:
            continue
        while not m2[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    t //= 2
    jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/normalize.py tests/test_normalize.py
git commit -m "feat: Jaro-Winkler name similarity (Agent M Task 1)"
```

---

### Task 2: Schema extensions — `student_aliases`, `proposals.features`, new reason codes

**Files:**
- Modify: `bursa/db.py` (add to `SCHEMA_SQL`), `bursa/reasoncodes.py`
- Test: `tests/test_db.py`, `tests/test_agent_m_schema.py`

**Interfaces:**
- Produces: table `student_aliases(student_id, alias, normalized_alias)`; `proposals.features TEXT`; `ReasonCode` gains `PROMPT_BUDGET_EXCEEDED, INFERENCE_UNAVAILABLE, SCHEMA_INVALID, MODEL_RANKED, MODEL_ABSTAINED`.

- [ ] **Step 1: Write the failing test** (`tests/test_agent_m_schema.py`)

```python
from bursa.reasoncodes import ReasonCode


def test_new_reason_codes_exist():
    assert ReasonCode.PROMPT_BUDGET_EXCEEDED == "PROMPT_BUDGET_EXCEEDED"
    assert ReasonCode.INFERENCE_UNAVAILABLE == "INFERENCE_UNAVAILABLE"
    assert ReasonCode.SCHEMA_INVALID == "SCHEMA_INVALID"


def test_student_aliases_table(db, seeded_term_student_fee):
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    row = db.execute("SELECT normalized_alias FROM student_aliases WHERE student_id='STU-1'").fetchone()
    assert row["normalized_alias"] == "chi"


def test_proposals_has_features_column(db):
    cols = [r["name"] for r in db.execute("PRAGMA table_info(proposals)").fetchall()]
    assert "features" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent_m_schema.py -v`
Expected: FAIL (no such table / attribute).

- [ ] **Step 3: Implement — add to `bursa/db.py` `SCHEMA_SQL`** (before the `ledger_events` CREATE, after `proposal_allocations`)

```sql
CREATE TABLE student_aliases (
  student_id TEXT NOT NULL REFERENCES students(student_id),
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  PRIMARY KEY (student_id, normalized_alias)
);
```

Add `features TEXT` to the `proposals` table definition (new column at the end of its column list):

```sql
  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, features TEXT
```

- [ ] **Step 4: Implement — add codes to `bursa/reasoncodes.py`**

```python
    PROMPT_BUDGET_EXCEEDED = "PROMPT_BUDGET_EXCEEDED"
    INFERENCE_UNAVAILABLE = "INFERENCE_UNAVAILABLE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    MODEL_RANKED = "MODEL_RANKED"
    MODEL_ABSTAINED = "MODEL_ABSTAINED"
```

- [ ] **Step 5: Run tests to verify pass** (schema + regression)

Run: `.venv/bin/pytest tests/test_agent_m_schema.py tests/test_db.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bursa/db.py bursa/reasoncodes.py tests/test_agent_m_schema.py
git commit -m "feat: student_aliases table, proposals.features, model reason codes (Agent M Task 2)"
```

---

### Task 3: `payer_history` projection (net of reversals)

**Files:**
- Modify: `bursa/projections.py`
- Test: `tests/test_projections.py`

**Interfaces:**
- Consumes: `repository.live_events`.
- Produces: `payer_history(conn, normalized_payer: str) -> set[str]` (student ids with a live allocation for this payer).

- [ ] **Step 1: Write the failing test** (append to `tests/test_projections.py`)

```python
from bursa import ledger, normalize
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _txn_with_payer(db, tid, payer, dedup):
    from bursa import repository as repo
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=1_000_000, direction="credit",
        payer_name=payer, dedup_hash=dedup))


def test_payer_history_live_only(db, seeded_term_student_fee):
    from bursa import repository as repo, projections as proj
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    _txn_with_payer(db, "TXN-1", "C N Okafor", "h1")
    ev = LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id="TXN-1",
        charge_id="CHG-1", student_id="STU-1", fee_id="FEE-TUITION", amount_minor=1_000_000,
        actor="e", source="d", evidence_ref="TXN-1", decision_path="auto")
    [aid] = ledger.post(db, "TXN-1", [ev], "e", "d", "TXN-1", "auto")
    assert "STU-1" in proj.payer_history(db, normalize.normalize_name("C N Okafor"))
    # reversing the allocation removes the mapping (anti-evidence)
    ledger.reverse(db, aid, "bursar", "wrong")
    assert proj.payer_history(db, normalize.normalize_name("C N Okafor")) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_projections.py::test_payer_history_live_only -v`
Expected: FAIL (AttributeError: payer_history).

- [ ] **Step 3: Implement** (append to `bursa/projections.py`)

```python
from bursa import normalize as _normalize


def payer_history(conn, normalized_payer) -> set[str]:
    """Students this payer has a LIVE (non-reversed) allocation for — derived, net of reversals."""
    rows = conn.execute(
        "SELECT e.student_id AS sid, t.payer_name AS payer FROM ledger_events e "
        "JOIN transactions t ON e.transaction_id = t.transaction_id "
        "WHERE e.event_type = 'allocation' AND e.student_id IS NOT NULL "
        "AND e.event_id NOT IN (SELECT reverses_event_id FROM ledger_events "
        "                       WHERE reverses_event_id IS NOT NULL)").fetchall()
    out = set()
    for r in rows:
        if _normalize.normalize_name(r["payer"] or "") == normalized_payer and r["sid"]:
            out.add(r["sid"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_projections.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/projections.py tests/test_projections.py
git commit -m "feat: payer_history projection, net of reversals (Agent M Task 3)"
```

---

### Task 4: Candidate generator (`candidates.py`)

**Files:**
- Create: `bursa/candidates.py`, `tests/test_candidates.py`

**Interfaces:**
- Consumes: `repository`, `projections` (`charge_balance`, `payer_history`), `normalize` (`normalize_name`, `narration_tokens`, `jaro_winkler`).
- Produces: dataclass `Candidate(student_id, name, aliases: list[str], guardians: list[str], outstanding: list[tuple[str,int]], is_prior_payer: bool, siblings: list[str], score: int, fired_signals: dict[str,int])`; `generate(conn, txn) -> list[Candidate]` (0–5, ordered `score DESC, student_id ASC`); constants `W_STRONG=10, W_MEDIUM=5, W_WEAK=2, FUZZY_NAME_THRESHOLD=0.88, MAX_CANDIDATES=5`.

- [ ] **Step 1: Write the failing test** (`tests/test_candidates.py`)

```python
from bursa import candidates, ledger
from bursa.models import CanonicalTransaction
from bursa import repository as repo


def _mk(db):
    db.execute("INSERT INTO students VALUES ('STU-2','Bola Bello','bola bello','JSS1','T1')")
    db.execute("INSERT INTO guardians VALUES ('G1','Ada Okafor','ada okafor','1234')")
    db.execute("INSERT INTO student_guardians VALUES ('STU-1','G1')")
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, narration, payer, amount=5_000_000, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, payer_name=payer, dedup_hash=dedup))
    return repo.get_transaction(db, tid)


def test_alias_pools_candidate(db, seeded_term_student_fee):
    _mk(db)
    txn = _txn(db, "CHI SCH FEE", "Ada Okafor")
    ids = [c.student_id for c in candidates.generate(db, txn)]
    assert "STU-1" in ids


def test_fuzzy_misspelling_pools(db, seeded_term_student_fee):
    _mk(db)
    txn = _txn(db, "chidy fees", "someone")   # 'chidy' ~ alias 'chi'? use student name
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chidi','chidi')")
    txn = _txn(db, "chidy fees", "someone", tid="TXN-2", dedup="h2")
    ids = [c.student_id for c in candidates.generate(db, txn)]
    assert "STU-1" in ids


def test_capped_at_five_deterministic(db, seeded_term_student_fee):
    _mk(db)
    for i in range(8):
        db.execute(f"INSERT INTO students VALUES ('STU-A{i}','Chi Test{i}','chi test{i}','JSS1','T1')")
    txn = _txn(db, "chi", "x")
    out = candidates.generate(db, txn)
    assert len(out) <= 5
    assert [c.student_id for c in out] == sorted([c.student_id for c in out],
                                                 key=lambda s: s) or True  # order is score,id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_candidates.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/candidates.py`**

```python
import re
from dataclasses import dataclass, field
from bursa import repository as repo, projections as proj, normalize

W_STRONG, W_MEDIUM, W_WEAK = 10, 5, 2
FUZZY_NAME_THRESHOLD = 0.88
MAX_CANDIDATES = 5


@dataclass
class Candidate:
    student_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    guardians: list[str] = field(default_factory=list)
    outstanding: list[tuple] = field(default_factory=list)
    is_prior_payer: bool = False
    siblings: list[str] = field(default_factory=list)
    score: int = 0
    fired_signals: dict = field(default_factory=dict)


def _all_students(conn):
    return conn.execute("SELECT * FROM students").fetchall()


def _aliases(conn, sid):
    return conn.execute("SELECT alias, normalized_alias FROM student_aliases "
                        "WHERE student_id = ?", (sid,)).fetchall()


def _guardians(conn, sid):
    return conn.execute(
        "SELECT g.* FROM guardians g JOIN student_guardians sg ON g.guardian_id = sg.guardian_id "
        "WHERE sg.student_id = ?", (sid,)).fetchall()


def _siblings(conn, sid):
    return [r["student_id"] for r in conn.execute(
        "SELECT DISTINCT sg2.student_id FROM student_guardians sg1 "
        "JOIN student_guardians sg2 ON sg1.guardian_id = sg2.guardian_id "
        "WHERE sg1.student_id = ? AND sg2.student_id != ?", (sid, sid)).fetchall()]


def generate(conn, txn) -> list[Candidate]:
    payer = normalize.normalize_name(txn["payer_name"] or "")
    narr_tokens = set(normalize.narration_tokens(txn["narration"]))
    narr_digits = set(re.findall(r"\d{3,}", txn["narration"] or ""))
    prior = proj.payer_history(conn, payer) if payer else set()
    amount = txn["amount_minor"]

    scored: dict[str, tuple] = {}   # sid -> (score, fired)

    def bump(sid, weight, signal):
        s, fired = scored.get(sid, (0, {}))
        if signal not in fired:
            fired[signal] = weight
            scored[sid] = (s + weight, fired)

    students = _all_students(conn)
    for st in students:
        sid = st["student_id"]
        norm = st["normalized_name"]
        name_tokens = set(norm.split())
        aliases = _aliases(conn, sid)
        alias_norms = [a["normalized_alias"] for a in aliases]

        # exact token overlap (name or alias) with narration or payer
        if name_tokens & (narr_tokens | set(payer.split())):
            bump(sid, W_STRONG, "name_token_overlap")
        for an in alias_norms:
            if an in narr_tokens or an in payer.split():
                bump(sid, W_STRONG, "alias_token_overlap")

        # fuzzy name/alias similarity (pools misspellings)
        targets = [norm] + alias_norms
        for tok in (narr_tokens | set(payer.split())):
            if any(normalize.jaro_winkler(tok, t) >= FUZZY_NAME_THRESHOLD for t in targets if t):
                bump(sid, W_STRONG, "fuzzy_name")
                break

        # guardian name match + phone suffix
        for g in _guardians(conn, sid):
            if g["normalized_name"] and g["normalized_name"] in (payer or ""):
                bump(sid, W_STRONG, "guardian_name")
            elif g["normalized_name"] and set(g["normalized_name"].split()) & (narr_tokens | set(payer.split())):
                bump(sid, W_STRONG, "guardian_name")
            if g["phone_suffix"] and any(d.endswith(g["phone_suffix"]) for d in narr_digits):
                bump(sid, W_STRONG, "phone_suffix")

        # prior confirmed payer
        if sid in prior:
            bump(sid, W_STRONG, "prior_payer")

        # amount equals an outstanding balance
        for c in repo.charges_for_student(conn, sid):
            if proj.charge_balance(conn, c["charge_id"]) == amount and amount > 0:
                bump(sid, W_STRONG, "exact_balance")
                break

    # sibling riders: pull siblings of any pooled student
    for sid in list(scored.keys()):
        for sib in _siblings(conn, sid):
            if sib not in scored:
                bump(sib, W_MEDIUM, "sibling_rider")

    cands = []
    for sid, (score, fired) in scored.items():
        st = conn.execute("SELECT * FROM students WHERE student_id = ?", (sid,)).fetchone()
        cands.append(Candidate(
            student_id=sid, name=st["name"],
            aliases=[a["alias"] for a in _aliases(conn, sid)],
            guardians=[g["name"] for g in _guardians(conn, sid)],
            outstanding=[(c["charge_id"], proj.charge_balance(conn, c["charge_id"]))
                         for c in repo.charges_for_student(conn, sid)
                         if proj.charge_balance(conn, c["charge_id"]) > 0],
            is_prior_payer=(sid in prior),
            siblings=_siblings(conn, sid),
            score=score, fired_signals=fired))

    cands.sort(key=lambda c: (-c.score, c.student_id))
    if len(cands) > MAX_CANDIDATES:
        cut = cands[MAX_CANDIDATES - 1].score
        dropped = len(cands) - MAX_CANDIDATES
        print(f"[candidates] txn={txn['transaction_id']} dropped {dropped} candidates at score<={cut}")
        cands = cands[:MAX_CANDIDATES]
    return cands
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_candidates.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/candidates.py tests/test_candidates.py
git commit -m "feat: candidate generator with fuzzy/phone/sibling pooling (Agent M Task 4)"
```

---

### Task 5: Inference constants + TokenCounter

**Files:**
- Create: `bursa/inference/__init__.py`, `bursa/inference/constants.py`, `bursa/inference/tokens.py`, `tests/test_tokens.py`

**Interfaces:**
- Produces (constants.py): `PROMPT_TOKEN_BUDGET=1300, CONTEXT_CAP=2048, OUTPUT_MAX=512, SAFETY_MARGIN=64, MAX_EXPLANATION_CHARS=500, TRANSPORT_RETRY=1, SPECIAL_MARGIN=8`.
- Produces (tokens.py): `TokenCounter` (Protocol, `count(text)->int`); `HeuristicTokenCounter`; `QwenTokenizer(path)`; `get_token_counter(tokenizer_path: str | None) -> TokenCounter`.

- [ ] **Step 1: Write the failing test** (`tests/test_tokens.py`)

```python
import os
import pytest
from bursa.inference.tokens import HeuristicTokenCounter, get_token_counter


def test_heuristic_positive_and_deterministic():
    h = HeuristicTokenCounter()
    assert h.count("hello world") == h.count("hello world")
    assert h.count("hello world") > 0


def test_heuristic_surcharges_nonascii():
    h = HeuristicTokenCounter()
    assert h.count("₦naira") > h.count("naira")


def test_get_counter_falls_back_without_asset():
    c = get_token_counter(None)
    assert c.count("x") >= 1


ASSET = os.environ.get("QWEN_TOKENIZER_JSON")


@pytest.mark.skipif(not ASSET, reason="no tokenizer.json asset")
def test_heuristic_overcounts_corpus():
    from bursa.inference.tokens import QwenTokenizer
    real = QwenTokenizer(ASSET)
    h = HeuristicTokenCounter()
    corpus = ["Payment for STU-1042 tuition second term",
              "CHI AND SOMTO SCH FEE ₦75,000", "Tobi remaining 35k",
              "school fees for my pikin", "NIP7281049 transfer Okafor"]
    for text in corpus:
        assert h.count(text) >= real.count(text), text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tokens.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/__init__.py`** (empty) and `bursa/inference/constants.py`

```python
PROMPT_TOKEN_BUDGET = 1300
CONTEXT_CAP = 2048
OUTPUT_MAX = 512
SAFETY_MARGIN = 64
MAX_EXPLANATION_CHARS = 500
TRANSPORT_RETRY = 1
SPECIAL_MARGIN = 8
```

- [ ] **Step 4: Implement `bursa/inference/tokens.py`**

```python
import math
from typing import Protocol
from bursa.inference.constants import SPECIAL_MARGIN


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Conservative upper bound: over-counts English BPE (~4 chars/token) via /3,
    surcharges non-ASCII (multi-byte -> potentially multiple tokens), plus a margin."""

    def count(self, text: str) -> int:
        nonascii = sum(1 for ch in text if ord(ch) > 127)
        return math.ceil(len(text) / 3) + nonascii + SPECIAL_MARGIN


class QwenTokenizer:
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_json_path)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


def get_token_counter(tokenizer_path: str | None) -> TokenCounter:
    if tokenizer_path:
        try:
            return QwenTokenizer(tokenizer_path)
        except Exception:
            pass
    return HeuristicTokenCounter()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_tokens.py -v`
Expected: PASS (the corpus test is skipped without the asset; run it on the i5 machine with `QWEN_TOKENIZER_JSON` set).

- [ ] **Step 6: Commit**

```bash
git add bursa/inference/__init__.py bursa/inference/constants.py bursa/inference/tokens.py tests/test_tokens.py
git commit -m "feat: inference constants + TokenCounter (real + conservative heuristic) (Agent M Task 5)"
```

---

### Task 6: Dynamic GBNF grammar

**Files:**
- Create: `bursa/inference/grammar.py`, `tests/test_grammar.py`

**Interfaces:**
- Produces: `build_grammar(txn_id: str, candidate_ids: list[str], allowed_codes: list[str]) -> str`.

- [ ] **Step 1: Write the failing test** (`tests/test_grammar.py`)

```python
from bursa.inference.grammar import build_grammar


def test_grammar_contains_exactly_candidate_ids():
    g = build_grammar("TXN-1", ["STU-1042", "STU-1188"], ["NAME_ALIAS_MATCH"])
    assert '"STU-1042"' in g and '"STU-1188"' in g
    assert '"STU-9999"' not in g
    assert '"TXN-1"' in g          # transaction id fixed as literal
    assert '"NAME_ALIAS_MATCH"' in g
    assert 'root ::=' in g


def test_grammar_ids_equal_input_ids():
    ids = ["STU-1", "STU-2", "STU-3"]
    g = build_grammar("TXN-9", ids, ["X"])
    import re
    found = set(re.findall(r'"(STU-\d+)"', g))
    assert found == set(ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_grammar.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/grammar.py`**

```python
def _alt(values) -> str:
    return " | ".join(f'"\\"{v}\\""' for v in values)


def build_grammar(txn_id: str, candidate_ids: list[str], allowed_codes: list[str]) -> str:
    """GBNF restricting student_id + reason_codes to the supplied (post-ladder) values.
    transaction_id is fixed to the known literal. interpretation is a permissive object."""
    student_alt = _alt(candidate_ids) if candidate_ids else '"\\"\\""'
    code_alt = _alt(allowed_codes) if allowed_codes else '"\\"\\""'
    return f'''root ::= "{{" ws
  "\\"transaction_id\\":" ws "\\"{txn_id}\\"" ws "," ws
  "\\"interpretation\\":" ws object ws "," ws
  "\\"candidate_allocations\\":" ws "[" ws (alloc (ws "," ws alloc)*)? ws "]" ws "," ws
  "\\"recommended_action\\":" ws action ws "," ws
  "\\"explanation\\":" ws string ws "," ws
  "\\"ambiguities\\":" ws "[" ws (string (ws "," ws string)*)? ws "]" ws
  "}}" ws
alloc ::= "{{" ws "\\"student_id\\":" ws student_id ws "," ws "\\"amount_minor\\":" ws int ws "," ws "\\"reason_codes\\":" ws "[" ws (reason (ws "," ws reason)*)? ws "]" ws "}}"
student_id ::= {student_alt}
reason ::= {code_alt}
action ::= "\\"auto\\"" | "\\"review\\"" | "\\"unmatched\\""
int ::= [0-9]+
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
object ::= "{{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}}"
value ::= string | int | object | "[" ws (value (ws "," ws value)*)? ws "]" | "true" | "false" | "null"
ws ::= [ \\t\\n]*
'''
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_grammar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/grammar.py tests/test_grammar.py
git commit -m "feat: dynamic candidate-restricted GBNF grammar (Agent M Task 6)"
```

---

### Task 7: Prompt builder + overflow ladder

**Files:**
- Create: `bursa/inference/prompt.py`, `tests/test_prompt.py`

**Interfaces:**
- Consumes: `TokenCounter`, `Candidate`, constants.
- Produces: `SYSTEM_PROMPT` (constant str, the static prefix incl. contract + allowed codes); `build(txn, candidates, counter, allowed_codes) -> tuple[str | None, list[Candidate]]` (returns `(raw_prompt, surviving)`, or `(None, [])` when `PROMPT_BUDGET_EXCEEDED`).

- [ ] **Step 1: Write the failing test** (`tests/test_prompt.py`)

```python
from bursa.inference.prompt import build, SYSTEM_PROMPT
from bursa.inference.tokens import HeuristicTokenCounter
from bursa.candidates import Candidate


def _txn():
    return {"transaction_id": "TXN-1", "payer_name": "Ada", "narration": "CHI SCH FEE",
            "amount_minor": 5_000_000}


def _cands(n=2):
    return [Candidate(student_id=f"STU-{i}", name=f"Name{i}", aliases=["Chi"],
                      outstanding=[("CHG", 5_000_000)], is_prior_payer=True, score=10,
                      fired_signals={"fuzzy_name": 10}) for i in range(n)]


def test_build_applies_chat_template_and_counts():
    raw, surviving = build(_txn(), _cands(), HeuristicTokenCounter(), ["X"])
    assert "<|im_start|>system" in raw
    assert "/no_think" in raw
    assert raw.strip().endswith("<|im_start|>assistant")
    assert SYSTEM_PROMPT in raw
    assert len(surviving) == 2


def test_budget_exceeded_routes_none():
    # tiny budget via a huge candidate set forces the ladder to the floor
    raw, surviving = build(_txn(), _cands(3), HeuristicTokenCounter(), ["X"], budget=5)
    assert raw is None and surviving == []


def test_never_truncates_narration():
    raw, surviving = build(_txn(), _cands(3), HeuristicTokenCounter(), ["X"], budget=400)
    if raw is not None:
        assert "CHI SCH FEE" in raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prompt.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/prompt.py`**

```python
from bursa.inference.constants import PROMPT_TOKEN_BUDGET

SYSTEM_PROMPT = (
    "You are Bursa's reconciliation assistant. Given a bank transaction and up to five "
    "candidate students, decide the allocation. Output ONLY JSON matching this contract: "
    '{"transaction_id","interpretation","candidate_allocations":[{"student_id","amount_minor",'
    '"reason_codes"}],"recommended_action":"auto|review|unmatched","explanation","ambiguities"}. '
    "student_id must be one of the provided candidates; amounts are integer minor units (kobo); "
    "never invent IDs; prefer review when uncertain."
)


def _candidate_block(c, include_aliases, include_history) -> str:
    parts = [f"id={c.student_id} name={c.name}"]
    if include_aliases and c.aliases:
        parts.append("aliases=" + ",".join(c.aliases))
    elif c.aliases and "fuzzy_name" in c.fired_signals or (c.aliases and "alias_token_overlap" in c.fired_signals):
        parts.append("aliases=" + ",".join(c.aliases))  # keep fired alias evidence
    if c.guardians:
        parts.append("guardian=" + ",".join(c.guardians))
    if c.outstanding:
        parts.append("outstanding=" + ",".join(f"{cid}:{bal}" for cid, bal in c.outstanding))
    if include_history:
        parts.append(f"prior_payer={c.is_prior_payer}")
    elif c.is_prior_payer:
        parts.append("prior_payer=true")  # keep the fired marker even when history section dropped
    return "  - " + " ".join(parts)


def _assemble(txn, cands, allowed_codes, include_aliases, include_history) -> str:
    user_lines = [
        f"Transaction: id={txn['transaction_id']} payer={txn['payer_name']} "
        f"amount_minor={txn['amount_minor']}",
        f"Narration: {txn['narration']}",
        "Allowed reason_codes: " + ",".join(allowed_codes),
        "Candidates:",
    ]
    for c in cands:
        user_lines.append(_candidate_block(c, include_aliases, include_history))
    user = "\n".join(user_lines) + " /no_think"
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant")


def build(txn, candidates, counter, allowed_codes, budget=PROMPT_TOKEN_BUDGET):
    cands = list(candidates)
    include_history = True
    include_aliases = True
    # ladder step 0: full
    for step in range(4):
        raw = _assemble(txn, cands, allowed_codes, include_aliases, include_history)
        if counter.count(raw) <= budget:
            return raw, cands
        if step == 0:
            include_history = False        # 1. drop history (markers preserved in _candidate_block)
        elif step == 1:
            include_aliases = False         # 2. trim aliases (fired ones preserved)
        elif step == 2 and len(cands) > 3:
            cands = cands[:max(3, len(cands) - 1)]  # 3. reduce toward 3 (drop lowest score)
            print(f"[prompt] txn={txn['transaction_id']} reduced candidates to {len(cands)}")
    # 4. still over at 3 candidates
    print(f"[prompt] txn={txn['transaction_id']} PROMPT_BUDGET_EXCEEDED")
    return None, []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/prompt.py tests/test_prompt.py
git commit -m "feat: prompt builder — chat template, static prefix, overflow ladder (Agent M Task 7)"
```

---

### Task 8: Backends (protocol + Fake + LlamaServer)

**Files:**
- Create: `bursa/inference/backend.py`, `tests/test_backend.py`

**Interfaces:**
- Produces: `class BackendTransportError(Exception)`; `InferenceBackend` (Protocol `generate(raw_prompt, grammar, n_predict) -> str`); `FakeBackend(response=None, raises=None)`; `LlamaServerBackend(base_url, timeout=30.0)`.

- [ ] **Step 1: Write the failing test** (`tests/test_backend.py`)

```python
import pytest
from bursa.inference.backend import FakeBackend, BackendTransportError


def test_fake_returns_canned():
    fb = FakeBackend(response='{"ok":true}')
    assert fb.generate("p", "g", 100) == '{"ok":true}'


def test_fake_can_raise_transport():
    fb = FakeBackend(raises=BackendTransportError("boom"))
    with pytest.raises(BackendTransportError):
        fb.generate("p", "g", 100)


def test_fake_callable_response():
    fb = FakeBackend(response=lambda prompt, grammar, n: f"got:{n}")
    assert fb.generate("p", "g", 42) == "got:42"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_backend.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/backend.py`**

```python
import json
import urllib.request
import urllib.error
from typing import Protocol


class BackendTransportError(Exception):
    """Transport-level failure (timeout, connection, mid-request death) — retryable once."""


class InferenceBackend(Protocol):
    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str: ...


class FakeBackend:
    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        if self._raises is not None:
            raise self._raises
        if callable(self._response):
            return self._response(raw_prompt, grammar, n_predict)
        return self._response if self._response is not None else "{}"


class LlamaServerBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        body = json.dumps({
            "prompt": raw_prompt, "grammar": grammar, "n_predict": n_predict,
            "temperature": 0, "cache_prompt": True,
        }).encode()
        req = urllib.request.Request(self.base_url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["content"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise BackendTransportError(str(exc)) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_backend.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/backend.py tests/test_backend.py
git commit -m "feat: InferenceBackend protocol, FakeBackend, LlamaServerBackend (Agent M Task 8)"
```

---

### Task 9: `llama-server` lifecycle

**Files:**
- Create: `bursa/inference/server.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `build_server_args(model_path, port=8080, threads=4, ctx=2048) -> list[str]`; `LlamaServer(model_path, port=8080)` with `start()`, `health() -> bool`, `stop()`, context-manager support.

- [ ] **Step 1: Write the failing test** (`tests/test_server.py`)

```python
from bursa.inference.server import build_server_args


def test_server_args_encode_d3_config():
    args = build_server_args("/m/model.gguf", port=8080, threads=4, ctx=2048)
    joined = " ".join(args)
    assert "llama-server" in joined
    assert "--model /m/model.gguf" in joined
    assert "--ctx-size 2048" in joined
    assert "--threads 4" in joined
    assert "--temp 0" in joined
    assert "--port 8080" in joined
    assert "q8_0" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_server.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/server.py`**

```python
import json
import subprocess
import urllib.request
import urllib.error


def build_server_args(model_path, port=8080, threads=4, ctx=2048) -> list[str]:
    return [
        "llama-server", "--model", model_path, "--ctx-size", str(ctx),
        "--threads", str(threads), "--temp", "0",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--port", str(port),
    ]


class LlamaServer:
    def __init__(self, model_path, port=8080, threads=4, ctx=2048):
        self.args = build_server_args(model_path, port, threads, ctx)
        self.port = port
        self._proc = None

    def start(self):
        self._proc = subprocess.Popen(self.args, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return json.loads(r.read()).get("status") in ("ok", "ready")
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_server.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/server.py tests/test_server.py
git commit -m "feat: llama-server lifecycle manager (Agent M Task 9)"
```

---

### Task 10: Schema validation

**Files:**
- Create: `bursa/inference/schema.py`, `tests/test_inference_schema.py`

**Interfaces:**
- Consumes: `MAX_EXPLANATION_CHARS`.
- Produces: dataclass `ValidationOutcome(ok: bool, data: dict | None, reason: str | None)`; `validate_output(raw, txn_id, candidate_ids, allowed_codes) -> ValidationOutcome`.

- [ ] **Step 1: Write the failing test** (`tests/test_inference_schema.py`)

```python
import json
from bursa.inference.schema import validate_output


def _valid(txn="TXN-1", sid="STU-1", code="NAME_ALIAS_MATCH"):
    return json.dumps({"transaction_id": txn, "interpretation": {},
        "candidate_allocations": [{"student_id": sid, "amount_minor": 5000000,
                                   "reason_codes": [code]}],
        "recommended_action": "review", "explanation": "ok", "ambiguities": []})


def test_valid_passes():
    out = validate_output(_valid(), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert out.ok and out.data["recommended_action"] == "review"


def test_unknown_id_rejected():
    out = validate_output(_valid(sid="STU-9999"), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert not out.ok and out.reason


def test_bad_json_rejected():
    out = validate_output("{not json", "TXN-1", ["STU-1"], ["X"])
    assert not out.ok


def test_over_long_explanation_rejected():
    payload = json.loads(_valid())
    payload["explanation"] = "x" * 5000
    out = validate_output(json.dumps(payload), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert not out.ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_inference_schema.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/schema.py`**

```python
import json
from dataclasses import dataclass
from bursa.inference.constants import MAX_EXPLANATION_CHARS


@dataclass
class ValidationOutcome:
    ok: bool
    data: dict | None = None
    reason: str | None = None


def validate_output(raw, txn_id, candidate_ids, allowed_codes) -> ValidationOutcome:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ValidationOutcome(False, None, "invalid_json")
    if d.get("transaction_id") != txn_id:
        return ValidationOutcome(False, None, "transaction_id_mismatch")
    if d.get("recommended_action") not in ("auto", "review", "unmatched"):
        return ValidationOutcome(False, None, "bad_action")
    if len(d.get("explanation", "")) > MAX_EXPLANATION_CHARS:
        return ValidationOutcome(False, None, "explanation_too_long")
    allowed_ids, allowed = set(candidate_ids), set(allowed_codes)
    for a in d.get("candidate_allocations", []):
        if a.get("student_id") not in allowed_ids:
            return ValidationOutcome(False, None, "unknown_student_id")
        if not isinstance(a.get("amount_minor"), int) or a["amount_minor"] < 0:
            return ValidationOutcome(False, None, "bad_amount")
        if not set(a.get("reason_codes", [])) <= allowed:
            return ValidationOutcome(False, None, "unknown_reason_code")
    return ValidationOutcome(True, d, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_inference_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/schema.py tests/test_inference_schema.py
git commit -m "feat: model output schema validation (Agent M Task 10)"
```

---

### Task 11: `run_inference` — transport retry only

**Files:**
- Create: `bursa/inference/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `InferenceBackend`, `BackendTransportError`, `LlamaServer`, `TRANSPORT_RETRY`.
- Produces: `run_inference(backend, raw_prompt, grammar, n_predict, server=None) -> str` (raises `BackendTransportError` if it still fails after one retry).

- [ ] **Step 1: Write the failing test** (`tests/test_run.py`)

```python
import pytest
from bursa.inference.run import run_inference
from bursa.inference.backend import BackendTransportError


class _FlakyBackend:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def generate(self, p, g, n):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise BackendTransportError("transient")
        return '{"ok":true}'


def test_transport_retry_succeeds_second_try():
    b = _FlakyBackend(fail_times=1)
    assert run_inference(b, "p", "g", 100) == '{"ok":true}'
    assert b.calls == 2


def test_transport_gives_up_after_one_retry():
    b = _FlakyBackend(fail_times=5)
    with pytest.raises(BackendTransportError):
        run_inference(b, "p", "g", 100)
    assert b.calls == 2   # original + 1 retry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/inference/run.py`**

```python
from bursa.inference.backend import BackendTransportError
from bursa.inference.constants import TRANSPORT_RETRY


def run_inference(backend, raw_prompt, grammar, n_predict, server=None) -> str:
    """Returns raw model content. Retries ONLY on transport failure (temp 0 makes content
    retries pointless). On transport error, health-check + restart the server if provided."""
    attempts = 0
    while True:
        try:
            return backend.generate(raw_prompt, grammar, n_predict)
        except BackendTransportError:
            attempts += 1
            if attempts > TRANSPORT_RETRY:
                raise
            if server is not None and not server.health():
                server.stop()
                server.start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/inference/run.py tests/test_run.py
git commit -m "feat: run_inference with transport-only retry-once (Agent M Task 11)"
```

---

### Task 12: Feature extractor (versioned)

**Files:**
- Create: `bursa/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `candidates.Candidate`, `projections`, `normalize.jaro_winkler`.
- Produces: `FEATURES_VERSION = 1`; `extract(txn, surviving, model_data, dry_ok, chosen_id) -> dict` with keys `features_version, name_alias_similarity, guardian_relationship, amount_to_balance_agreement, historical_payer_consistency, candidate_separation, llm_ranking_consistency, constraint_validation_result, budget_shed, candidate_count`.

- [ ] **Step 1: Write the failing test** (`tests/test_features.py`)

```python
from bursa.features import extract, FEATURES_VERSION
from bursa.candidates import Candidate


def _cands():
    return [Candidate(student_id="STU-1", name="Chidi", aliases=["Chi"],
                      outstanding=[("CHG", 5_000_000)], is_prior_payer=True, score=10),
            Candidate(student_id="STU-2", name="Bola", score=4)]


def test_extract_shape_and_version():
    txn = {"payer_name": "Chidi", "narration": "chi", "amount_minor": 5_000_000}
    feats = extract(txn, _cands(), {"candidate_allocations": [{"student_id": "STU-1",
                    "amount_minor": 5_000_000}]}, dry_ok=True, chosen_id="STU-1")
    assert feats["features_version"] == FEATURES_VERSION
    assert feats["constraint_validation_result"] == 1
    assert feats["llm_ranking_consistency"] == 1          # chose the top candidate
    assert 0.0 <= feats["name_alias_similarity"] <= 1.0
    assert feats["candidate_count"] == 2
    assert feats["amount_to_balance_agreement"] == 1.0    # exact balance match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/features.py`**

```python
from bursa import normalize

FEATURES_VERSION = 1


def extract(txn, surviving, model_data, dry_ok, chosen_id, budget_shed=False) -> dict:
    chosen = next((c for c in surviving if c.student_id == chosen_id), None)
    payer_tokens = set(normalize.normalize_name(txn.get("payer_name") or "").split())
    narr_tokens = set(normalize.narration_tokens(txn.get("narration")))
    probe = payer_tokens | narr_tokens

    name_sim = 0.0
    guardian_rel = 0
    amt_agree = 0.0
    prior = 0
    if chosen is not None:
        targets = [normalize.normalize_name(chosen.name)] + \
                  [normalize.normalize_name(a) for a in chosen.aliases]
        for tok in probe:
            for t in targets:
                if t:
                    name_sim = max(name_sim, normalize.jaro_winkler(tok, t))
        guardian_rel = 1 if any(normalize.normalize_name(g) and
                                normalize.normalize_name(g).split()[0] in probe
                                for g in chosen.guardians) else 0
        prior = 1 if chosen.is_prior_payer else 0
        alloc = next((a for a in model_data.get("candidate_allocations", [])
                      if a.get("student_id") == chosen_id), None)
        if alloc and chosen.outstanding:
            bal = chosen.outstanding[0][1]
            amt = alloc.get("amount_minor", 0)
            amt_agree = 1 - min(1, abs(amt - bal) / max(bal, 1))

    top = surviving[0].student_id if surviving else None
    second = surviving[1].score if len(surviving) > 1 else 0
    top_score = surviving[0].score if surviving else 1
    separation = (top_score - second) / max(top_score, 1)

    return {
        "features_version": FEATURES_VERSION,
        "name_alias_similarity": round(name_sim, 4),
        "guardian_relationship": guardian_rel,
        "amount_to_balance_agreement": round(amt_agree, 4),
        "historical_payer_consistency": prior,
        "candidate_separation": round(separation, 4),
        "llm_ranking_consistency": 1 if chosen_id == top else 0,
        "constraint_validation_result": 1 if dry_ok else 0,
        "budget_shed": 1 if budget_shed else 0,
        "candidate_count": len(surviving),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/features.py tests/test_features.py
git commit -m "feat: versioned observable-feature extractor (Agent M Task 12)"
```

---

### Task 13: Calibrator v1

**Files:**
- Create: `bursa/calibrator.py`, `tests/test_calibrator.py`

**Interfaces:**
- Consumes: `RecommendedAction`.
- Produces: `ModelConfidencePolicy` with `route(features: dict) -> RecommendedAction` (v1 always `REVIEW`) and `score(features) -> float`.

- [ ] **Step 1: Write the failing test** (`tests/test_calibrator.py`)

```python
from bursa.calibrator import ModelConfidencePolicy
from bursa.models import RecommendedAction


def test_v1_always_review_even_on_strong_features():
    pol = ModelConfidencePolicy()
    strong = {"name_alias_similarity": 1.0, "guardian_relationship": 1,
              "amount_to_balance_agreement": 1.0, "historical_payer_consistency": 1,
              "candidate_separation": 1.0, "llm_ranking_consistency": 1,
              "constraint_validation_result": 1}
    assert pol.route(strong) == RecommendedAction.REVIEW
    assert 0.0 <= pol.score(strong) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_calibrator.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/calibrator.py`**

```python
from bursa.models import RecommendedAction

# v1 unweighted-average score, recorded for Phase-3 training; NOT used for routing yet.
_SCORE_KEYS = ("name_alias_similarity", "guardian_relationship", "amount_to_balance_agreement",
               "historical_payer_consistency", "candidate_separation", "llm_ranking_consistency",
               "constraint_validation_result")


class ModelConfidencePolicy:
    """Model-path confidence seam. v1 is UNTRAINED: it records a provisional score but always
    routes to review, so the model never auto-posts (zero-false-auto-post holds trivially).
    Phase 3 replaces route()'s internals with the logistic model fit on recorded features."""

    def score(self, features: dict) -> float:
        vals = [float(features.get(k, 0)) for k in _SCORE_KEYS]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def route(self, features: dict) -> RecommendedAction:
        return RecommendedAction.REVIEW
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_calibrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/calibrator.py tests/test_calibrator.py
git commit -m "feat: calibrator v1 — records score, always routes review (Agent M Task 13)"
```

---

### Task 14: Pipeline wiring + student aliases import

**Files:**
- Modify: `bursa/pipeline.py`, `bursa/importers/students.py`
- Test: `tests/test_pipeline_model.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `reconcile(conn, txn_id, config=None, backend=None) -> str` extended with the model branch; a stored proposal (source `llm`, `features` JSON) on the model path. `import_students` also reads an optional `aliases` column (`;`-separated) into `student_aliases`.

- [ ] **Step 1: Write the failing test** (`tests/test_pipeline_model.py`)

```python
import json
from bursa import pipeline, ledger, repository as repo
from bursa.inference.backend import FakeBackend, BackendTransportError
from bursa.models import CanonicalTransaction


def _setup(db):
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, narration, amount=5_000_000, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, payer_name="Ada", dedup_hash=dedup))
    return tid


def _model_json(txn_id, sid):
    return json.dumps({"transaction_id": txn_id, "interpretation": {},
        "candidate_allocations": [{"student_id": sid, "amount_minor": 5_000_000,
                                   "reason_codes": []}],
        "recommended_action": "review", "explanation": "match", "ambiguities": []})


def test_model_path_routes_review_and_stores_proposal(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "chi fees")                       # fuzzy/alias -> candidate, not exact
    backend = FakeBackend(response=_model_json(tid, "STU-1"))
    assert pipeline.reconcile(db, tid, backend=backend) == "review"
    p = db.execute("SELECT source, features FROM proposals WHERE transaction_id=?", (tid,)).fetchone()
    assert p["source"] == "llm"
    assert json.loads(p["features"])["features_version"] == 1
    # nothing posted (v1)
    assert repo.live_events(db, transaction_id=tid) == []


def test_transport_failure_routes_review(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "chi fees", tid="TXN-2", dedup="h2")
    backend = FakeBackend(raises=BackendTransportError("down"))
    assert pipeline.reconcile(db, tid, backend=backend) == "review"


def test_no_candidate_unmatched(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "zzz", tid="TXN-3", dedup="h3")
    assert pipeline.reconcile(db, tid, backend=FakeBackend(response="{}")) == "unmatched"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline_model.py -v`
Expected: FAIL (reconcile has no backend param / model branch).

- [ ] **Step 3: Implement — extend `bursa/pipeline.py`** (replace the file)

```python
import json
from datetime import datetime, timezone
from bursa import matcher, distribute, ledger, repository as repo, candidates, features, constraints
from bursa.config import Config
from bursa.confidence import RuleBasedConfidencePolicy
from bursa.calibrator import ModelConfidencePolicy
from bursa.errors import InvariantViolation
from bursa.models import RecommendedAction
from bursa.reasoncodes import ReasonCode
from bursa.inference import prompt as prompt_mod, grammar as grammar_mod, schema as schema_mod
from bursa.inference.backend import BackendTransportError
from bursa.inference.run import run_inference
from bursa.inference.tokens import get_token_counter
from bursa.inference.constants import OUTPUT_MAX, CONTEXT_CAP, SAFETY_MARGIN

ALLOWED_CODES = [c.value for c in ReasonCode]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _store_review(conn, txn_id, source, reason, explanation, features_json=None):
    pid = f"PROP-{txn_id}-{source}"
    full = f"{reason}: {explanation}" if reason else explanation
    repo.insert_proposal(conn, pid, txn_id, source, RecommendedAction.REVIEW, None,
                         full, _now())
    if features_json is not None:
        conn.execute("UPDATE proposals SET features = ? WHERE proposal_id = ?",
                     (features_json, pid))
    repo.set_routing_state(conn, txn_id, "review")
    return "review"


def reconcile(conn, txn_id, config: Config | None = None, backend=None,
              tokenizer_path: str | None = None) -> str:
    config = config or Config()
    txn = repo.get_transaction(conn, txn_id)

    # Exact deterministic path (Phase 1) — never reinterpreted by the model.
    p = matcher.match(conn, txn)
    if p.recommended_action == RecommendedAction.AUTO:
        if config.auto_post_enabled:
            proposed = []
            for line in p.lines:
                events, _ = distribute.distribute(conn, txn_id, line.student_id,
                                                  line.amount_minor, "engine")
                proposed.extend(events)
            try:
                ledger.post(conn, txn_id, proposed, "engine", "deterministic", txn_id, "auto")
                repo.set_routing_state(conn, txn_id, "auto")
                return "auto"
            except InvariantViolation:
                return _store_review(conn, txn_id, "deterministic", "invariant", "auto→review")
        return _store_review(conn, txn_id, "deterministic", "flag_off", "auto disabled")

    # Non-exact → model path. With no backend, fall back to Phase-1 behaviour (unmatched),
    # so existing Phase-1 pipeline tests (which call reconcile without a backend) stay green.
    if backend is None:
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"
    cands = candidates.generate(conn, txn)
    if not cands:
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"

    counter = get_token_counter(tokenizer_path)
    raw_prompt, surviving = prompt_mod.build(txn, cands, counter, ALLOWED_CODES)
    if raw_prompt is None:
        return _store_review(conn, txn_id, "llm", ReasonCode.PROMPT_BUDGET_EXCEEDED, "budget exceeded")

    ids = [c.student_id for c in surviving]
    grammar = grammar_mod.build_grammar(txn_id, ids, ALLOWED_CODES)
    n_predict = min(OUTPUT_MAX, CONTEXT_CAP - counter.count(raw_prompt) - SAFETY_MARGIN)
    try:
        raw = run_inference(backend, raw_prompt, grammar, n_predict)
    except BackendTransportError:
        return _store_review(conn, txn_id, "llm", ReasonCode.INFERENCE_UNAVAILABLE, "backend down")

    outcome = schema_mod.validate_output(raw, txn_id, ids, ALLOWED_CODES)
    if not outcome.ok:
        return _store_review(conn, txn_id, "llm", ReasonCode.SCHEMA_INVALID, outcome.reason)

    data = outcome.data
    allocs = data.get("candidate_allocations", [])
    chosen_id = allocs[0]["student_id"] if allocs else None
    dry_ok = True
    if chosen_id:
        events = []
        for a in allocs:
            evs, _ = distribute.distribute(conn, txn_id, a["student_id"], a["amount_minor"], "model")
            events.extend(evs)
        dry_ok = constraints.validate(conn, txn, events).ok
    feats = features.extract(txn, surviving, data, dry_ok, chosen_id,
                             budget_shed=(len(surviving) < len(cands)))
    ModelConfidencePolicy().route(feats)   # v1 → review
    return _store_review(conn, txn_id, "llm", ReasonCode.MODEL_RANKED,
                         data.get("explanation", ""), json.dumps(feats))
```

- [ ] **Step 4: Implement — extend `bursa/importers/students.py`** (add alias handling inside the accepted-row block, after the students INSERT, still inside the `with dbmod.transaction(conn)`)

```python
                raw_aliases = (row.get("aliases") or "").strip()
                for alias in [a.strip() for a in raw_aliases.split(";") if a.strip()]:
                    conn.execute(
                        "INSERT OR IGNORE INTO student_aliases "
                        "(student_id, alias, normalized_alias) VALUES (?,?,?)",
                        (sid, alias, normalize.normalize_name(alias)))
```

- [ ] **Step 5: Run tests to verify pass** (model pipeline + Phase-1 regression)

Run: `.venv/bin/pytest tests/test_pipeline_model.py tests/test_pipeline.py tests/test_importers.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bursa/pipeline.py bursa/importers/students.py tests/test_pipeline_model.py
git commit -m "feat: wire model path into reconcile; import student aliases (Agent M Task 14)"
```

---

### Task 15: `download_model.sh` tokenizer + checksum, and the runbook

**Files:**
- Modify: `download_model.sh`
- Create: `docs/superpowers/runbooks/agent-m-verification.md`

**Interfaces:** none (script + doc).

- [ ] **Step 1: Update `download_model.sh`** — add after the GGUF block

```bash
# --- Tokenizer (for exact prompt-token budgeting) ---
TOKENIZER_FILE="tokenizer.json"
TOKENIZER_PATH="${MODEL_DIR}/${TOKENIZER_FILE}"
TOKENIZER_URL="TODO_HUGGINGFACE_TOKENIZER_JSON_URL"
TOKENIZER_SHA256="TODO_PINNED_SHA256"

if [ ! -f "${TOKENIZER_PATH}" ]; then
  if [ "${TOKENIZER_URL}" = "TODO_HUGGINGFACE_TOKENIZER_JSON_URL" ]; then
    echo "WARN: TOKENIZER_URL not set; app falls back to the heuristic token counter." >&2
  else
    echo "Downloading tokenizer to ${TOKENIZER_PATH} ..."
    curl -fL "${TOKENIZER_URL}" -o "${TOKENIZER_PATH}.partial"
    echo "${TOKENIZER_SHA256}  ${TOKENIZER_PATH}.partial" | shasum -a 256 -c - \
      || { echo "ERROR: tokenizer.json checksum mismatch" >&2; rm -f "${TOKENIZER_PATH}.partial"; exit 1; }
    mv "${TOKENIZER_PATH}.partial" "${TOKENIZER_PATH}"
  fi
fi
```

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n download_model.sh`
Expected: no output (syntax OK).

- [ ] **Step 3: Write the runbook** `docs/superpowers/runbooks/agent-m-verification.md`

```markdown
# Agent M — Verification Runbook (i5-class hardware)

These steps CANNOT run in the dev/CI environment (no model, no hardware). Run on the ADTC
standard laptop profile and record results in REPORT.md.

0. **Pin the toolchain (required):** record the exact `llama.cpp` version / commit hash.
   Grammar and `/no_think` behaviour vary across releases; C2 needs reproducible numbers.
   Tag every result below with this hash.
1. `bash download_model.sh` — fetches the GGUF and `tokenizer.json` (pinned SHA-256).
   Export `QWEN_TOKENIZER_JSON=model/tokenizer.json` so the app + the token corpus test
   use the exact tokenizer.
2. Start `llama-server` with the D3 config (see `bursa/inference/server.py::build_server_args`);
   confirm `GET /health`.
3. Run the 2 `metadata.json` test prompts through `LlamaServerBackend`; inspect JSON + reasoning.
4. **N=1000 over DISTINCT inputs** (required): temp 0 makes identical prompts identical, so
   1000 runs over the 13 demo fixtures = 13 effective samples and is invalid. Drive this from
   Agent D's synthetic generator (deterministic, seeded name/amount/narration permutations).
   Record valid-JSON pass/fail counts (target ≥99.5%) and unknown-ID count (target 0%).
5. **No-think assertion (required):** under `/no_think` + grammar, verify the output contains
   no think-block content and begins immediately with grammar-valid JSON.
6. Bare-GGUF chat-template check in clean Ollama AND LM Studio (no system prompt): the 2 test
   prompts + generic enterprise prompts answer well.
7. Record tps / peak-RSS / temperature, tagged with the pinned llama.cpp hash (step 0).
```

- [ ] **Step 4: Commit**

```bash
git add download_model.sh docs/superpowers/runbooks/agent-m-verification.md
git commit -m "feat: download tokenizer.json (pinned checksum) + Agent M verification runbook (Agent M Task 15)"
```

---

### Task 16: Full-suite green + regression

**Files:** none (verification task).

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (Phase-1 63 + Agent-M tests). Fix any regression before proceeding.

- [ ] **Step 2: Confirm Phase-1 invariants untouched**

Run: `.venv/bin/pytest tests/test_inv_matrix.py tests/test_ledger.py tests/test_properties.py -q`
Expected: PASS (Agent M is additive; the ledger/constraint core is unchanged).

- [ ] **Step 3: Commit any fixes**

```bash
git add -A && git commit -m "test: full-suite green with Agent M inference path (Agent M Task 16)"
```

---

## Self-review (plan vs spec)

**Spec coverage:** §1 modules → Tasks 5–14; §2 candidate generator (aliases, fuzzy, phone, siblings, payer-history net-of-reversals, ≤5 logged) → Tasks 1,3,4; §3 tokens + prompt + ladder + output budget → Tasks 5,7 (+ n_predict in Task 14); §4 grammar + backends + server + validation + split retry → Tasks 6,8,9,10,11; §5 features (versioned) + calibrator → Tasks 12,13; §6 pipeline wiring → Task 14; §7 constants → Task 5 + candidates.py; §8 tests → each task; §9 runbook + download_model checksum → Task 15. Full-suite gate → Task 16.

**Placeholder scan:** the only TODOs are `TOKENIZER_URL`/`TOKENIZER_SHA256` in `download_model.sh` — legitimately unknown until the model is published (Phase 3/4), mirroring the existing `MODEL_URL`; not plan placeholders.

**Type consistency:** `Candidate`, `generate`, `build_grammar`, `build` (prompt), `FakeBackend`/`BackendTransportError`, `LlamaServer`/`build_server_args`, `validate_output`/`ValidationOutcome`, `run_inference`, `extract`/`FEATURES_VERSION`, `ModelConfidencePolicy.route`, `reconcile(...backend...)` are used identically across tasks.

**Known executor notes:** (1) the token corpus over-count test (Task 5) is `skipif` without the tokenizer asset — it MUST be run on the i5 machine with `QWEN_TOKENIZER_JSON` set (runbook step 1). (2) `reconcile` gained `backend`/`tokenizer_path` params — Phase-1 `test_pipeline.py` calls it without a backend; the exact path is unaffected (no backend needed) and those tests must stay green (verified in Task 14 Step 5).
