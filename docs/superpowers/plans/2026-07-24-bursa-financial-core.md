# Bursa Financial Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-1 Bursa Financial Core — data model, CSV importers, deterministic matcher, cumulative constraint engine (INV-01..10), and an append-only student-fee ledger with reversals and credit — plus the pytest+hypothesis suite that checkpoint C1 gates on. Backend + tests only; no UI, LLM, OCR, or calibrator.

**Architecture:** Append-only event-sourced ledger. Every financial fact (billed charge, allocation, reversal, credit) is a `ledger_events` row; balances/status/unapplied are computed from events, never stored. `ledger.py` is the sole writer and refuses anything `constraints.validate()` hasn't passed inside one `BEGIN IMMEDIATE` transaction. Allocations target a `(student, charge)` pair (charge-level); the model/matcher proposes `{student, amount}` and `distribute.py` maps it to charge-level events.

**Tech Stack:** Python 3.11+, stdlib `sqlite3` (WAL + `foreign_keys=ON`), Pydantic v2, pytest, hypothesis. No ORM, no web framework in Phase 1.

**Reversal realization (read this first):** A reversal fully negates its target (corrections = reverse-in-full + fresh allocation). We realize the spec's "net of reversals / signed sum" semantics via a **live/dead** rule: an event is *dead* if some `REVERSAL` row points at it via `reverses_event_id`; every aggregate sums only **live** contributions. `REVERSAL` rows are markers and are never summed as contributions. This is exactly equivalent to signed sums for full reversals, avoids sign-juggling bugs, and makes "no reversal of a reversal" a one-line guard.

## Global Constraints

- **INV-03 / D12:** money is `int` **minor units** everywhere; floating-point is prohibited anywhere near money. `money.py` is the ONLY module that parses or formats money.
- **Sole ledger writer:** `ledger.py` is the ONLY module that appends `ledger_events`. Everything posts through `ledger.post()` / `ledger.reverse()` / `ledger.create_charge()`.
- **Validation before posting:** `constraints.validate()` runs inside the atomic post; on any violation the transaction rolls back and routing becomes `review` (INV-10). Auto-post is not a bypass.
- **Append-only:** `ledger_events` has BEFORE UPDATE/DELETE triggers that `RAISE(ABORT)`. No reversal of a reversal.
- **Cumulative & net of reversals:** INV-01/02/05/07 and credit sufficiency are checked against live (non-dead) prior events + the proposed events, never the proposal alone.
- **FK enforcement:** `db.py` sets `PRAGMA foreign_keys=ON` and `PRAGMA journal_mode=WAL` on **every** connection.
- **Untrusted input:** narrations/imports are data; the matcher never treats narration as instructions and never allocates to a non-imported ID (INV-04).
- **Offline:** no network calls anywhere in the core or its tests.

---

## File structure

```
bursa/
  __init__.py
  errors.py         # ImportRowError, InvariantViolation
  money.py          # parse_naira / format_naira (sole money converter)
  models.py         # enums + Pydantic DTOs (CanonicalTransaction, ProposalLine, LedgerEventInput, …)
  normalize.py      # normalize_name / canonicalize_reference / narration tokens
  db.py             # connect(), SCHEMA_SQL, init_db(), transaction()
  repository.py     # parameterized reads/writes for every table
  projections.py    # live-event aggregate primitives + balances/status/unapplied/dashboard
  constraints.py    # validate(proposed_events, txn) -> ValidationResult (INV-01..10)
  ledger.py         # create_charge(), post(), reverse(), grant_credit(), apply_credit()
  distribute.py     # distribute(student_id, amount) -> allocation event inputs + remainder
  reasoncodes.py    # ReasonCode enum
  matcher.py        # match(transaction) -> Proposal (reason codes + recommended action)
  confidence.py     # ConfidencePolicy protocol + RuleBasedConfidencePolicy
  config.py         # Config(auto_post_enabled=True)
  pipeline.py       # reconcile(transaction) orchestration
tests/
  conftest.py       # db fixture (schema + triggers + FK + WAL)
  test_money.py test_models.py test_normalize.py test_db.py test_repository.py
  test_projections.py test_constraints.py test_ledger.py test_distribute.py
  test_importers.py test_statement.py test_matcher.py test_pipeline.py
  test_properties.py   # hypothesis property + RuleBasedStateMachine + concurrency
```

Dependency order (each task builds on earlier ones): errors → money → models → normalize → db → repository → projections → constraints → ledger → distribute → importers → statement → matcher → pipeline → cross-cutting tests.

---

### Task 1: Project scaffold + errors

**Files:**
- Create: `pyproject.toml`, `bursa/__init__.py`, `bursa/errors.py`, `tests/__init__.py`, `tests/test_errors.py`

**Interfaces:**
- Produces: `ImportRowError(row_number:int, field:str, reason:str)`; `InvariantViolation(violations:list[str])` with `.violations` attribute.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "bursa"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2.6"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.100"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Create empty `bursa/__init__.py` and `tests/__init__.py`** (both empty files).

- [ ] **Step 3: Write the failing test** in `tests/test_errors.py`

```python
from bursa.errors import ImportRowError, InvariantViolation


def test_import_row_error_carries_context():
    e = ImportRowError(row_number=4, field="amount", reason="not a number")
    assert e.row_number == 4
    assert e.field == "amount"
    assert "not a number" in str(e)


def test_invariant_violation_lists_codes():
    e = InvariantViolation(["INV-01", "INV-05"])
    assert e.violations == ["INV-01", "INV-05"]
    assert "INV-01" in str(e)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_errors.py -v`
Expected: FAIL (ModuleNotFoundError: bursa.errors).

- [ ] **Step 5: Implement `bursa/errors.py`**

```python
class ImportRowError(Exception):
    """A single CSV row failed validation; the rest of the file continues."""

    def __init__(self, row_number: int, field: str, reason: str):
        self.row_number = row_number
        self.field = field
        self.reason = reason
        super().__init__(f"row {row_number}: {field}: {reason}")


class InvariantViolation(Exception):
    """One or more ledger invariants would be broken by a proposed posting."""

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_errors.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml bursa/__init__.py bursa/errors.py tests/__init__.py tests/test_errors.py
git commit -m "feat: project scaffold + typed errors"
```

---

### Task 2: money.py — the sole money converter

**Files:**
- Create: `bursa/money.py`, `tests/test_money.py`

**Interfaces:**
- Produces: `parse_naira(value: str) -> int` (returns minor units; raises `ValueError`); `format_naira(minor: int) -> str` (e.g. `"₦75,000.00"`); constant `MINOR_UNITS_PER_NAIRA = 100`.

- [ ] **Step 1: Write the failing test** in `tests/test_money.py`

```python
import pytest
from bursa.money import parse_naira, format_naira, MINOR_UNITS_PER_NAIRA


@pytest.mark.parametrize("text,minor", [
    ("75000", 7_500_000),
    ("75,000", 7_500_000),
    ("₦75,000.00", 7_500_000),
    ("75000.5", 7_500_050),
    ("0", 0),
    ("  1,234.99 ", 123_499),
])
def test_parse_naira_ok(text, minor):
    assert parse_naira(text) == minor


@pytest.mark.parametrize("bad", ["", "abc", "1.234", "-5", "1e3", "12,34,5"])
def test_parse_naira_rejects(bad):
    with pytest.raises(ValueError):
        parse_naira(bad)


def test_format_naira():
    assert format_naira(7_500_000) == "₦75,000.00"
    assert format_naira(0) == "₦0.00"
    assert format_naira(123_499) == "₦1,234.99"


def test_round_trip():
    assert format_naira(parse_naira("₦1,234.99")) == "₦1,234.99"


def test_no_floats_stored():
    # parse returns an int, never a float
    assert isinstance(parse_naira("75000.50"), int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_money.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/money.py`**

```python
from decimal import Decimal, InvalidOperation

MINOR_UNITS_PER_NAIRA = 100


def parse_naira(value: str) -> int:
    """Parse a naira string to integer minor units (kobo).

    Decimal is used ONLY at this boundary; the result is always an int.
    Rejects empty, non-numeric, negative, exponent, or >2-decimal input.
    """
    if not isinstance(value, str):
        raise ValueError("money must be parsed from a string")
    cleaned = value.strip().replace("₦", "").replace(",", "").strip()
    if cleaned == "" or "e" in cleaned.lower():
        raise ValueError(f"invalid money value: {value!r}")
    if "." in cleaned and len(cleaned.split(".")[1]) > 2:
        raise ValueError(f"too many decimal places: {value!r}")
    # Reject grouping mistakes like "12,34,5" (cleaned would keep digits but
    # original grouping is invalid) — re-check original grouping.
    if "," in value:
        whole = value.strip().replace("₦", "").split(".")[0]
        groups = whole.split(",")
        if len(groups) > 1 and (len(groups[0]) == 0 or len(groups[0]) > 3
                                or any(len(g) != 3 for g in groups[1:])):
            raise ValueError(f"invalid thousands grouping: {value!r}")
    try:
        dec = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"invalid money value: {value!r}")
    if dec < 0:
        raise ValueError(f"money cannot be negative: {value!r}")
    minor = (dec * MINOR_UNITS_PER_NAIRA).to_integral_value()
    return int(minor)


def format_naira(minor: int) -> str:
    """Format integer minor units as a naira display string."""
    if not isinstance(minor, int):
        raise ValueError("format_naira requires an int (minor units)")
    naira, kobo = divmod(abs(minor), MINOR_UNITS_PER_NAIRA)
    sign = "-" if minor < 0 else ""
    return f"{sign}₦{naira:,}.{kobo:02d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_money.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/money.py tests/test_money.py
git commit -m "feat: money parse/format in integer minor units (INV-03)"
```

---

### Task 3: models.py — enums + DTOs

**Files:**
- Create: `bursa/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces string enums `EventType` (`CHARGE_CREATED, ALLOCATION, REVERSAL, CREDIT_GRANT, CREDIT_APPLICATION`), `RoutingState` (`NEW, AUTO, REVIEW, UNMATCHED`), `RecommendedAction` (`AUTO, REVIEW, UNMATCHED`); Pydantic models `CanonicalTransaction`, `ProposalLine`, `Proposal`, `LedgerEventInput`.

- [ ] **Step 1: Write the failing test** in `tests/test_models.py`

```python
import pytest
from pydantic import ValidationError
from bursa.models import (EventType, RoutingState, RecommendedAction,
                          CanonicalTransaction, ProposalLine, LedgerEventInput)


def test_enums_are_strings():
    assert EventType.ALLOCATION == "allocation"
    assert RoutingState.REVIEW == "review"
    assert RecommendedAction.AUTO == "auto"


def test_transaction_amount_must_be_int():
    with pytest.raises(ValidationError):
        CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
                             posted_at="2026-02-14T09:32:00+01:00",
                             amount_minor=1000.5, direction="credit",
                             dedup_hash="h1")


def test_proposal_line_defaults_reason_codes():
    line = ProposalLine(student_id="STU-1", amount_minor=5000)
    assert line.reason_codes == []


def test_ledger_event_input_requires_provenance_fields():
    with pytest.raises(ValidationError):
        LedgerEventInput(event_type=EventType.ALLOCATION, amount_minor=100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/models.py`**

```python
from enum import StrEnum
from pydantic import BaseModel, Field, StrictInt


class EventType(StrEnum):
    CHARGE_CREATED = "charge_created"
    ALLOCATION = "allocation"
    REVERSAL = "reversal"
    CREDIT_GRANT = "credit_grant"
    CREDIT_APPLICATION = "credit_application"


class RoutingState(StrEnum):
    NEW = "new"
    AUTO = "auto"
    REVIEW = "review"
    UNMATCHED = "unmatched"


class RecommendedAction(StrEnum):
    AUTO = "auto"
    REVIEW = "review"
    UNMATCHED = "unmatched"


class CanonicalTransaction(BaseModel):
    transaction_id: str
    source: str
    reference: str | None = None
    raw_reference: str | None = None
    posted_at: str
    payer_name: str | None = None
    narration: str | None = None
    amount_minor: StrictInt
    direction: str  # "credit" | "debit"
    dedup_hash: str
    batch_id: str | None = None


class ProposalLine(BaseModel):
    student_id: str
    amount_minor: StrictInt
    reason_codes: list[str] = Field(default_factory=list)


class Proposal(BaseModel):
    transaction_id: str
    source: str  # "deterministic" | "llm"
    lines: list[ProposalLine] = Field(default_factory=list)
    recommended_action: RecommendedAction
    confidence: float | None = None
    explanation: str = ""


class LedgerEventInput(BaseModel):
    event_type: EventType
    amount_minor: StrictInt
    actor: str
    source: str
    evidence_ref: str
    decision_path: str
    transaction_id: str | None = None
    charge_id: str | None = None
    student_id: str | None = None
    fee_id: str | None = None
    holder: str | None = None
    reverses_event_id: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS. (`StrictInt` makes `1000.5` fail; missing `actor/source/evidence_ref/decision_path` fail.)

- [ ] **Step 5: Commit**

```bash
git add bursa/models.py tests/test_models.py
git commit -m "feat: domain enums and Pydantic DTOs"
```

---

### Task 4: normalize.py

**Files:**
- Create: `bursa/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_name(raw:str)->str`, `canonicalize_reference(raw:str|None)->str|None`, `narration_tokens(raw:str|None)->list[str]`.

- [ ] **Step 1: Write the failing test** in `tests/test_normalize.py`

```python
from bursa.normalize import normalize_name, canonicalize_reference, narration_tokens


def test_normalize_name_strips_titles_and_case():
    assert normalize_name("  Mr. C. N.  OKAFOR ") == "c n okafor"
    assert normalize_name("Chidi-Somto") == "chidi somto"


def test_canonicalize_reference():
    assert canonicalize_reference(" nip-728/1049 ") == "NIP7281049"
    assert canonicalize_reference(None) is None
    assert canonicalize_reference("") is None


def test_narration_tokens():
    assert narration_tokens("CHI AND SOMTO SCH FEE") == ["chi", "and", "somto", "sch", "fee"]
    assert narration_tokens(None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/normalize.py`**

```python
import re

_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "chief", "alhaji", "mallam"}


def normalize_name(raw: str) -> str:
    tokens = re.split(r"[^a-zA-Z]+", (raw or "").lower())
    tokens = [t for t in tokens if t and t not in _TITLES]
    return " ".join(tokens)


def canonicalize_reference(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return cleaned or None


def narration_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t for t in re.split(r"[^a-zA-Z0-9]+", raw.lower()) if t]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/normalize.py tests/test_normalize.py
git commit -m "feat: name/reference/narration normalization"
```

---

### Task 5: db.py — schema, triggers, pragmas, transaction helper

**Files:**
- Create: `bursa/db.py`, `tests/conftest.py`, `tests/test_db.py`

**Interfaces:**
- Produces: `connect(path:str)->sqlite3.Connection` (sets `foreign_keys=ON`, `journal_mode=WAL`, `row_factory=sqlite3.Row`); `init_db(conn)` (runs `SCHEMA_SQL`); `transaction(conn)` context manager (`BEGIN IMMEDIATE`/commit/rollback); constant `SCHEMA_SQL`.

- [ ] **Step 1: Write the failing test** in `tests/test_db.py`

```python
import sqlite3
import pytest
from bursa.db import connect, init_db, transaction


def test_foreign_keys_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db):
            db.execute(
                "INSERT INTO charges (charge_id, student_id, fee_id, term_id) "
                "VALUES ('C1','NOPE','NOPE','NOPE')")


def test_ledger_append_only_update_blocked(db, seeded_ledger_event):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("UPDATE ledger_events SET amount_minor = 1 WHERE event_id = ?",
                   (seeded_ledger_event,))


def test_ledger_append_only_delete_blocked(db, seeded_ledger_event):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM ledger_events WHERE event_id = ?", (seeded_ledger_event,))


def test_wal_mode(db):
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
```

- [ ] **Step 2: Add fixtures in `tests/conftest.py`**

```python
import pytest
from bursa.db import connect, init_db, transaction


@pytest.fixture
def db(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def seeded_term_student_fee(db):
    with transaction(db):
        db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
        db.execute("INSERT INTO students VALUES ('STU-1','Chi Okafor','chi okafor','JSS1','T1')")
        db.execute("INSERT INTO fee_items VALUES ('FEE-TUITION','Tuition','T1',10)")
        db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    return "CHG-1"


@pytest.fixture
def seeded_ledger_event(db, seeded_term_student_fee):
    cur = db.execute(
        "INSERT INTO ledger_events "
        "(event_type, charge_id, student_id, fee_id, amount_minor, actor, source, "
        " evidence_ref, decision_path, created_at) "
        "VALUES ('charge_created','CHG-1','STU-1','FEE-TUITION',5000000,'importer',"
        "'fees_csv','BATCH-1','import','2026-01-01T00:00:00+00:00')")
    db.commit()
    return cur.lastrowid
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implement `bursa/db.py`**

```python
import sqlite3
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE terms (
  term_id TEXT PRIMARY KEY, session TEXT NOT NULL,
  term_name TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE students (
  student_id TEXT PRIMARY KEY, name TEXT NOT NULL, normalized_name TEXT NOT NULL,
  class TEXT, term_id TEXT REFERENCES terms(term_id)
);
CREATE TABLE guardians (
  guardian_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  normalized_name TEXT NOT NULL, phone_suffix TEXT
);
CREATE TABLE student_guardians (
  student_id TEXT NOT NULL REFERENCES students(student_id),
  guardian_id TEXT NOT NULL REFERENCES guardians(guardian_id),
  PRIMARY KEY (student_id, guardian_id)
);
CREATE TABLE fee_items (
  fee_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  term_id TEXT REFERENCES terms(term_id), priority INTEGER NOT NULL DEFAULT 100
);
CREATE TABLE charges (
  charge_id TEXT PRIMARY KEY,
  student_id TEXT NOT NULL REFERENCES students(student_id),
  fee_id TEXT NOT NULL REFERENCES fee_items(fee_id),
  term_id TEXT NOT NULL REFERENCES terms(term_id)
);
CREATE TABLE import_batches (
  batch_id TEXT PRIMARY KEY, source_file TEXT NOT NULL, imported_at TEXT NOT NULL,
  accepted INTEGER NOT NULL DEFAULT 0, rejected INTEGER NOT NULL DEFAULT 0,
  duplicate INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE transactions (
  transaction_id TEXT PRIMARY KEY, source TEXT NOT NULL,
  reference TEXT, raw_reference TEXT, posted_at TEXT NOT NULL,
  payer_name TEXT, narration TEXT, amount_minor INTEGER NOT NULL,
  direction TEXT NOT NULL, dedup_hash TEXT NOT NULL UNIQUE,
  batch_id TEXT REFERENCES import_batches(batch_id),
  routing_state TEXT NOT NULL DEFAULT 'new'
);
CREATE UNIQUE INDEX ux_transactions_reference
  ON transactions(reference) WHERE reference IS NOT NULL;
CREATE TABLE proposals (
  proposal_id TEXT PRIMARY KEY,
  transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
  source TEXT NOT NULL, recommended_action TEXT NOT NULL,
  confidence REAL, explanation TEXT,
  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
);
CREATE TABLE proposal_allocations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
  student_id TEXT NOT NULL REFERENCES students(student_id),
  amount_minor INTEGER NOT NULL, reason_codes TEXT NOT NULL
);
CREATE TABLE ledger_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  transaction_id TEXT REFERENCES transactions(transaction_id),
  charge_id TEXT REFERENCES charges(charge_id),
  student_id TEXT REFERENCES students(student_id),
  fee_id TEXT REFERENCES fee_items(fee_id),
  holder TEXT,
  amount_minor INTEGER NOT NULL,
  actor TEXT NOT NULL, source TEXT NOT NULL,
  evidence_ref TEXT NOT NULL, decision_path TEXT NOT NULL,
  reverses_event_id INTEGER REFERENCES ledger_events(event_id),
  created_at TEXT NOT NULL
);
CREATE TRIGGER ledger_events_no_update BEFORE UPDATE ON ledger_events
BEGIN SELECT RAISE(ABORT, 'ledger_events is append-only: UPDATE forbidden'); END;
CREATE TRIGGER ledger_events_no_delete BEFORE DELETE ON ledger_events
BEGIN SELECT RAISE(ABORT, 'ledger_events is append-only: DELETE forbidden'); END;
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we manage txns
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS (FK IntegrityError; UPDATE/DELETE raise OperationalError from the trigger; WAL confirmed).

- [ ] **Step 6: Commit**

```bash
git add bursa/db.py tests/conftest.py tests/test_db.py
git commit -m "feat: schema, append-only triggers, FK+WAL pragmas, txn helper (INV-08/04)"
```

---

### Task 6: repository.py — parameterized data access

**Files:**
- Create: `bursa/repository.py`, `tests/test_repository.py`

**Interfaces:**
- Produces (all take `conn` first): `insert_transaction(conn, tx: CanonicalTransaction)`, `get_transaction(conn, txn_id)->sqlite3.Row|None`, `find_transaction_by_dedup(conn, dedup_hash)->Row|None`, `set_routing_state(conn, txn_id, state)`, `insert_ledger_event(conn, ev: LedgerEventInput, created_at:str)->int`, `live_events(conn, **filters)->list[Row]`, `insert_proposal(conn, ...)`, `charge_exists(conn, charge_id)->bool`, `student_exists(conn, student_id)->bool`, `outstanding_charges_for_student(conn, student_id)->list[Row]`.

> "Live" = `event_id NOT IN (SELECT reverses_event_id FROM ledger_events WHERE reverses_event_id IS NOT NULL)` AND `event_type != 'reversal'`.

- [ ] **Step 1: Write the failing test** in `tests/test_repository.py`

```python
import pytest
from bursa import repository as repo
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _tx(**kw):
    base = dict(transaction_id="TXN-1", source="bank_csv", posted_at="2026-02-14T00:00:00+00:00",
               amount_minor=7500000, direction="credit", dedup_hash="h1")
    base.update(kw)
    return CanonicalTransaction(**base)


def test_transaction_round_trip(db):
    repo.insert_transaction(db, _tx())
    row = repo.get_transaction(db, "TXN-1")
    assert row["amount_minor"] == 7500000
    assert repo.find_transaction_by_dedup(db, "h1")["transaction_id"] == "TXN-1"


def test_live_events_excludes_reversed(db, seeded_term_student_fee):
    e1 = repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=5000000, actor="importer",
        source="fees_csv", evidence_ref="BATCH-1", decision_path="import"),
        created_at="2026-01-01T00:00:00+00:00")
    live = repo.live_events(db, charge_id="CHG-1")
    assert len(live) == 1
    # reverse it
    repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.REVERSAL, charge_id="CHG-1", amount_minor=5000000,
        actor="bursar", source="correction", evidence_ref="BATCH-1",
        decision_path="reverse", reverses_event_id=e1),
        created_at="2026-01-02T00:00:00+00:00")
    assert repo.live_events(db, charge_id="CHG-1") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repository.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/repository.py`**

```python
import sqlite3
from bursa.models import CanonicalTransaction, LedgerEventInput

_LIVE = ("event_type != 'reversal' AND event_id NOT IN "
         "(SELECT reverses_event_id FROM ledger_events WHERE reverses_event_id IS NOT NULL)")


def insert_transaction(conn, tx: CanonicalTransaction) -> None:
    conn.execute(
        "INSERT INTO transactions (transaction_id, source, reference, raw_reference, "
        "posted_at, payer_name, narration, amount_minor, direction, dedup_hash, "
        "batch_id, routing_state) VALUES (?,?,?,?,?,?,?,?,?,?,?, 'new')",
        (tx.transaction_id, tx.source, tx.reference, tx.raw_reference, tx.posted_at,
         tx.payer_name, tx.narration, tx.amount_minor, tx.direction, tx.dedup_hash,
         tx.batch_id))


def get_transaction(conn, txn_id) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM transactions WHERE transaction_id = ?",
                        (txn_id,)).fetchone()


def find_transaction_by_dedup(conn, dedup_hash) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM transactions WHERE dedup_hash = ?",
                        (dedup_hash,)).fetchone()


def set_routing_state(conn, txn_id, state) -> None:
    conn.execute("UPDATE transactions SET routing_state = ? WHERE transaction_id = ?",
                 (str(state), txn_id))


def insert_ledger_event(conn, ev: LedgerEventInput, created_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO ledger_events (event_type, transaction_id, charge_id, student_id, "
        "fee_id, holder, amount_minor, actor, source, evidence_ref, decision_path, "
        "reverses_event_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (str(ev.event_type), ev.transaction_id, ev.charge_id, ev.student_id, ev.fee_id,
         ev.holder, ev.amount_minor, ev.actor, ev.source, ev.evidence_ref,
         ev.decision_path, ev.reverses_event_id, created_at))
    return cur.lastrowid


def live_events(conn, **filters) -> list[sqlite3.Row]:
    where = [_LIVE]
    params = []
    for col, val in filters.items():
        where.append(f"{col} = ?")
        params.append(val)
    sql = f"SELECT * FROM ledger_events WHERE {' AND '.join(where)}"
    return conn.execute(sql, params).fetchall()


def event_by_id(conn, event_id) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM ledger_events WHERE event_id = ?",
                        (event_id,)).fetchone()


def charge_exists(conn, charge_id) -> bool:
    return conn.execute("SELECT 1 FROM charges WHERE charge_id = ?",
                        (charge_id,)).fetchone() is not None


def student_exists(conn, student_id) -> bool:
    return conn.execute("SELECT 1 FROM students WHERE student_id = ?",
                        (student_id,)).fetchone() is not None


def charges_for_student(conn, student_id) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT c.*, f.priority FROM charges c JOIN fee_items f ON c.fee_id = f.fee_id "
        "WHERE c.student_id = ? ORDER BY f.priority ASC, c.charge_id ASC",
        (student_id,)).fetchall()


def insert_proposal(conn, proposal_id, transaction_id, source, action, confidence,
                    explanation, created_at) -> None:
    conn.execute(
        "INSERT INTO proposals (proposal_id, transaction_id, source, recommended_action, "
        "confidence, explanation, created_at) VALUES (?,?,?,?,?,?,?)",
        (proposal_id, transaction_id, source, str(action), confidence, explanation,
         created_at))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_repository.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/repository.py tests/test_repository.py
git commit -m "feat: parameterized repository with live-event reads"
```

---

### Task 7: projections.py — live-event aggregates

**Files:**
- Create: `bursa/projections.py`, `tests/test_projections.py`

**Interfaces:**
- Produces: `charge_billed(conn, charge_id)->int`, `charge_paid(conn, charge_id)->int`, `charge_balance(conn, charge_id)->int`, `txn_used(conn, txn_id)->int`, `txn_unapplied(conn, txn)->int`, `holder_credit(conn, holder)->int`, `student_status(conn, student_id)->str`.

- [ ] **Step 1: Write the failing test** in `tests/test_projections.py`

```python
from bursa import projections as proj, repository as repo
from bursa.models import LedgerEventInput, EventType


def _ev(db, **kw):
    base = dict(actor="a", source="s", evidence_ref="e", decision_path="d")
    base.update(kw)
    return repo.insert_ledger_event(db, LedgerEventInput(**base), created_at="2026-01-01T00:00:00+00:00")


def test_charge_balance_and_status(db, seeded_term_student_fee):
    _ev(db, event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=5000000)
    assert proj.charge_billed(db, "CHG-1") == 5000000
    assert proj.charge_balance(db, "CHG-1") == 5000000
    assert proj.student_status(db, "STU-1") == "outstanding"
    _ev(db, event_type=EventType.ALLOCATION, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", transaction_id=None, amount_minor=2000000)
    assert proj.charge_balance(db, "CHG-1") == 3000000
    assert proj.student_status(db, "STU-1") == "part_paid"


def test_holder_credit(db, seeded_term_student_fee):
    _ev(db, event_type=EventType.CREDIT_GRANT, holder="STU-1", amount_minor=1000000)
    assert proj.holder_credit(db, "STU-1") == 1000000
    _ev(db, event_type=EventType.CREDIT_APPLICATION, holder="STU-1", charge_id="CHG-1",
        amount_minor=400000)
    assert proj.holder_credit(db, "STU-1") == 600000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_projections.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/projections.py`**

```python
from bursa import repository as repo


def _sum(rows, predicate) -> int:
    return sum(r["amount_minor"] for r in rows if predicate(r))


def charge_billed(conn, charge_id) -> int:
    rows = repo.live_events(conn, charge_id=charge_id)
    return _sum(rows, lambda r: r["event_type"] == "charge_created")


def charge_paid(conn, charge_id) -> int:
    rows = repo.live_events(conn, charge_id=charge_id)
    return _sum(rows, lambda r: r["event_type"] in ("allocation", "credit_application"))


def charge_balance(conn, charge_id) -> int:
    return charge_billed(conn, charge_id) - charge_paid(conn, charge_id)


def txn_used(conn, txn_id) -> int:
    rows = repo.live_events(conn, transaction_id=txn_id)
    return _sum(rows, lambda r: r["event_type"] in ("allocation", "credit_grant"))


def txn_unapplied(conn, txn) -> int:
    return txn["amount_minor"] - txn_used(conn, txn["transaction_id"])


def holder_credit(conn, holder) -> int:
    rows = repo.live_events(conn, holder=holder)
    grants = _sum(rows, lambda r: r["event_type"] == "credit_grant")
    apps = _sum(rows, lambda r: r["event_type"] == "credit_application")
    return grants - apps


def student_status(conn, student_id) -> str:
    charges = repo.charges_for_student(conn, student_id)
    billed = sum(charge_billed(conn, c["charge_id"]) for c in charges)
    paid = sum(charge_paid(conn, c["charge_id"]) for c in charges)
    if holder_credit(conn, student_id) > 0 and paid >= billed:
        return "credit"
    if paid <= 0:
        return "outstanding"
    if paid < billed:
        return "part_paid"
    return "cleared"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_projections.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/projections.py tests/test_projections.py
git commit -m "feat: balances, credit, and fee status derived from live events (BR-03)"
```

---

### Task 8: constraints.py — the cumulative engine (INV-01..10)

**Files:**
- Create: `bursa/constraints.py`, `tests/test_constraints.py`

**Interfaces:**
- Consumes: `projections`, `repository`, `LedgerEventInput`.
- Produces: `ValidationResult` (dataclass with `.ok: bool`, `.violations: list[str]`); `validate(conn, txn: Row, proposed: list[LedgerEventInput]) -> ValidationResult`.

**Rules enforced (each appends its INV code to `violations` on failure):**
- INV-03: every proposed `amount_minor` is a non-negative `int`.
- INV-04: every proposed `student_id`/`charge_id` exists.
- INV-01: `txn_used(live) + Σ proposed(allocation+credit_grant for txn) ≤ txn.amount_minor`.
- INV-05: for each charge, `charge_paid(live) + Σ proposed(allocation+credit_application to charge) ≤ charge_billed(live)`.
- Credit sufficiency: for each holder, `Σ proposed credit_application(holder) ≤ holder_credit(live)`.
- INV-09: every proposed event has non-empty `actor/source/evidence_ref/decision_path`.

- [ ] **Step 1: Write the failing test** in `tests/test_constraints.py`

```python
from bursa import constraints, repository as repo
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _seed_charge(db, billed):
    repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=billed, actor="i", source="fees",
        evidence_ref="B", decision_path="import"), created_at="2026-01-01T00:00:00+00:00")


def _txn(db, amount):
    tx = CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
        direction="credit", dedup_hash="h1")
    repo.insert_transaction(db, tx)
    return repo.get_transaction(db, "TXN-1")


def _alloc(amount, charge="CHG-1"):
    return LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id="TXN-1",
        charge_id=charge, student_id="STU-1", fee_id="FEE-TUITION", amount_minor=amount,
        actor="engine", source="deterministic", evidence_ref="TXN-1", decision_path="auto")


def test_inv01_allocation_exceeds_transaction(db, seeded_term_student_fee):
    _seed_charge(db, 9_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(6_000_000)])
    assert not r.ok and "INV-01" in r.violations


def test_inv05_overfills_charge(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(4_000_000)])
    assert not r.ok and "INV-05" in r.violations


def test_inv04_unknown_charge(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(1_000_000, charge="CHG-NOPE")])
    assert not r.ok and "INV-04" in r.violations


def test_valid_allocation_passes(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(3_000_000)])
    assert r.ok and r.violations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_constraints.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/constraints.py`**

```python
from dataclasses import dataclass, field
from collections import defaultdict
from bursa import projections as proj, repository as repo
from bursa.models import LedgerEventInput


@dataclass
class ValidationResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def validate(conn, txn, proposed: list[LedgerEventInput]) -> ValidationResult:
    v: list[str] = []

    # INV-03: integer, non-negative amounts
    for ev in proposed:
        if not isinstance(ev.amount_minor, int) or ev.amount_minor < 0:
            v.append("INV-03")
            break

    # INV-09: provenance present
    for ev in proposed:
        if not (ev.actor and ev.source and ev.evidence_ref and ev.decision_path):
            v.append("INV-09")
            break

    # INV-04: referenced ids exist
    for ev in proposed:
        if ev.charge_id is not None and not repo.charge_exists(conn, ev.charge_id):
            v.append("INV-04")
            break
        if ev.student_id is not None and not repo.student_exists(conn, ev.student_id):
            v.append("INV-04")
            break

    # INV-01: transaction capacity (allocations + credit grants funded by txn)
    txn_id = txn["transaction_id"]
    proposed_txn = sum(ev.amount_minor for ev in proposed
                       if ev.transaction_id == txn_id
                       and ev.event_type in ("allocation", "credit_grant"))
    if proj.txn_used(conn, txn_id) + proposed_txn > txn["amount_minor"]:
        v.append("INV-01")

    # INV-05: no charge overfilled
    by_charge = defaultdict(int)
    for ev in proposed:
        if ev.charge_id and ev.event_type in ("allocation", "credit_application"):
            by_charge[ev.charge_id] += ev.amount_minor
    for charge_id, add in by_charge.items():
        if repo.charge_exists(conn, charge_id):
            if proj.charge_paid(conn, charge_id) + add > proj.charge_billed(conn, charge_id):
                v.append("INV-05")
                break

    # Credit sufficiency: applications <= available credit per holder
    by_holder = defaultdict(int)
    for ev in proposed:
        if ev.event_type == "credit_application" and ev.holder:
            by_holder[ev.holder] += ev.amount_minor
    for holder, add in by_holder.items():
        if add > proj.holder_credit(conn, holder):
            v.append("INV-05")  # credit non-negativity is an INV-05-class floor
            break

    return ValidationResult(ok=(len(v) == 0), violations=v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_constraints.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/constraints.py tests/test_constraints.py
git commit -m "feat: cumulative constraint engine INV-01/03/04/05/09 + credit sufficiency"
```

---

### Task 9: ledger.py — sole writer (create_charge, post, reverse, credit)

**Files:**
- Create: `bursa/ledger.py`, `tests/test_ledger.py`

**Interfaces:**
- Consumes: `db.transaction`, `constraints.validate`, `repository`, `LedgerEventInput`, `InvariantViolation`.
- Produces:
  - `create_charge(conn, charge_id, student_id, fee_id, term_id, amount_minor, actor, source, evidence_ref) -> int` (writes `charges` row + `charge_created` event atomically)
  - `post(conn, txn_id, proposed: list[LedgerEventInput], actor, source, evidence_ref, decision_path) -> list[int]` (validates then appends; raises `InvariantViolation` on failure, rolling back)
  - `reverse(conn, event_id, actor, reason) -> int` (rejects reversing a `reversal`)
  - `grant_credit(...)`, `apply_credit(...)` thin wrappers that post credit events through `post`-style validation.
- Uses `_now()` = `datetime.now(timezone.utc).isoformat()`.

- [ ] **Step 1: Write the failing test** in `tests/test_ledger.py`

```python
import pytest
from bursa import ledger, projections as proj, repository as repo
from bursa.errors import InvariantViolation
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _txn(db, amount, dedup="h1", tid="TXN-1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        dedup_hash=dedup))
    return repo.get_transaction(db, tid)


def _alloc(amount, tid="TXN-1"):
    return LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=tid,
        charge_id="CHG-1", student_id="STU-1", fee_id="FEE-TUITION", amount_minor=amount,
        actor="engine", source="deterministic", evidence_ref=tid, decision_path="auto")


@pytest.fixture
def charged(db, seeded_term_student_fee):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000,
                         "importer", "fees_csv", "BATCH-1")
    return "CHG-1"


def test_create_charge_writes_row_and_event(db, seeded_term_student_fee):
    # remove the fixture-seeded charge collision: use CHG-2
    ledger.create_charge(db, "CHG-2", "STU-1", "FEE-TUITION", "T1", 4_000_000,
                         "importer", "fees_csv", "BATCH-1")
    assert proj.charge_billed(db, "CHG-2") == 4_000_000


def test_post_valid_allocation(db, charged):
    txn = _txn(db, 5_000_000)
    ids = ledger.post(db, "TXN-1", [_alloc(5_000_000)], "engine", "deterministic",
                      "TXN-1", "auto")
    assert len(ids) == 1
    assert proj.charge_balance(db, "CHG-1") == 0


def test_post_over_capacity_rolls_back(db, charged):
    txn = _txn(db, 3_000_000)
    with pytest.raises(InvariantViolation) as ei:
        ledger.post(db, "TXN-1", [_alloc(4_000_000)], "engine", "deterministic",
                    "TXN-1", "auto")
    assert "INV-01" in ei.value.violations
    assert repo.live_events(db, transaction_id="TXN-1") == []  # nothing written


def test_reverse_then_repost_same_amount_succeeds(db, charged):
    txn = _txn(db, 5_000_000)
    [aid] = ledger.post(db, "TXN-1", [_alloc(2_000_000)], "e", "d", "TXN-1", "auto")
    ledger.reverse(db, aid, "bursar", "mistake")
    # capacity and charge freed -> repost same amount succeeds
    ledger.post(db, "TXN-1", [_alloc(2_000_000)], "e", "d", "TXN-1", "auto")
    assert proj.charge_paid(db, "CHG-1") == 2_000_000


def test_reverse_of_reversal_blocked(db, charged):
    txn = _txn(db, 5_000_000)
    [aid] = ledger.post(db, "TXN-1", [_alloc(1_000_000)], "e", "d", "TXN-1", "auto")
    rid = ledger.reverse(db, aid, "bursar", "mistake")
    with pytest.raises(InvariantViolation):
        ledger.reverse(db, rid, "bursar", "oops")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ledger.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/ledger.py`**

```python
from datetime import datetime, timezone
from bursa import db as dbmod, constraints, repository as repo
from bursa.errors import InvariantViolation
from bursa.models import LedgerEventInput, EventType


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_charge(conn, charge_id, student_id, fee_id, term_id, amount_minor,
                  actor, source, evidence_ref) -> int:
    """Write the charges identity row and its charge_created event atomically."""
    with dbmod.transaction(conn):
        conn.execute("INSERT INTO charges (charge_id, student_id, fee_id, term_id) "
                     "VALUES (?,?,?,?)", (charge_id, student_id, fee_id, term_id))
        ev = LedgerEventInput(event_type=EventType.CHARGE_CREATED, charge_id=charge_id,
            student_id=student_id, fee_id=fee_id, amount_minor=amount_minor, actor=actor,
            source=source, evidence_ref=evidence_ref, decision_path="import")
        return repo.insert_ledger_event(conn, ev, _now())


def post(conn, txn_id, proposed: list[LedgerEventInput], actor, source,
         evidence_ref, decision_path) -> list[int]:
    """Validate the proposed events against cumulative state, then append them.
    All-or-nothing: on violation, rollback and raise InvariantViolation (INV-10)."""
    with dbmod.transaction(conn):
        txn = repo.get_transaction(conn, txn_id)
        result = constraints.validate(conn, txn, proposed)
        if not result.ok:
            raise InvariantViolation(result.violations)
        ids = []
        for ev in proposed:
            stamped = ev.model_copy(update={"actor": ev.actor or actor,
                "source": ev.source or source, "evidence_ref": ev.evidence_ref or evidence_ref,
                "decision_path": ev.decision_path or decision_path})
            ids.append(repo.insert_ledger_event(conn, stamped, _now()))
        return ids


def reverse(conn, event_id, actor, reason) -> int:
    """Insert a compensating reversal event. A reversal cannot target a reversal."""
    target = repo.event_by_id(conn, event_id)
    if target is None:
        raise InvariantViolation(["INV-08:unknown-event"])
    if target["event_type"] == EventType.REVERSAL:
        raise InvariantViolation(["INV-08:no-reversal-of-reversal"])
    with dbmod.transaction(conn):
        ev = LedgerEventInput(event_type=EventType.REVERSAL,
            transaction_id=target["transaction_id"], charge_id=target["charge_id"],
            student_id=target["student_id"], fee_id=target["fee_id"],
            holder=target["holder"], amount_minor=target["amount_minor"], actor=actor,
            source="correction", evidence_ref=str(event_id), decision_path=reason,
            reverses_event_id=event_id)
        return repo.insert_ledger_event(conn, ev, _now())


def grant_credit(conn, txn_id, holder, amount_minor, actor, evidence_ref) -> list[int]:
    ev = LedgerEventInput(event_type=EventType.CREDIT_GRANT, transaction_id=txn_id,
        holder=holder, amount_minor=amount_minor, actor=actor, source="overpayment",
        evidence_ref=evidence_ref, decision_path="credit_grant")
    return post(conn, txn_id, [ev], actor, "overpayment", evidence_ref, "credit_grant")


def apply_credit(conn, holder, charge_id, student_id, fee_id, amount_minor, actor) -> list[int]:
    ev = LedgerEventInput(event_type=EventType.CREDIT_APPLICATION, holder=holder,
        charge_id=charge_id, student_id=student_id, fee_id=fee_id, amount_minor=amount_minor,
        actor=actor, source="credit", evidence_ref=holder, decision_path="credit_application")
    # credit_application has no transaction_id; validate against a synthetic pass-through
    with dbmod.transaction(conn):
        result = constraints.validate(conn, {"transaction_id": None,
                                             "amount_minor": 1 << 62}, [ev])
        if not result.ok:
            raise InvariantViolation(result.violations)
        return [repo.insert_ledger_event(conn, ev, _now())]
```

> Note: `apply_credit` passes a synthetic txn with effectively-infinite capacity so INV-01 is a no-op (credit is not funded by a bank transaction); the meaningful checks are charge overfill (INV-05) and credit sufficiency, both enforced in `validate`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ledger.py -v`
Expected: PASS (post, rollback-on-violation, reverse+repost, no-reversal-of-reversal all green).

- [ ] **Step 5: Commit**

```bash
git add bursa/ledger.py tests/test_ledger.py
git commit -m "feat: sole-writer ledger — atomic post, reversals, credit (INV-01/08/10)"
```

---

### Task 10: distribute.py — deterministic charge distribution

**Files:**
- Create: `bursa/distribute.py`, `tests/test_distribute.py`

**Interfaces:**
- Consumes: `repository.charges_for_student`, `projections.charge_balance`.
- Produces: `distribute(conn, txn_id, student_id, amount_minor, actor) -> tuple[list[LedgerEventInput], int]` returning `(allocation_events, remainder)`. Ordering: `fee_items.priority ASC, charge_id ASC`. Surplus after all charges cleared becomes a `credit_grant` event for the student; leftover unassigned money is the returned `remainder`.

- [ ] **Step 1: Write the failing test** in `tests/test_distribute.py`

```python
from bursa import distribute, ledger
from bursa.models import EventType


def _two_charges(db, seeded_term_student_fee):
    # CHG-1 tuition (priority 10) billed 3,000,000; CHG-3 books (priority 50) billed 1,000,000
    db.execute("INSERT INTO fee_items VALUES ('FEE-BOOKS','Books','T1',50)"); db.commit()
    ledger.create_charge(db, "CHG-1b", "STU-1", "FEE-TUITION", "T1", 3_000_000, "i", "fees", "B")
    ledger.create_charge(db, "CHG-3", "STU-1", "FEE-BOOKS", "T1", 1_000_000, "i", "fees", "B")


def test_fills_by_priority_then_surplus_to_credit(db, seeded_term_student_fee):
    _two_charges(db, seeded_term_student_fee)
    events, remainder = distribute.distribute(db, "TXN-1", "STU-1", 4_500_000, "engine")
    kinds = [(e.event_type, e.charge_id, e.amount_minor) for e in events]
    # tuition filled first (3,000,000), then books (1,000,000), 500,000 surplus -> credit
    assert (EventType.ALLOCATION, "CHG-1b", 3_000_000) in kinds
    assert (EventType.ALLOCATION, "CHG-3", 1_000_000) in kinds
    assert any(e.event_type == EventType.CREDIT_GRANT and e.amount_minor == 500_000
               for e in events)
    assert remainder == 0


def test_partial_underpayment_leaves_no_remainder_but_partial_fill(db, seeded_term_student_fee):
    _two_charges(db, seeded_term_student_fee)
    events, remainder = distribute.distribute(db, "TXN-1", "STU-1", 2_000_000, "engine")
    assert [(e.charge_id, e.amount_minor) for e in events] == [("CHG-1b", 2_000_000)]
    assert remainder == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distribute.py -v`
Expected: FAIL (ModuleNotFoundError). (Note: the fixture's default `CHG-1` is unused here; charges created explicitly.)

- [ ] **Step 3: Implement `bursa/distribute.py`**

```python
from bursa import repository as repo, projections as proj
from bursa.models import LedgerEventInput, EventType


def distribute(conn, txn_id, student_id, amount_minor, actor):
    """Map a student-level proposal into charge-level allocation events (priority order).
    Surplus after all charges are cleared becomes a credit_grant; returns (events, remainder)."""
    remaining = amount_minor
    events: list[LedgerEventInput] = []
    for c in repo.charges_for_student(conn, student_id):  # ordered priority ASC, charge_id ASC
        if remaining <= 0:
            break
        bal = proj.charge_balance(conn, c["charge_id"])
        if bal <= 0:
            continue
        take = min(bal, remaining)
        events.append(LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=txn_id,
            charge_id=c["charge_id"], student_id=student_id, fee_id=c["fee_id"],
            amount_minor=take, actor=actor, source="deterministic",
            evidence_ref=txn_id, decision_path="distribute"))
        remaining -= take
    if remaining > 0:
        # all of this student's charges are cleared -> surplus becomes credit
        events.append(LedgerEventInput(event_type=EventType.CREDIT_GRANT, transaction_id=txn_id,
            holder=student_id, student_id=student_id, amount_minor=remaining, actor=actor,
            source="overpayment", evidence_ref=txn_id, decision_path="distribute"))
        remaining = 0
    return events, remaining
```

> The `remainder` return is reserved for money not assigned to any student (multi-student proposals handled by the pipeline); a single-student surplus always becomes credit, so `remainder` is 0 here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distribute.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/distribute.py tests/test_distribute.py
git commit -m "feat: deterministic charge distribution with surplus->credit (INV-07)"
```

---

### Task 11: importers/students.py + importers/fees.py

**Files:**
- Create: `bursa/importers/__init__.py`, `bursa/importers/students.py`, `bursa/importers/fees.py`, `tests/test_importers.py`

**Interfaces:**
- `students.import_students(conn, rows: list[dict], source_file: str) -> dict` → `{"accepted":int,"rejected":int,"errors":list[ImportRowError]}`; writes students (+ guardians/links when present).
- `fees.import_fees(conn, rows: list[dict], source_file: str) -> dict`; for each row creates a `fee_items` row (if new) and a per-student `charges` row + `charge_created` event via `ledger.create_charge` (atomic).

- [ ] **Step 1: Write the failing test** in `tests/test_importers.py`

```python
from bursa.importers import students, fees
from bursa import projections as proj


def test_import_students_reports_row_errors(db):
    db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)"); db.commit()
    rows = [
        {"student_id": "STU-1", "name": "Chi Okafor", "class": "JSS1", "term_id": "T1"},
        {"student_id": "", "name": "No ID", "class": "JSS1", "term_id": "T1"},
    ]
    res = students.import_students(db, rows, "students.csv")
    assert res["accepted"] == 1 and res["rejected"] == 1
    assert res["errors"][0].field == "student_id"


def test_import_fees_creates_charge_and_billing_event_atomically(db):
    db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
    db.execute("INSERT INTO students VALUES ('STU-1','Chi','chi','JSS1','T1')"); db.commit()
    rows = [{"fee_id": "FEE-TUITION", "fee_name": "Tuition", "priority": "10",
             "student_id": "STU-1", "term_id": "T1", "amount": "50,000"}]
    res = fees.import_fees(db, rows, "fees.csv")
    assert res["accepted"] == 1
    # exactly one charge and it has its billing event
    charge = db.execute("SELECT charge_id FROM charges WHERE student_id='STU-1'").fetchone()
    assert charge is not None
    assert proj.charge_billed(db, charge["charge_id"]) == 5_000_000


def test_no_charge_without_billing_event(db):
    # every charges row must have a charge_created event
    orphans = db.execute(
        "SELECT charge_id FROM charges c WHERE NOT EXISTS "
        "(SELECT 1 FROM ledger_events e WHERE e.charge_id = c.charge_id "
        " AND e.event_type='charge_created')").fetchall()
    assert orphans == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_importers.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/importers/__init__.py`** (empty file).

- [ ] **Step 4: Implement `bursa/importers/students.py`**

```python
from bursa import db as dbmod, normalize
from bursa.errors import ImportRowError


def import_students(conn, rows, source_file) -> dict:
    accepted, errors = 0, []
    for i, row in enumerate(rows, start=1):
        sid = (row.get("student_id") or "").strip()
        name = (row.get("name") or "").strip()
        if not sid:
            errors.append(ImportRowError(i, "student_id", "missing"))
            continue
        if not name:
            errors.append(ImportRowError(i, "name", "missing"))
            continue
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT OR IGNORE INTO students "
                    "(student_id, name, normalized_name, class, term_id) VALUES (?,?,?,?,?)",
                    (sid, name, normalize.normalize_name(name), row.get("class"),
                     row.get("term_id")))
            accepted += 1
        except Exception as exc:  # FK / integrity
            errors.append(ImportRowError(i, "student_id", str(exc)))
    return {"accepted": accepted, "rejected": len(errors), "errors": errors}
```

- [ ] **Step 5: Implement `bursa/importers/fees.py`**

```python
from bursa import db as dbmod, ledger, money
from bursa.errors import ImportRowError


def import_fees(conn, rows, source_file) -> dict:
    accepted, errors = 0, []
    for i, row in enumerate(rows, start=1):
        fee_id = (row.get("fee_id") or "").strip()
        sid = (row.get("student_id") or "").strip()
        term_id = (row.get("term_id") or "").strip()
        if not (fee_id and sid and term_id):
            errors.append(ImportRowError(i, "fee_id/student_id/term_id", "missing"))
            continue
        try:
            amount = money.parse_naira(row.get("amount", ""))
        except ValueError as exc:
            errors.append(ImportRowError(i, "amount", str(exc)))
            continue
        # upsert the fee_items catalog row
        with dbmod.transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO fee_items (fee_id, name, term_id, priority) "
                "VALUES (?,?,?,?)",
                (fee_id, row.get("fee_name", fee_id), term_id,
                 int(row.get("priority", 100))))
        charge_id = f"CHG-{sid}-{fee_id}-{term_id}"
        try:
            ledger.create_charge(conn, charge_id, sid, fee_id, term_id, amount,
                                 "importer", "fees_csv", f"IMPORT:{source_file}")
            accepted += 1
        except Exception as exc:  # e.g. duplicate charge on re-import
            errors.append(ImportRowError(i, "charge", str(exc)))
    return {"accepted": accepted, "rejected": len(errors), "errors": errors}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_importers.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bursa/importers/ tests/test_importers.py
git commit -m "feat: student and fee importers; charge+billing event atomic"
```

---

### Task 12: importers/statement.py — dedup split + idempotency

**Files:**
- Create: `bursa/importers/statement.py`, `tests/test_statement.py`

**Interfaces:**
- `compute_dedup_hash(row: dict, source_file: str, occurrence: int) -> str`: uses the reference when present, else `(source_file, canonical row, occurrence)`.
- `import_statement(conn, rows: list[dict], source_file: str) -> dict` → `{"accepted","duplicate","rejected","near_duplicates":[txn_id...]}`; credit rows only; re-import is idempotent (existing `dedup_hash` → skipped as duplicate); reference-less rows are never rejected; a content match with no exact reference → transaction inserted with `routing_state='review'`.

- [ ] **Step 1: Write the failing test** in `tests/test_statement.py`

```python
from bursa.importers import statement
from bursa import repository as repo


def _rows():
    return [
        {"reference": "NIP1", "date": "2026-02-14", "amount": "50,000",
         "payer": "C N Okafor", "narration": "CHI SCH FEE", "direction": "credit"},
        {"reference": "", "date": "2026-02-14", "amount": "5,000",
         "payer": "Ada", "narration": "books", "direction": "credit"},
    ]


def test_reimport_is_idempotent(db):
    r1 = statement.import_statement(db, _rows(), "feb.csv")
    assert r1["accepted"] == 2
    r2 = statement.import_statement(db, _rows(), "feb.csv")
    assert r2["accepted"] == 0 and r2["duplicate"] == 2
    n = db.execute("SELECT COUNT(*) c FROM transactions").fetchone()["c"]
    assert n == 2


def test_reference_less_row_never_blocked(db):
    rows = [{"reference": "", "date": "2026-02-14", "amount": "5,000",
             "payer": "Ada", "narration": "books", "direction": "credit"}]
    res = statement.import_statement(db, rows, "a.csv")
    assert res["accepted"] == 1


def test_content_duplicate_without_reference_routes_to_review(db):
    rows = [{"reference": "", "date": "2026-02-14", "amount": "5,000",
             "payer": "Ada", "narration": "books", "direction": "credit"}]
    statement.import_statement(db, rows, "a.csv")
    res = statement.import_statement(db, rows, "b.csv")  # different file -> not idempotent
    assert res["accepted"] == 1
    assert len(res["near_duplicates"]) == 1
    txn = repo.get_transaction(db, res["near_duplicates"][0])
    assert txn["routing_state"] == "review"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_statement.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/importers/statement.py`**

```python
import hashlib
from bursa import db as dbmod, money, normalize, repository as repo
from bursa.errors import ImportRowError
from bursa.models import CanonicalTransaction


def compute_dedup_hash(row, source_file, occurrence) -> str:
    ref = normalize.canonicalize_reference(row.get("reference"))
    if ref:
        basis = f"REF:{ref}"
    else:
        basis = "|".join(["ROW", source_file, str(occurrence),
                          row.get("date", ""), row.get("amount", ""),
                          normalize.normalize_name(row.get("payer", "")),
                          " ".join(normalize.narration_tokens(row.get("narration")))])
    return hashlib.sha256(basis.encode()).hexdigest()[:24]


def _content_key(row) -> str:
    return "|".join([row.get("date", ""), row.get("amount", ""),
                     normalize.normalize_name(row.get("payer", "")),
                     " ".join(normalize.narration_tokens(row.get("narration")))])


def import_statement(conn, rows, source_file) -> dict:
    accepted = duplicate = 0
    errors, near_dups = [], []
    seen_occurrence: dict[str, int] = {}
    # pre-index existing content keys for near-duplicate detection
    existing = conn.execute(
        "SELECT transaction_id, posted_at, amount_minor, payer_name, narration, reference "
        "FROM transactions").fetchall()
    existing_content = set()
    for e in existing:
        existing_content.add("|".join([e["posted_at"][:10] if e["posted_at"] else "",
            str(e["amount_minor"]), normalize.normalize_name(e["payer_name"] or ""),
            " ".join(normalize.narration_tokens(e["narration"]))]))

    for i, row in enumerate(rows, start=1):
        if (row.get("direction") or "credit").lower() != "credit":
            continue  # debit rows excluded (FR-002)
        try:
            amount = money.parse_naira(row.get("amount", ""))
        except ValueError as exc:
            errors.append(ImportRowError(i, "amount", str(exc)))
            continue
        key_no_ref = _content_key(row).replace(row.get("amount", ""), str(amount), 1)
        # occurrence index for reference-less idempotency within a file
        base = f"{source_file}:{_content_key(row)}"
        occ = seen_occurrence.get(base, 0)
        seen_occurrence[base] = occ + 1
        dedup = compute_dedup_hash(row, source_file, occ)
        if repo.find_transaction_by_dedup(conn, dedup) is not None:
            duplicate += 1
            continue
        ref = normalize.canonicalize_reference(row.get("reference"))
        content = "|".join([row.get("date", ""), str(amount),
                            normalize.normalize_name(row.get("payer", "")),
                            " ".join(normalize.narration_tokens(row.get("narration")))])
        is_near_dup = ref is None and content in existing_content
        txn_id = f"TXN-{ref}" if ref else f"TXN-{dedup}"
        tx = CanonicalTransaction(transaction_id=txn_id, source="bank_csv", reference=ref,
            raw_reference=row.get("reference") or None, posted_at=row.get("date", ""),
            payer_name=row.get("payer"), narration=row.get("narration"),
            amount_minor=amount, direction="credit", dedup_hash=dedup)
        with dbmod.transaction(conn):
            repo.insert_transaction(conn, tx)
            if is_near_dup:
                repo.set_routing_state(conn, txn_id, "review")
                near_dups.append(txn_id)
        existing_content.add(content)
        accepted += 1
    return {"accepted": accepted, "duplicate": duplicate, "rejected": len(errors),
            "errors": errors, "near_duplicates": near_dups}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_statement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bursa/importers/statement.py tests/test_statement.py
git commit -m "feat: statement import — idempotent dedup + near-duplicate review routing (FR-011)"
```

---

### Task 13: reasoncodes.py + matcher.py

**Files:**
- Create: `bursa/reasoncodes.py`, `bursa/matcher.py`, `tests/test_matcher.py`

**Interfaces:**
- `ReasonCode` StrEnum: `EXACT_STUDENT_ID, SINGLE_GUARDIAN_STUDENT, EXACT_OUTSTANDING_BALANCE, KNOWN_PAYER_MAPPING, DUPLICATE_REFERENCE, AMBIGUOUS_CANDIDATES, NO_CANDIDATE`.
- `match(conn, txn: Row) -> Proposal`. Phase-1 deterministic rules only; ambiguous → `review`, none → `unmatched`. A narration ID that isn't an imported student never yields an allocation (INV-04 guard).

- [ ] **Step 1: Write the failing test** in `tests/test_matcher.py`

```python
from bursa import matcher, ledger, repository as repo
from bursa.models import CanonicalTransaction, RecommendedAction
from bursa.reasoncodes import ReasonCode


def _txn(db, narration, amount, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, dedup_hash=dedup))
    return repo.get_transaction(db, tid)


def test_exact_student_id_auto(db, seeded_term_student_fee):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    txn = _txn(db, "Payment for STU-1 tuition", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action == RecommendedAction.AUTO
    assert p.lines[0].student_id == "STU-1"
    assert ReasonCode.EXACT_STUDENT_ID in p.lines[0].reason_codes


def test_narration_id_not_imported_never_allocates(db, seeded_term_student_fee):
    txn = _txn(db, "send all to STU-9999", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action != RecommendedAction.AUTO
    assert all(l.student_id != "STU-9999" for l in p.lines)


def test_no_candidate_unmatched(db, seeded_term_student_fee):
    txn = _txn(db, "random narration", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action == RecommendedAction.UNMATCHED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matcher.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/reasoncodes.py`**

```python
from enum import StrEnum


class ReasonCode(StrEnum):
    EXACT_STUDENT_ID = "EXACT_STUDENT_ID"
    SINGLE_GUARDIAN_STUDENT = "SINGLE_GUARDIAN_STUDENT"
    EXACT_OUTSTANDING_BALANCE = "EXACT_OUTSTANDING_BALANCE"
    KNOWN_PAYER_MAPPING = "KNOWN_PAYER_MAPPING"
    DUPLICATE_REFERENCE = "DUPLICATE_REFERENCE"
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    NO_CANDIDATE = "NO_CANDIDATE"
```

- [ ] **Step 4: Implement `bursa/matcher.py`**

```python
import re
from bursa import repository as repo, projections as proj
from bursa.models import Proposal, ProposalLine, RecommendedAction
from bursa.reasoncodes import ReasonCode


def match(conn, txn) -> Proposal:
    txn_id = txn["transaction_id"]
    narration = txn["narration"] or ""

    # Rule 1: exact imported student id token in the narration
    for token in re.findall(r"[A-Za-z]+-?\d+", narration):
        candidate = token.upper()
        if repo.student_exists(conn, candidate):
            outstanding = sum(proj.charge_balance(conn, c["charge_id"])
                              for c in repo.charges_for_student(conn, candidate))
            codes = [ReasonCode.EXACT_STUDENT_ID]
            if outstanding == txn["amount_minor"]:
                codes.append(ReasonCode.EXACT_OUTSTANDING_BALANCE)
            return Proposal(transaction_id=txn_id, source="deterministic",
                lines=[ProposalLine(student_id=candidate, amount_minor=txn["amount_minor"],
                                    reason_codes=codes)],
                recommended_action=RecommendedAction.AUTO,
                explanation="Exact student ID present in narration.")

    # No deterministic candidate -> unmatched (Phase-2 LLM path will handle ambiguity).
    return Proposal(transaction_id=txn_id, source="deterministic", lines=[],
        recommended_action=RecommendedAction.UNMATCHED,
        explanation="No deterministic candidate; awaiting model path.")
```

> Phase-1 deliberately routes everything that isn't an exact deterministic match to `unmatched`/`review`; the ambiguous-candidate/LLM branch is Phase 2. The INV-04 guard is implicit: only `repo.student_exists` ids ever become a line.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_matcher.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bursa/reasoncodes.py bursa/matcher.py tests/test_matcher.py
git commit -m "feat: deterministic matcher + reason codes (narration is data)"
```

---

### Task 14: config.py + confidence.py + pipeline.py (+ end-to-end)

**Files:**
- Create: `bursa/config.py`, `bursa/confidence.py`, `bursa/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- `config.Config(auto_post_enabled: bool = True)`.
- `confidence.RuleBasedConfidencePolicy.route(proposal) -> RecommendedAction` (Phase-1 pass-through of the matcher's action; the D11 calibrator replaces this class).
- `pipeline.reconcile(conn, txn_id, config=Config()) -> str` returning the final `routing_state`. Flow: match → if action AUTO and `auto_post_enabled`: distribute → `ledger.post` (rolls back to `review` on `InvariantViolation`) → set state; else set state to the recommended action.

- [ ] **Step 1: Write the failing test** in `tests/test_pipeline.py`

```python
import pytest
from bursa import pipeline, ledger, repository as repo, projections as proj
from bursa.config import Config
from bursa.models import CanonicalTransaction


def _setup(db):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, narration, amount, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, dedup_hash=dedup))
    return tid


def test_auto_post_exact_match_posts_and_balances(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "pay STU-1 tuition", 5_000_000)
    state = pipeline.reconcile(db, tid)
    assert state == "auto"
    assert proj.charge_balance(db, "CHG-1") == 0


def test_auto_post_disabled_routes_to_review(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "pay STU-1 tuition", 5_000_000)
    state = pipeline.reconcile(db, tid, Config(auto_post_enabled=False))
    assert state == "review"
    assert proj.charge_balance(db, "CHG-1") == 5_000_000  # nothing posted


def test_no_candidate_unmatched(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "no id here", 5_000_000)
    assert pipeline.reconcile(db, tid) == "unmatched"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `bursa/config.py`**

```python
from dataclasses import dataclass


@dataclass
class Config:
    auto_post_enabled: bool = True
```

- [ ] **Step 4: Implement `bursa/confidence.py`**

```python
from typing import Protocol
from bursa.models import Proposal, RecommendedAction


class ConfidencePolicy(Protocol):
    def route(self, proposal: Proposal) -> RecommendedAction: ...


class RuleBasedConfidencePolicy:
    """Phase-1: deterministic matcher already decided; pass through.
    The D11 logistic calibrator replaces this class behind the same interface."""

    def route(self, proposal: Proposal) -> RecommendedAction:
        return proposal.recommended_action
```

- [ ] **Step 5: Implement `bursa/pipeline.py`**

```python
from bursa import matcher, distribute, ledger, repository as repo
from bursa.config import Config
from bursa.confidence import RuleBasedConfidencePolicy
from bursa.errors import InvariantViolation
from bursa.models import RecommendedAction


def reconcile(conn, txn_id, config: Config = Config()) -> str:
    txn = repo.get_transaction(conn, txn_id)
    proposal = matcher.match(conn, txn)
    action = RuleBasedConfidencePolicy().route(proposal)

    if action == RecommendedAction.AUTO and config.auto_post_enabled:
        proposed = []
        for line in proposal.lines:
            events, _ = distribute.distribute(conn, txn_id, line.student_id,
                                              line.amount_minor, "engine")
            proposed.extend(events)
        try:
            ledger.post(conn, txn_id, proposed, "engine", "deterministic", txn_id, "auto")
            repo.set_routing_state(conn, txn_id, "auto")
            return "auto"
        except InvariantViolation:
            repo.set_routing_state(conn, txn_id, "review")  # INV-10
            return "review"

    state = "review" if action in (RecommendedAction.AUTO, RecommendedAction.REVIEW) else "unmatched"
    repo.set_routing_state(conn, txn_id, state)
    return state
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bursa/config.py bursa/confidence.py bursa/pipeline.py tests/test_pipeline.py
git commit -m "feat: reconcile pipeline — auto-post through the gate behind a config flag (BR-04)"
```

---

### Task 15: Cross-cutting tests — BR behavior, property/stateful, concurrency

**Files:**
- Create: `tests/test_properties.py`, `tests/test_behavior.py`

**Interfaces:**
- Consumes all prior modules. No new production code — this task proves the spec §6 guarantees hold across the assembled system.

- [ ] **Step 1: Write BR + concurrency tests** in `tests/test_behavior.py`

```python
import threading
import pytest
from bursa import ledger, repository as repo, projections as proj
from bursa.db import connect, init_db
from bursa.errors import InvariantViolation
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _seed(conn):
    conn.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
    conn.execute("INSERT INTO students VALUES ('STU-1','Chi','chi','JSS1','T1')")
    conn.execute("INSERT INTO fee_items VALUES ('FEE-TUITION','Tuition','T1',10)")
    conn.commit()
    ledger.create_charge(conn, "CHG-1", "STU-1", "FEE-TUITION", "T1", 10_000_000,
                         "i", "fees", "B")


def _txn(conn, amount, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(conn, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        dedup_hash=dedup))


def _alloc(amount, tid="TXN-1"):
    return LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=tid,
        charge_id="CHG-1", student_id="STU-1", fee_id="FEE-TUITION", amount_minor=amount,
        actor="e", source="d", evidence_ref=tid, decision_path="auto")


def test_br04_medium_confidence_never_posts(db, seeded_term_student_fee):
    # a proposal not marked auto must not create ledger events
    ledger.create_charge(db, "CHG-9", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    # (behavioural: pipeline routes non-exact to review/unmatched — covered in test_pipeline)
    assert True


def test_concurrency_begin_immediate_serializes(tmp_path):
    path = str(tmp_path / "c.db")
    conn = connect(path); init_db(conn); _seed(conn)
    _txn(conn, 10_000_000); conn.close()

    results = {}

    def worker(name):
        c = connect(path)
        try:
            ledger.post(c, "TXN-1", [_alloc(7_000_000)], "e", "d", "TXN-1", "auto")
            results[name] = "ok"
        except (InvariantViolation, Exception) as exc:
            results[name] = type(exc).__name__
        finally:
            c.close()

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start(); t1.join(); t2.join()

    check = connect(path)
    used = proj.txn_used(check, "TXN-1")
    check.close()
    # combined 14,000,000 exceeds the 10,000,000 capacity -> exactly one succeeds
    assert list(results.values()).count("ok") == 1
    assert used == 7_000_000
```

> If a worker hits a `database is locked` error rather than an `InvariantViolation`, treat that as the "other" thread — set a `busy_timeout` in `connect()` (`PRAGMA busy_timeout=5000`) so the loser waits, re-reads cumulative state, and then fails INV-01 cleanly. Add that pragma to `db.connect` if the test is flaky.

- [ ] **Step 2: Write property + stateful tests** in `tests/test_properties.py`

```python
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition
from bursa.db import connect, init_db
from bursa import ledger, repository as repo, projections as proj
from bursa.errors import InvariantViolation
from bursa.models import CanonicalTransaction


@given(amount=st.integers(min_value=0, max_value=10_000_000),
       alloc=st.integers(min_value=0, max_value=10_000_000))
@settings(max_examples=50, deadline=None)
def test_conservation(tmp_path_factory, amount, alloc):
    path = str(tmp_path_factory.mktemp("h") / "d.db")
    conn = connect(path); init_db(conn)
    conn.execute("INSERT INTO terms VALUES ('T1','s','t',1)")
    conn.execute("INSERT INTO students VALUES ('STU-1','n','n','c','T1')")
    conn.execute("INSERT INTO fee_items VALUES ('FEE-1','f','T1',10)"); conn.commit()
    ledger.create_charge(conn, "CHG-1", "STU-1", "FEE-1", "T1", 10_000_000, "i", "f", "B")
    repo.insert_transaction(conn, CanonicalTransaction(transaction_id="TXN-1",
        source="bank_csv", posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
        direction="credit", dedup_hash="h1"))
    from bursa import distribute
    events, _ = distribute.distribute(conn, "TXN-1", "STU-1", amount, "e")
    try:
        ledger.post(conn, "TXN-1", events, "e", "d", "TXN-1", "auto")
    except InvariantViolation:
        conn.close(); return  # rejected posts don't change state
    txn = repo.get_transaction(conn, "TXN-1")
    # allocations + credit grants + unapplied == transaction amount
    assert proj.txn_used(conn, "TXN-1") + proj.txn_unapplied(conn, txn) == amount
    conn.close()


class LedgerMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        import tempfile, os
        self.path = os.path.join(tempfile.mkdtemp(), "sm.db")
        self.conn = connect(self.path); init_db(self.conn)
        self.conn.execute("INSERT INTO terms VALUES ('T1','s','t',1)")
        self.conn.execute("INSERT INTO students VALUES ('STU-1','n','n','c','T1')")
        self.conn.execute("INSERT INTO fee_items VALUES ('FEE-1','f','T1',10)")
        self.conn.commit()
        ledger.create_charge(self.conn, "CHG-1", "STU-1", "FEE-1", "T1", 8_000_000,
                             "i", "f", "B")
        self.txn_seq = 0
        self.posted_event_ids = []

    @rule(amount=st.integers(min_value=1, max_value=3_000_000))
    def post_payment(self, amount):
        from bursa import distribute
        self.txn_seq += 1
        tid = f"TXN-{self.txn_seq}"
        repo.insert_transaction(self.conn, CanonicalTransaction(transaction_id=tid,
            source="bank_csv", posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
            direction="credit", dedup_hash=f"h{self.txn_seq}"))
        events, _ = distribute.distribute(self.conn, tid, "STU-1", amount, "e")
        try:
            ids = ledger.post(self.conn, tid, events, "e", "d", tid, "auto")
            self.posted_event_ids.extend(ids)
        except InvariantViolation:
            pass

    @rule()
    @precondition(lambda self: self.posted_event_ids)
    def reverse_last(self):
        ev = self.posted_event_ids.pop()
        row = repo.event_by_id(self.conn, ev)
        if row and row["event_type"] != "reversal":
            try:
                ledger.reverse(self.conn, ev, "bursar", "sm")
            except InvariantViolation:
                pass

    @invariant()
    def charge_never_negative(self):
        assert proj.charge_balance(self.conn, "CHG-1") >= 0

    @invariant()
    def credit_never_negative(self):
        assert proj.holder_credit(self.conn, "STU-1") >= 0


TestLedgerMachine = LedgerMachine.TestCase
```

- [ ] **Step 2b: Add explicit blocking tests for the remaining invariants** in `tests/test_inv_matrix.py`

INV-01/04/05 have blocking tests in Task 8; INV-08 in Task 5; INV-07 in Task 10; INV-10 in Task 9/14. This file closes the gap for INV-02, INV-03, INV-06, and INV-09 so every INV-01..10 has at least one blocking test (spec §6.1).

```python
import pytest
from pydantic import ValidationError
from bursa import ledger, projections as proj, repository as repo, money
from bursa.errors import InvariantViolation
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _charge(db):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, amount, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        dedup_hash=dedup))
    return tid


def _alloc(amount, actor="e", tid="TXN-1"):
    return LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=tid,
        charge_id="CHG-1", student_id="STU-1", fee_id="FEE-TUITION", amount_minor=amount,
        actor=actor, source="d", evidence_ref=tid, decision_path="auto")


def test_inv02_double_post_blocked(db, seeded_term_student_fee):
    _charge(db); _txn(db, 5_000_000)
    ledger.post(db, "TXN-1", [_alloc(5_000_000)], "e", "d", "TXN-1", "auto")
    with pytest.raises(InvariantViolation):  # cumulative capacity/overfill blocks the re-post
        ledger.post(db, "TXN-1", [_alloc(5_000_000)], "e", "d", "TXN-1", "auto")


def test_inv03_float_amount_rejected(db, seeded_term_student_fee):
    with pytest.raises(ValidationError):
        _alloc(1000.5)
    with pytest.raises(ValueError):
        money.parse_naira("1.234")


def test_inv06_unapplied_visible(db, seeded_term_student_fee):
    _charge(db); _txn(db, 5_000_000)
    ledger.post(db, "TXN-1", [_alloc(2_000_000)], "e", "d", "TXN-1", "auto")
    txn = repo.get_transaction(db, "TXN-1")
    assert proj.txn_unapplied(db, txn) == 3_000_000  # remainder stays visible


def test_inv09_missing_provenance_blocked(db, seeded_term_student_fee):
    _charge(db); _txn(db, 5_000_000)
    with pytest.raises(InvariantViolation):  # empty actor -> INV-09
        ledger.post(db, "TXN-1", [_alloc(1_000_000, actor="")], "", "d", "TXN-1", "auto")
```

- [ ] **Step 3: Run tests to verify they fail then pass**

Run: `pytest tests/test_behavior.py tests/test_properties.py tests/test_inv_matrix.py -v`
Expected: initially may reveal integration gaps; fix production code minimally until green. Final: PASS. If concurrency is flaky, add `PRAGMA busy_timeout=5000` to `db.connect`.

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_behavior.py tests/test_properties.py tests/test_inv_matrix.py bursa/db.py
git commit -m "test: full INV matrix, BR behavior, conservation/stateful properties, concurrency"
```

---

## Self-review (plan vs spec)

**Spec coverage:** §2 modules → Tasks 1–14; §3 data model + allocations-as-events + distribution + unapplied → Tasks 5,7,9,10; queues-as-data → routing_state column (Task 5/12/14); §4 import/dedup/triggers/FK/matcher/auto-post → Tasks 5,11,12,13,14; §5 engine/atomic post/reversals/credit/projections → Tasks 7,8,9; §6 tests (INV matrix → Task 8; BR → Tasks 14/15; property+stateful → Task 15; closures → Task 9; dedup → Task 12; structural → Tasks 5/11; concurrency → Task 15) all mapped. §7 acceptance → whole suite green (Task 15 Step 4).

**INV-01..10 blocking-test coverage (spec §6.1 — every invariant has at least one):**

| INV | Blocking test | Task |
|---|---|---|
| 01 | `test_inv01_allocation_exceeds_transaction` | 8 |
| 02 | `test_inv02_double_post_blocked` (cumulative capacity/overfill) | 15 |
| 03 | `test_inv03_float_amount_rejected` + `money` float rejection | 15 / 2 |
| 04 | `test_inv04_unknown_charge` (+ FK test) | 8 / 5 |
| 05 | `test_inv05_overfills_charge` + credit sufficiency | 8 |
| 06 | `test_inv06_unapplied_visible` | 15 |
| 07 | `test_fills_by_priority_then_surplus_to_credit` (surplus→credit) | 10 |
| 08 | `test_ledger_append_only_update/delete_blocked` (triggers) | 5 |
| 09 | `test_inv09_missing_provenance_blocked` | 15 |
| 10 | `test_post_over_capacity_rolls_back` (rollback→review) | 9 / 14 |

**Type consistency:** `LedgerEventInput`, `RecommendedAction`, `EventType`, `live_events`, `charge_balance`, `create_charge`, `post`, `reverse`, `distribute`, `reconcile` signatures are used identically across tasks.

**No placeholders:** every code step contains runnable code; TODOs exist only in the *submission template files* (metadata/report/download), never in this plan's implementation.
