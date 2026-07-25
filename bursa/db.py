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
  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, features TEXT
);
CREATE TABLE proposal_allocations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
  student_id TEXT NOT NULL REFERENCES students(student_id),
  amount_minor INTEGER NOT NULL, reason_codes TEXT NOT NULL
);
CREATE TABLE student_aliases (
  student_id TEXT NOT NULL REFERENCES students(student_id),
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  PRIMARY KEY (student_id, normalized_alias)
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
    conn.execute("PRAGMA busy_timeout = 5000")
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
