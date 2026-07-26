import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA_VERSION = 3

SCHEMA_V1_SQL = """
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

MIGRATION_2_COLUMNS = {
    "proposals": [
        ("candidate_snapshot_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("evidence_snapshot_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("raw_output_json", "TEXT"),
        ("failure_reason", "TEXT"),
        ("ambiguities_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("decision", "TEXT"),
        ("decision_actor", "TEXT"),
        ("decision_at", "TEXT"),
        ("decision_allocations_json", "TEXT"),
        ("unapplied_minor", "INTEGER"),
        ("credit_holder", "TEXT"),
        ("last_inference_at", "TEXT"),
    ],
    "import_batches": [
        ("kind", "TEXT NOT NULL DEFAULT 'unknown'"),
        ("status", "TEXT NOT NULL DEFAULT 'complete'"),
        ("mapping_json", "TEXT NOT NULL DEFAULT '{}'"),
    ],
}

MIGRATION_2_SQL = """
CREATE TABLE IF NOT EXISTS import_errors (
  error_id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT NOT NULL REFERENCES import_batches(batch_id),
  row_number INTEGER NOT NULL,
  field TEXT NOT NULL,
  reason TEXT NOT NULL,
  raw_row_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_import_errors_batch ON import_errors(batch_id);
CREATE INDEX IF NOT EXISTS ix_proposals_transaction ON proposals(transaction_id, created_at);
"""

MIGRATION_3_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_ledger_events_one_reversal
  ON ledger_events(reverses_event_id) WHERE reverses_event_id IS NOT NULL;
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we manage txns
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _apply_migration_2(conn: sqlite3.Connection) -> None:
    for table, additions in MIGRATION_2_COLUMNS.items():
        existing = _columns(conn, table)
        for name, declaration in additions:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
    conn.executescript(MIGRATION_2_SQL)


def schema_version(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "schema_migrations"):
        return 0
    row = conn.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    return int(row["version"] or 0)


def migrate(conn: sqlite3.Connection) -> int:
    """Adopt a pre-migration Bursa database as v1 and upgrade it idempotently."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = schema_version(conn)
    if current == 0:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            (_now(),),
        )
        current = 1
    if current < 2:
        _apply_migration_2(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (2, ?)",
            (_now(),),
        )
        current = 2
    if current < 3:
        conn.executescript(MIGRATION_3_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, ?)",
            (_now(),),
        )
        current = 3
    return current


def init_db(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "terms"):
        conn.executescript(SCHEMA_V1_SQL)
    migrate(conn)


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
