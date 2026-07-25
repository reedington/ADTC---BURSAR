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
    # RAISE(ABORT) in a trigger surfaces as sqlite3.IntegrityError in Python.
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE ledger_events SET amount_minor = 1 WHERE event_id = ?",
                   (seeded_ledger_event,))


def test_ledger_append_only_delete_blocked(db, seeded_ledger_event):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM ledger_events WHERE event_id = ?", (seeded_ledger_event,))


def test_wal_mode(db):
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
