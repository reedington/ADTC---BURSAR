import sqlite3

from bursa import db


def test_existing_v1_database_is_adopted_and_upgraded_idempotently(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(db.SCHEMA_V1_SQL)
    legacy.close()

    conn = db.connect(str(path))
    assert db.migrate(conn) == db.SCHEMA_VERSION
    assert db.migrate(conn) == db.SCHEMA_VERSION
    assert db.schema_version(conn) == 3
    proposal_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(proposals)").fetchall()
    }
    assert {
        "candidate_snapshot_json",
        "raw_output_json",
        "decision_allocations_json",
    } <= proposal_columns
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='import_errors'"
    ).fetchone()
    conn.close()
