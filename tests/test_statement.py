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
