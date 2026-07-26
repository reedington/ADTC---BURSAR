import threading
from bursa import ledger, repository as repo, projections as proj, pipeline
from bursa.db import connect, init_db
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _seed(conn):
    conn.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
    conn.execute("INSERT INTO students VALUES ('STU-1','Chi','chi','JSS1','T1')")
    conn.execute("INSERT INTO fee_items VALUES ('FEE-TUITION','Tuition','T1',10)")
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


def test_br04_candidate_bearing_without_model_routes_review_and_never_posts(
        db, seeded_term_student_fee):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    repo.insert_transaction(db, CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=5_000_000, direction="credit",
        narration="no id here", dedup_hash="h1"))
    assert pipeline.reconcile(db, "TXN-1") == "review"
    assert repo.live_events(db, transaction_id="TXN-1") == []  # BR-04: nothing posted


def test_concurrency_begin_immediate_serializes(tmp_path):
    path = str(tmp_path / "c.db")
    conn = connect(path); init_db(conn); _seed(conn); _txn(conn, 10_000_000); conn.close()

    results = {}

    def worker(name):
        c = connect(path)
        try:
            ledger.post(c, "TXN-1", [_alloc(7_000_000)], "e", "d", "TXN-1", "auto")
            results[name] = "ok"
        except Exception as exc:  # InvariantViolation or a lock error -> the "other" thread
            results[name] = type(exc).__name__
        finally:
            c.close()

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start(); t1.join(); t2.join()

    check = connect(path)
    used = proj.txn_used(check, "TXN-1")
    check.close()
    # combined 14,000,000 exceeds the 10,000,000 capacity -> exactly one commits
    assert list(results.values()).count("ok") == 1
    assert used == 7_000_000
