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
