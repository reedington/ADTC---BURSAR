from bursa import ledger, pipeline, projections, repository as repo, review
from bursa.models import CanonicalTransaction


def _overpayment_proposal(db):
    ledger.create_charge(
        db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000,
        "importer", "fees", "BATCH-1",
    )
    repo.insert_transaction(
        db,
        CanonicalTransaction(
            transaction_id="TXN-1",
            source="bank_csv",
            posted_at="2026-02-14",
            amount_minor=6_000_000,
            direction="credit",
            narration="fees STU-1",
            dedup_hash="h1",
        ),
    )
    assert pipeline.reconcile(db, "TXN-1") == "review"
    return db.execute(
        "SELECT proposal_id FROM proposals WHERE transaction_id='TXN-1' "
        "AND status='pending'"
    ).fetchone()[0]


def test_review_keeps_partial_remainder_unapplied_without_explicit_credit(
        db, seeded_term_student_fee):
    proposal_id = _overpayment_proposal(db)
    review.approve(
        db,
        proposal_id,
        [{"student_id": "STU-1", "amount_minor": 6_000_000}],
        "bursar",
    )
    txn = repo.get_transaction(db, "TXN-1")
    assert projections.charge_balance(db, "CHG-1") == 0
    assert projections.txn_unapplied(db, txn) == 1_000_000
    assert projections.holder_credit(db, "STU-1") == 0


def test_review_creates_credit_only_after_explicit_holder_decision(
        db, seeded_term_student_fee):
    proposal_id = _overpayment_proposal(db)
    review.approve(
        db,
        proposal_id,
        [{"student_id": "STU-1", "amount_minor": 6_000_000}],
        "bursar",
        credit_holder="STU-1",
    )
    txn = repo.get_transaction(db, "TXN-1")
    assert projections.txn_unapplied(db, txn) == 0
    assert projections.holder_credit(db, "STU-1") == 1_000_000


def test_reject_retains_proposal_and_writes_no_ledger_events(
        db, seeded_term_student_fee):
    proposal_id = _overpayment_proposal(db)
    review.reject(db, proposal_id, "bursar")
    proposal = repo.get_proposal(db, proposal_id)
    assert proposal["status"] == "rejected"
    assert proposal["decision"] == "reject"
    assert repo.live_events(db, transaction_id="TXN-1") == []
