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
