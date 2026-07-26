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
    ledger.create_charge(db, "CHG-2", "STU-1", "FEE-TUITION", "T1", 4_000_000,
                         "importer", "fees_csv", "BATCH-1")
    assert proj.charge_billed(db, "CHG-2") == 4_000_000


def test_post_valid_allocation(db, charged):
    _txn(db, 5_000_000)
    ids = ledger.post(db, "TXN-1", [_alloc(5_000_000)], "engine", "deterministic",
                      "TXN-1", "auto")
    assert len(ids) == 1
    assert proj.charge_balance(db, "CHG-1") == 0


def test_post_over_capacity_rolls_back(db, charged):
    _txn(db, 3_000_000)
    with pytest.raises(InvariantViolation) as ei:
        ledger.post(db, "TXN-1", [_alloc(4_000_000)], "engine", "deterministic",
                    "TXN-1", "auto")
    assert "INV-01" in ei.value.violations
    assert repo.live_events(db, transaction_id="TXN-1") == []


def test_reverse_then_repost_same_amount_succeeds(db, charged):
    _txn(db, 5_000_000)
    [aid] = ledger.post(db, "TXN-1", [_alloc(2_000_000)], "e", "d", "TXN-1", "auto")
    ledger.reverse(db, aid, "bursar", "mistake")
    ledger.post(db, "TXN-1", [_alloc(2_000_000)], "e", "d", "TXN-1", "auto")
    assert proj.charge_paid(db, "CHG-1") == 2_000_000


def test_reverse_of_reversal_blocked(db, charged):
    _txn(db, 5_000_000)
    [aid] = ledger.post(db, "TXN-1", [_alloc(1_000_000)], "e", "d", "TXN-1", "auto")
    rid = ledger.reverse(db, aid, "bursar", "mistake")
    with pytest.raises(InvariantViolation):
        ledger.reverse(db, rid, "bursar", "oops")


def test_same_event_cannot_be_reversed_twice(db, charged):
    _txn(db, 5_000_000)
    aid = ledger.post(
        db, "TXN-1", [_alloc(5_000_000)], "bursar", "bank", "TXN-1", "review"
    )[0]
    ledger.reverse(db, aid, "bursar", "mistake")
    with pytest.raises(InvariantViolation):
        ledger.reverse(db, aid, "bursar", "duplicate correction")
