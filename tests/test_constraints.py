from bursa import constraints, repository as repo
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _seed_charge(db, billed):
    db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=billed, actor="i", source="fees",
        evidence_ref="B", decision_path="import"), created_at="2026-01-01T00:00:00+00:00")


def _txn(db, amount):
    tx = CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
        direction="credit", dedup_hash="h1")
    repo.insert_transaction(db, tx)
    return repo.get_transaction(db, "TXN-1")


def _alloc(amount, charge="CHG-1"):
    return LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id="TXN-1",
        charge_id=charge, student_id="STU-1", fee_id="FEE-TUITION", amount_minor=amount,
        actor="engine", source="deterministic", evidence_ref="TXN-1", decision_path="auto")


def test_inv01_allocation_exceeds_transaction(db, seeded_term_student_fee):
    _seed_charge(db, 9_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(6_000_000)])
    assert not r.ok and "INV-01" in r.violations


def test_inv05_overfills_charge(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(4_000_000)])
    assert not r.ok and "INV-05" in r.violations


def test_inv04_unknown_charge(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(1_000_000, charge="CHG-NOPE")])
    assert not r.ok and "INV-04" in r.violations


def test_valid_allocation_passes(db, seeded_term_student_fee):
    _seed_charge(db, 3_000_000)
    txn = _txn(db, 5_000_000)
    r = constraints.validate(db, txn, [_alloc(3_000_000)])
    assert r.ok and r.violations == []
