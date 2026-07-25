from bursa import projections as proj, repository as repo
from bursa.models import LedgerEventInput, EventType


def _ev(db, **kw):
    base = dict(actor="a", source="s", evidence_ref="e", decision_path="d")
    base.update(kw)
    return repo.insert_ledger_event(db, LedgerEventInput(**base),
                                    created_at="2026-01-01T00:00:00+00:00")


def test_charge_balance_and_status(db, seeded_term_student_fee):
    db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    _ev(db, event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=5000000)
    assert proj.charge_billed(db, "CHG-1") == 5000000
    assert proj.charge_balance(db, "CHG-1") == 5000000
    assert proj.student_status(db, "STU-1") == "outstanding"
    _ev(db, event_type=EventType.ALLOCATION, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", transaction_id=None, amount_minor=2000000)
    assert proj.charge_balance(db, "CHG-1") == 3000000
    assert proj.student_status(db, "STU-1") == "part_paid"


def test_holder_credit(db, seeded_term_student_fee):
    db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    _ev(db, event_type=EventType.CREDIT_GRANT, holder="STU-1", amount_minor=1000000)
    assert proj.holder_credit(db, "STU-1") == 1000000
    _ev(db, event_type=EventType.CREDIT_APPLICATION, holder="STU-1", charge_id="CHG-1",
        amount_minor=400000)
    assert proj.holder_credit(db, "STU-1") == 600000


def test_payer_history_live_only(db, seeded_term_student_fee):
    from bursa import ledger, normalize
    from bursa.models import CanonicalTransaction, LedgerEventInput, EventType
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    repo.insert_transaction(db, CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=1_000_000, direction="credit",
        payer_name="C N Okafor", dedup_hash="h1"))
    ev = LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id="TXN-1",
        charge_id="CHG-1", student_id="STU-1", fee_id="FEE-TUITION", amount_minor=1_000_000,
        actor="e", source="d", evidence_ref="TXN-1", decision_path="auto")
    [aid] = ledger.post(db, "TXN-1", [ev], "e", "d", "TXN-1", "auto")
    assert "STU-1" in proj.payer_history(db, normalize.normalize_name("C N Okafor"))
    ledger.reverse(db, aid, "bursar", "wrong")
    assert proj.payer_history(db, normalize.normalize_name("C N Okafor")) == set()

