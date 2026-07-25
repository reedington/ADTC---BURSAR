from bursa import projections as proj, repository as repo
from bursa.models import LedgerEventInput, EventType


def _ev(db, **kw):
    base = dict(actor="a", source="s", evidence_ref="e", decision_path="d")
    base.update(kw)
    return repo.insert_ledger_event(db, LedgerEventInput(**base),
                                    created_at="2026-01-01T00:00:00+00:00")


def test_charge_balance_and_status(db, seeded_term_student_fee):
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
    _ev(db, event_type=EventType.CREDIT_GRANT, holder="STU-1", amount_minor=1000000)
    assert proj.holder_credit(db, "STU-1") == 1000000
    _ev(db, event_type=EventType.CREDIT_APPLICATION, holder="STU-1", charge_id="CHG-1",
        amount_minor=400000)
    assert proj.holder_credit(db, "STU-1") == 600000
