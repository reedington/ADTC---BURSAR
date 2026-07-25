from bursa import distribute, ledger
from bursa.models import EventType


def _two_charges(db, seeded_term_student_fee):
    # CHG-1b tuition (priority 10) billed 3,000,000; CHG-3 books (priority 50) billed 1,000,000
    db.execute("INSERT INTO fee_items VALUES ('FEE-BOOKS','Books','T1',50)")
    ledger.create_charge(db, "CHG-1b", "STU-1", "FEE-TUITION", "T1", 3_000_000, "i", "fees", "B")
    ledger.create_charge(db, "CHG-3", "STU-1", "FEE-BOOKS", "T1", 1_000_000, "i", "fees", "B")


def test_fills_by_priority_then_surplus_to_credit(db, seeded_term_student_fee):
    _two_charges(db, seeded_term_student_fee)
    events, remainder = distribute.distribute(db, "TXN-1", "STU-1", 4_500_000, "engine")
    kinds = [(e.event_type, e.charge_id, e.amount_minor) for e in events]
    assert (EventType.ALLOCATION, "CHG-1b", 3_000_000) in kinds
    assert (EventType.ALLOCATION, "CHG-3", 1_000_000) in kinds
    assert any(e.event_type == EventType.CREDIT_GRANT and e.amount_minor == 500_000
               for e in events)
    assert remainder == 0


def test_partial_underpayment_leaves_no_remainder_but_partial_fill(db, seeded_term_student_fee):
    _two_charges(db, seeded_term_student_fee)
    events, remainder = distribute.distribute(db, "TXN-1", "STU-1", 2_000_000, "engine")
    assert [(e.charge_id, e.amount_minor) for e in events] == [("CHG-1b", 2_000_000)]
    assert remainder == 0
