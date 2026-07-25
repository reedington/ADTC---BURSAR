from bursa import repository as repo
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType


def _tx(**kw):
    base = dict(transaction_id="TXN-1", source="bank_csv", posted_at="2026-02-14T00:00:00+00:00",
               amount_minor=7500000, direction="credit", dedup_hash="h1")
    base.update(kw)
    return CanonicalTransaction(**base)


def test_transaction_round_trip(db):
    repo.insert_transaction(db, _tx())
    row = repo.get_transaction(db, "TXN-1")
    assert row["amount_minor"] == 7500000
    assert repo.find_transaction_by_dedup(db, "h1")["transaction_id"] == "TXN-1"


def test_live_events_excludes_reversed(db, seeded_term_student_fee):
    db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    e1 = repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.CHARGE_CREATED, charge_id="CHG-1", student_id="STU-1",
        fee_id="FEE-TUITION", amount_minor=5000000, actor="importer",
        source="fees_csv", evidence_ref="BATCH-1", decision_path="import"),
        created_at="2026-01-01T00:00:00+00:00")
    live = repo.live_events(db, charge_id="CHG-1")
    assert len(live) == 1
    repo.insert_ledger_event(db, LedgerEventInput(
        event_type=EventType.REVERSAL, charge_id="CHG-1", amount_minor=5000000,
        actor="bursar", source="correction", evidence_ref="BATCH-1",
        decision_path="reverse", reverses_event_id=e1),
        created_at="2026-01-02T00:00:00+00:00")
    assert repo.live_events(db, charge_id="CHG-1") == []
