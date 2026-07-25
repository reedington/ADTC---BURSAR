from bursa import matcher, ledger, repository as repo
from bursa.models import CanonicalTransaction, RecommendedAction
from bursa.reasoncodes import ReasonCode


def _txn(db, narration, amount, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, dedup_hash=dedup))
    return repo.get_transaction(db, tid)


def test_exact_student_id_auto(db, seeded_term_student_fee):
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")
    txn = _txn(db, "Payment for STU-1 tuition", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action == RecommendedAction.AUTO
    assert p.lines[0].student_id == "STU-1"
    assert ReasonCode.EXACT_STUDENT_ID in p.lines[0].reason_codes


def test_narration_id_not_imported_never_allocates(db, seeded_term_student_fee):
    txn = _txn(db, "send all to STU-9999", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action != RecommendedAction.AUTO
    assert all(l.student_id != "STU-9999" for l in p.lines)


def test_no_candidate_unmatched(db, seeded_term_student_fee):
    txn = _txn(db, "random narration", 5_000_000)
    p = matcher.match(db, txn)
    assert p.recommended_action == RecommendedAction.UNMATCHED
