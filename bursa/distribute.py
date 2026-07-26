from bursa import repository as repo, projections as proj
from bursa.models import LedgerEventInput, EventType


def distribute(conn, txn_id, student_id, amount_minor, actor, *, create_credit=True):
    """Map a student-level proposal into charge-level allocation events (priority order).
    Surplus becomes credit only when the caller explicitly permits it; otherwise it remains
    visible as unapplied money for a bursar decision."""
    remaining = amount_minor
    events: list[LedgerEventInput] = []
    for c in repo.charges_for_student(conn, student_id):  # ordered priority ASC, charge_id ASC
        if remaining <= 0:
            break
        bal = proj.charge_balance(conn, c["charge_id"])
        if bal <= 0:
            continue
        take = min(bal, remaining)
        events.append(LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=txn_id,
            charge_id=c["charge_id"], student_id=student_id, fee_id=c["fee_id"],
            amount_minor=take, actor=actor, source="deterministic",
            evidence_ref=txn_id, decision_path="distribute"))
        remaining -= take
    if remaining > 0 and create_credit:
        # all of this student's charges are cleared -> surplus becomes credit
        events.append(LedgerEventInput(event_type=EventType.CREDIT_GRANT, transaction_id=txn_id,
            holder=student_id, student_id=student_id, amount_minor=remaining, actor=actor,
            source="overpayment", evidence_ref=txn_id, decision_path="distribute"))
        remaining = 0
    return events, remaining
