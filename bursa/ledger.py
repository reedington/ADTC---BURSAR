from datetime import datetime, timezone
from bursa import db as dbmod, constraints, repository as repo
from bursa.errors import InvariantViolation
from bursa.models import LedgerEventInput, EventType


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_charge(conn, charge_id, student_id, fee_id, term_id, amount_minor,
                  actor, source, evidence_ref) -> int:
    """Write the charges identity row and its charge_created event atomically."""
    with dbmod.transaction(conn):
        conn.execute("INSERT INTO charges (charge_id, student_id, fee_id, term_id) "
                     "VALUES (?,?,?,?)", (charge_id, student_id, fee_id, term_id))
        ev = LedgerEventInput(event_type=EventType.CHARGE_CREATED, charge_id=charge_id,
            student_id=student_id, fee_id=fee_id, amount_minor=amount_minor, actor=actor,
            source=source, evidence_ref=evidence_ref, decision_path="import")
        return repo.insert_ledger_event(conn, ev, _now())


def post(conn, txn_id, proposed: list[LedgerEventInput], actor, source,
         evidence_ref, decision_path) -> list[int]:
    """Validate the proposed events against cumulative state, then append them.
    All-or-nothing: on violation, rollback and raise InvariantViolation (INV-10)."""
    with dbmod.transaction(conn):
        txn = repo.get_transaction(conn, txn_id)
        result = constraints.validate(conn, txn, proposed)
        if not result.ok:
            raise InvariantViolation(result.violations)
        ids = []
        for ev in proposed:
            stamped = ev.model_copy(update={
                "actor": ev.actor or actor, "source": ev.source or source,
                "evidence_ref": ev.evidence_ref or evidence_ref,
                "decision_path": ev.decision_path or decision_path})
            ids.append(repo.insert_ledger_event(conn, stamped, _now()))
        return ids


def reverse(conn, event_id, actor, reason) -> int:
    """Insert a compensating reversal event. A reversal cannot target a reversal."""
    target = repo.event_by_id(conn, event_id)
    if target is None:
        raise InvariantViolation(["INV-08:unknown-event"])
    if target["event_type"] == EventType.REVERSAL:
        raise InvariantViolation(["INV-08:no-reversal-of-reversal"])
    with dbmod.transaction(conn):
        ev = LedgerEventInput(event_type=EventType.REVERSAL,
            transaction_id=target["transaction_id"], charge_id=target["charge_id"],
            student_id=target["student_id"], fee_id=target["fee_id"],
            holder=target["holder"], amount_minor=target["amount_minor"], actor=actor,
            source="correction", evidence_ref=str(event_id), decision_path=reason,
            reverses_event_id=event_id)
        return repo.insert_ledger_event(conn, ev, _now())


def grant_credit(conn, txn_id, holder, amount_minor, actor, evidence_ref) -> list[int]:
    ev = LedgerEventInput(event_type=EventType.CREDIT_GRANT, transaction_id=txn_id,
        holder=holder, amount_minor=amount_minor, actor=actor, source="overpayment",
        evidence_ref=evidence_ref, decision_path="credit_grant")
    return post(conn, txn_id, [ev], actor, "overpayment", evidence_ref, "credit_grant")


def apply_credit(conn, holder, charge_id, student_id, fee_id, amount_minor, actor) -> list[int]:
    """Apply existing credit to a charge. Validated for charge overfill + credit
    sufficiency; not funded by a transaction, so INV-01 does not apply."""
    ev = LedgerEventInput(event_type=EventType.CREDIT_APPLICATION, holder=holder,
        charge_id=charge_id, student_id=student_id, fee_id=fee_id, amount_minor=amount_minor,
        actor=actor, source="credit", evidence_ref=holder, decision_path="credit_application")
    with dbmod.transaction(conn):
        result = constraints.validate(conn, {"transaction_id": None,
                                             "amount_minor": 0}, [ev])
        if not result.ok:
            raise InvariantViolation(result.violations)
        return [repo.insert_ledger_event(conn, ev, _now())]
