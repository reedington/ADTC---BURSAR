"""Human decision boundary between proposals and the append-only ledger."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from bursa import db as dbmod
from bursa import distribute, ledger, repository as repo
from bursa.errors import InvariantViolation
from bursa.models import EventType, LedgerEventInput


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _allowed_students(conn, proposal) -> set[str]:
    candidates = _json(proposal["candidate_snapshot_json"], [])
    allowed = {
        candidate.get("student_id")
        for candidate in candidates
        if candidate.get("student_id")
    }
    allowed.update(
        row["student_id"] for row in repo.proposal_allocations(conn, proposal["proposal_id"])
    )
    return allowed


def approve(
    conn,
    proposal_id: str,
    allocations: list[dict],
    actor: str,
    *,
    credit_holder: str | None = None,
) -> list[int]:
    """Approve an edited/split proposal after rebuilding all ledger events server-side."""
    proposal = repo.get_proposal(conn, proposal_id)
    if proposal is None:
        raise InvariantViolation(["UNKNOWN_PROPOSAL"])
    if proposal["status"] != "pending":
        raise InvariantViolation(["PROPOSAL_ALREADY_DECIDED"])
    txn = repo.get_transaction(conn, proposal["transaction_id"])
    if txn is None:
        raise InvariantViolation(["UNKNOWN_TRANSACTION"])

    combined: dict[str, int] = defaultdict(int)
    for allocation in allocations:
        student_id = str(allocation.get("student_id") or "").strip()
        amount_minor = allocation.get("amount_minor")
        if not student_id or not isinstance(amount_minor, int) or amount_minor < 0:
            raise InvariantViolation(["INV-03"])
        if amount_minor:
            combined[student_id] += amount_minor

    allowed = _allowed_students(conn, proposal)
    if not combined.keys() <= allowed:
        raise InvariantViolation(["INV-04"])
    if sum(combined.values()) > txn["amount_minor"]:
        raise InvariantViolation(["INV-01"])
    if credit_holder and not repo.holder_exists(conn, credit_holder):
        raise InvariantViolation(["INV-04"])

    decision_allocations = [
        {"student_id": student_id, "amount_minor": amount}
        for student_id, amount in sorted(combined.items())
    ]
    proposed = []
    for student_id, amount in combined.items():
        events, _ = distribute.distribute(
            conn,
            txn["transaction_id"],
            student_id,
            amount,
            actor,
            create_credit=False,
        )
        proposed.extend(events)

    allocated_minor = sum(
        event.amount_minor for event in proposed if event.event_type == EventType.ALLOCATION
    )
    unapplied_minor = txn["amount_minor"] - allocated_minor
    if credit_holder and unapplied_minor:
        proposed.append(
            LedgerEventInput(
                event_type=EventType.CREDIT_GRANT,
                transaction_id=txn["transaction_id"],
                student_id=credit_holder if repo.student_exists(conn, credit_holder) else None,
                holder=credit_holder,
                amount_minor=unapplied_minor,
                actor=actor,
                source="human_review",
                evidence_ref=proposal_id,
                decision_path="explicit_credit",
            )
        )
        unapplied_minor = 0

    with dbmod.transaction(conn):
        event_ids = ledger.post_within_transaction(
            conn,
            txn["transaction_id"],
            proposed,
            actor,
            "human_review",
            proposal_id,
            "approve",
        )
        repo.record_proposal_decision(
            conn,
            proposal_id,
            "approve",
            actor,
            _now(),
            decision_allocations,
            unapplied_minor,
            credit_holder,
        )
        repo.set_routing_state(conn, txn["transaction_id"], "approved")
    return event_ids


def reject(conn, proposal_id: str, actor: str) -> None:
    proposal = repo.get_proposal(conn, proposal_id)
    if proposal is None:
        raise InvariantViolation(["UNKNOWN_PROPOSAL"])
    if proposal["status"] != "pending":
        raise InvariantViolation(["PROPOSAL_ALREADY_DECIDED"])
    txn = repo.get_transaction(conn, proposal["transaction_id"])
    with dbmod.transaction(conn):
        repo.record_proposal_decision(
            conn,
            proposal_id,
            "reject",
            actor,
            _now(),
            [],
            txn["amount_minor"],
        )
        repo.set_routing_state(conn, txn["transaction_id"], "unmatched")
