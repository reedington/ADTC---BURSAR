import json
import sqlite3
from bursa.models import CanonicalTransaction, LedgerEventInput

_LIVE = ("event_type != 'reversal' AND event_id NOT IN "
         "(SELECT reverses_event_id FROM ledger_events WHERE reverses_event_id IS NOT NULL)")


def insert_transaction(conn, tx: CanonicalTransaction) -> None:
    conn.execute(
        "INSERT INTO transactions (transaction_id, source, reference, raw_reference, "
        "posted_at, payer_name, narration, amount_minor, direction, dedup_hash, "
        "batch_id, routing_state) VALUES (?,?,?,?,?,?,?,?,?,?,?, 'new')",
        (tx.transaction_id, tx.source, tx.reference, tx.raw_reference, tx.posted_at,
         tx.payer_name, tx.narration, tx.amount_minor, tx.direction, tx.dedup_hash,
         tx.batch_id))


def get_transaction(conn, txn_id) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM transactions WHERE transaction_id = ?",
                        (txn_id,)).fetchone()


def find_transaction_by_dedup(conn, dedup_hash) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM transactions WHERE dedup_hash = ?",
                        (dedup_hash,)).fetchone()


def set_routing_state(conn, txn_id, state) -> None:
    conn.execute("UPDATE transactions SET routing_state = ? WHERE transaction_id = ?",
                 (str(state), txn_id))


def insert_ledger_event(conn, ev: LedgerEventInput, created_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO ledger_events (event_type, transaction_id, charge_id, student_id, "
        "fee_id, holder, amount_minor, actor, source, evidence_ref, decision_path, "
        "reverses_event_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (str(ev.event_type), ev.transaction_id, ev.charge_id, ev.student_id, ev.fee_id,
         ev.holder, ev.amount_minor, ev.actor, ev.source, ev.evidence_ref,
         ev.decision_path, ev.reverses_event_id, created_at))
    return cur.lastrowid


def live_events(conn, **filters) -> list[sqlite3.Row]:
    where = [_LIVE]
    params = []
    for col, val in filters.items():
        where.append(f"{col} = ?")
        params.append(val)
    sql = f"SELECT * FROM ledger_events WHERE {' AND '.join(where)}"
    return conn.execute(sql, params).fetchall()


def event_by_id(conn, event_id) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM ledger_events WHERE event_id = ?",
                        (event_id,)).fetchone()


def charge_exists(conn, charge_id) -> bool:
    return conn.execute("SELECT 1 FROM charges WHERE charge_id = ?",
                        (charge_id,)).fetchone() is not None


def student_exists(conn, student_id) -> bool:
    return conn.execute("SELECT 1 FROM students WHERE student_id = ?",
                        (student_id,)).fetchone() is not None


def holder_exists(conn, holder_id) -> bool:
    return student_exists(conn, holder_id) or conn.execute(
        "SELECT 1 FROM guardians WHERE guardian_id = ?", (holder_id,)
    ).fetchone() is not None


def charges_for_student(conn, student_id) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT c.*, f.priority FROM charges c JOIN fee_items f ON c.fee_id = f.fee_id "
        "WHERE c.student_id = ? ORDER BY f.priority ASC, c.charge_id ASC",
        (student_id,)).fetchall()


def insert_proposal(
    conn,
    proposal_id,
    transaction_id,
    source,
    action,
    confidence,
    explanation,
    created_at,
    *,
    features=None,
    candidates=None,
    evidence=None,
    raw_output=None,
    failure_reason=None,
    ambiguities=None,
    allocations=None,
    status="pending",
) -> None:
    conn.execute(
        "INSERT INTO proposals (proposal_id, transaction_id, source, recommended_action, "
        "confidence, explanation, status, created_at, features, candidate_snapshot_json, "
        "evidence_snapshot_json, raw_output_json, failure_reason, ambiguities_json, "
        "last_inference_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            proposal_id,
            transaction_id,
            source,
            str(action),
            confidence,
            explanation,
            status,
            created_at,
            json.dumps(features, sort_keys=True) if isinstance(features, (dict, list)) else features,
            json.dumps(candidates or [], sort_keys=True),
            json.dumps(evidence or {}, sort_keys=True),
            raw_output,
            failure_reason,
            json.dumps(ambiguities or [], sort_keys=True),
            created_at if source == "llm" else None,
        ),
    )
    for allocation in allocations or []:
        conn.execute(
            "INSERT INTO proposal_allocations "
            "(proposal_id, student_id, amount_minor, reason_codes) VALUES (?,?,?,?)",
            (
                proposal_id,
                allocation["student_id"],
                allocation["amount_minor"],
                json.dumps(allocation.get("reason_codes", []), sort_keys=True),
            ),
        )


def get_proposal(conn, proposal_id) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
    ).fetchone()


def proposals_for_transaction(conn, transaction_id) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM proposals WHERE transaction_id = ? ORDER BY created_at DESC, proposal_id",
        (transaction_id,),
    ).fetchall()


def proposal_allocations(conn, proposal_id) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM proposal_allocations WHERE proposal_id = ? ORDER BY id",
        (proposal_id,),
    ).fetchall()


def supersede_pending_proposals(conn, transaction_id) -> None:
    conn.execute(
        "UPDATE proposals SET status='superseded' "
        "WHERE transaction_id=? AND status='pending'",
        (transaction_id,),
    )


def record_proposal_decision(
    conn,
    proposal_id,
    decision,
    actor,
    decided_at,
    allocations,
    unapplied_minor,
    credit_holder=None,
) -> None:
    conn.execute(
        "UPDATE proposals SET status=?, decision=?, decision_actor=?, decision_at=?, "
        "decision_allocations_json=?, unapplied_minor=?, credit_holder=? "
        "WHERE proposal_id=?",
        (
            "approved" if decision == "approve" else "rejected",
            decision,
            actor,
            decided_at,
            json.dumps(allocations or [], sort_keys=True),
            unapplied_minor,
            credit_holder,
            proposal_id,
        ),
    )
