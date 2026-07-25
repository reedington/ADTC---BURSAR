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


def charges_for_student(conn, student_id) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT c.*, f.priority FROM charges c JOIN fee_items f ON c.fee_id = f.fee_id "
        "WHERE c.student_id = ? ORDER BY f.priority ASC, c.charge_id ASC",
        (student_id,)).fetchall()


def insert_proposal(conn, proposal_id, transaction_id, source, action, confidence,
                    explanation, created_at) -> None:
    conn.execute(
        "INSERT INTO proposals (proposal_id, transaction_id, source, recommended_action, "
        "confidence, explanation, created_at) VALUES (?,?,?,?,?,?,?)",
        (proposal_id, transaction_id, source, str(action), confidence, explanation,
         created_at))
