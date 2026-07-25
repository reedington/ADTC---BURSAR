import hashlib
import sqlite3
from bursa import db as dbmod, ledger, normalize, repository as repo
from bursa.models import CanonicalTransaction, LedgerEventInput, EventType
from bursa_eval.models import GoldCase, naira_to_minor


def charge_id_for(student_id, fee_id):
    return f"CHG-{student_id}-{fee_id}"


def _dedup_and_id(tx, source_file, occ):
    ref = normalize.canonicalize_reference(tx.reference)
    if ref:
        basis, txn_id = f"REF:{ref}", f"TXN-{ref}"
    else:
        basis = "|".join(["ROW", source_file, str(occ), tx.date,
                          str(naira_to_minor(tx.amount_naira)),
                          normalize.normalize_name(tx.payer_name or ""),
                          " ".join(normalize.narration_tokens(tx.narration))])
        txn_id = None
    dedup = hashlib.sha256(basis.encode()).hexdigest()[:24]
    return (txn_id or f"TXN-{dedup}"), dedup, ref


def _insert_transaction(conn, tx, source, source_file, occ=0):
    txn_id, dedup, ref = _dedup_and_id(tx, source_file, occ)
    repo.insert_transaction(conn, CanonicalTransaction(
        transaction_id=txn_id, source=source, reference=ref, raw_reference=tx.reference,
        posted_at=tx.date, payer_name=tx.payer_name, narration=tx.narration,
        amount_minor=naira_to_minor(tx.amount_naira), direction="credit", dedup_hash=dedup))
    return txn_id


def _ensure_fee(conn, fee_id, term_id):
    conn.execute("INSERT OR IGNORE INTO fee_items (fee_id, name, term_id, priority) "
                 "VALUES (?,?,?,100)", (fee_id, fee_id, term_id))


def _resolve_charge(conn, student_id, fee_id):
    if fee_id is not None:
        return charge_id_for(student_id, fee_id), fee_id
    rows = repo.charges_for_student(conn, student_id)
    if len(rows) == 1:
        return rows[0]["charge_id"], rows[0]["fee_id"]
    raise ValueError(f"allocation for {student_id} needs an explicit fee_id "
                     f"({len(rows)} charges present)")


def materialize(case: GoldCase) -> sqlite3.Connection:
    """Build the case's world in an in-memory DB: term/students/guardians/aliases, charges
    (via the real ledger's charge_created events), and setup.history replayed through ledger.post."""
    conn = dbmod.connect(":memory:")
    dbmod.init_db(conn)
    s = case.setup
    with dbmod.transaction(conn):
        conn.execute("INSERT INTO terms VALUES (?,?,?,1)", (s.term.id, s.term.session, s.term.name))
        for g in s.guardians:
            conn.execute("INSERT INTO guardians VALUES (?,?,?,?)",
                         (g.id, g.name, normalize.normalize_name(g.name), g.phone_suffix))
        for st in s.students:
            conn.execute("INSERT INTO students VALUES (?,?,?,?,?)",
                         (st.id, st.name, normalize.normalize_name(st.name),
                          st.student_class, s.term.id))
            for a in st.aliases:
                conn.execute("INSERT OR IGNORE INTO student_aliases VALUES (?,?,?)",
                             (st.id, a, normalize.normalize_name(a)))
            for gid in st.guardians:
                conn.execute("INSERT OR IGNORE INTO student_guardians VALUES (?,?)", (st.id, gid))
    for st in s.students:
        for ch in st.charges:
            _ensure_fee(conn, ch.fee_id, s.term.id)
            ledger.create_charge(conn, charge_id_for(st.id, ch.fee_id), st.id, ch.fee_id,
                                 s.term.id, naira_to_minor(ch.amount_naira),
                                 "goldset", "gold_yaml", case.id)
    for i, h in enumerate(s.history):
        txn_id = _insert_transaction(conn, h.transaction, "gold_history", f"hist:{case.id}", i)
        events = []
        for a in h.allocations:
            cid, fid = _resolve_charge(conn, a.student_id, a.fee_id)
            events.append(LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=txn_id,
                charge_id=cid, student_id=a.student_id, fee_id=fid,
                amount_minor=naira_to_minor(a.amount_naira), actor="goldset", source="history",
                evidence_ref=txn_id, decision_path="replay"))
        if events:
            ledger.post(conn, txn_id, events, "goldset", "history", txn_id, "replay")
    return conn


def insert_case_transaction(conn, case: GoldCase) -> str:
    return _insert_transaction(conn, case.transaction, "gold_case", f"case:{case.id}")


def build_expected_events(conn, case: GoldCase, txn_id: str) -> list[LedgerEventInput]:
    events = []
    for a in case.expected.allocations:
        cid, fid = _resolve_charge(conn, a.student_id, a.fee_id)
        events.append(LedgerEventInput(event_type=EventType.ALLOCATION, transaction_id=txn_id,
            charge_id=cid, student_id=a.student_id, fee_id=fid,
            amount_minor=naira_to_minor(a.amount_naira), actor="goldset", source="expected",
            evidence_ref=txn_id, decision_path="gold"))
    for c in case.expected.credits:
        holder_is_student = repo.student_exists(conn, c.holder)
        events.append(LedgerEventInput(event_type=EventType.CREDIT_GRANT, transaction_id=txn_id,
            holder=c.holder, student_id=c.holder if holder_is_student else None,
            amount_minor=naira_to_minor(c.amount_naira), actor="goldset", source="expected",
            evidence_ref=txn_id, decision_path="gold"))
    return events
