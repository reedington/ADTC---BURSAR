from bursa import repository as repo


def _sum(rows, predicate) -> int:
    return sum(r["amount_minor"] for r in rows if predicate(r))


def charge_billed(conn, charge_id) -> int:
    rows = repo.live_events(conn, charge_id=charge_id)
    return _sum(rows, lambda r: r["event_type"] == "charge_created")


def charge_paid(conn, charge_id) -> int:
    rows = repo.live_events(conn, charge_id=charge_id)
    return _sum(rows, lambda r: r["event_type"] in ("allocation", "credit_application"))


def charge_balance(conn, charge_id) -> int:
    return charge_billed(conn, charge_id) - charge_paid(conn, charge_id)


def txn_used(conn, txn_id) -> int:
    rows = repo.live_events(conn, transaction_id=txn_id)
    return _sum(rows, lambda r: r["event_type"] in ("allocation", "credit_grant"))


def txn_unapplied(conn, txn) -> int:
    return txn["amount_minor"] - txn_used(conn, txn["transaction_id"])


def holder_credit(conn, holder) -> int:
    rows = repo.live_events(conn, holder=holder)
    grants = _sum(rows, lambda r: r["event_type"] == "credit_grant")
    apps = _sum(rows, lambda r: r["event_type"] == "credit_application")
    return grants - apps


def student_status(conn, student_id) -> str:
    charges = repo.charges_for_student(conn, student_id)
    billed = sum(charge_billed(conn, c["charge_id"]) for c in charges)
    paid = sum(charge_paid(conn, c["charge_id"]) for c in charges)
    if holder_credit(conn, student_id) > 0 and paid >= billed:
        return "credit"
    if paid <= 0:
        return "outstanding"
    if paid < billed:
        return "part_paid"
    return "cleared"
