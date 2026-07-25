from bursa import db as dbmod, ledger, money
from bursa.errors import ImportRowError


def import_fees(conn, rows, source_file) -> dict:
    accepted, errors = 0, []
    for i, row in enumerate(rows, start=1):
        fee_id = (row.get("fee_id") or "").strip()
        sid = (row.get("student_id") or "").strip()
        term_id = (row.get("term_id") or "").strip()
        if not (fee_id and sid and term_id):
            errors.append(ImportRowError(i, "fee_id/student_id/term_id", "missing"))
            continue
        try:
            amount = money.parse_naira(row.get("amount", ""))
        except ValueError as exc:
            errors.append(ImportRowError(i, "amount", str(exc)))
            continue
        with dbmod.transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO fee_items (fee_id, name, term_id, priority) "
                "VALUES (?,?,?,?)",
                (fee_id, row.get("fee_name", fee_id), term_id,
                 int(row.get("priority", 100))))
        charge_id = f"CHG-{sid}-{fee_id}-{term_id}"
        try:
            ledger.create_charge(conn, charge_id, sid, fee_id, term_id, amount,
                                 "importer", "fees_csv", f"IMPORT:{source_file}")
            accepted += 1
        except Exception as exc:  # e.g. duplicate charge on re-import
            errors.append(ImportRowError(i, "charge", str(exc)))
    return {"accepted": accepted, "rejected": len(errors), "errors": errors}
