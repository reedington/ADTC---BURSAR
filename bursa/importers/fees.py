from bursa import db as dbmod, ledger, money
from bursa.errors import ImportRowError
from bursa.importers import batches


def import_fees(conn, rows, source_file, column_mapping=None) -> dict:
    original_rows = list(rows)
    rows = batches.mapped_rows(original_rows, column_mapping)
    batch_id = batches.start(conn, source_file, "fees", column_mapping)
    accepted, duplicate, errors = 0, 0, []
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
        try:
            priority = int(row.get("priority", 100))
        except (TypeError, ValueError) as exc:
            errors.append(ImportRowError(i, "priority", str(exc)))
            continue
        charge_id = f"CHG-{sid}-{fee_id}-{term_id}"
        if conn.execute(
            "SELECT 1 FROM charges WHERE charge_id=?", (charge_id,)
        ).fetchone() is not None:
            duplicate += 1
            continue
        with dbmod.transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO fee_items (fee_id, name, term_id, priority) "
                "VALUES (?,?,?,?)",
                (fee_id, row.get("fee_name", fee_id), term_id, priority))
        try:
            ledger.create_charge(conn, charge_id, sid, fee_id, term_id, amount,
                                 "importer", "fees_csv", f"IMPORT:{source_file}")
            accepted += 1
        except Exception as exc:  # e.g. duplicate charge on re-import
            errors.append(ImportRowError(i, "charge", str(exc)))
    return batches.finish(
        conn,
        batch_id,
        {
            "accepted": accepted,
            "duplicate": duplicate,
            "rejected": len(errors),
            "errors": errors,
        },
        original_rows,
    )
