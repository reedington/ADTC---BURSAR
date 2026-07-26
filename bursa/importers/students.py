from bursa import db as dbmod, normalize
from bursa.errors import ImportRowError
from bursa.importers import batches


def import_students(conn, rows, source_file, column_mapping=None) -> dict:
    original_rows = list(rows)
    rows = batches.mapped_rows(original_rows, column_mapping)
    batch_id = batches.start(conn, source_file, "students", column_mapping)
    accepted, duplicate, errors = 0, 0, []
    for i, row in enumerate(rows, start=1):
        sid = (row.get("student_id") or "").strip()
        name = (row.get("name") or "").strip()
        if not sid:
            errors.append(ImportRowError(i, "student_id", "missing"))
            continue
        if not name:
            errors.append(ImportRowError(i, "name", "missing"))
            continue
        if conn.execute(
            "SELECT 1 FROM students WHERE student_id=?", (sid,)
        ).fetchone() is not None:
            duplicate += 1
            continue
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT INTO students "
                    "(student_id, name, normalized_name, class, term_id) VALUES (?,?,?,?,?)",
                    (sid, name, normalize.normalize_name(name), row.get("class"),
                     row.get("term_id")))
                raw_aliases = (row.get("aliases") or "").strip()
                for alias in [a.strip() for a in raw_aliases.split(";") if a.strip()]:
                    conn.execute(
                        "INSERT OR IGNORE INTO student_aliases "
                        "(student_id, alias, normalized_alias) VALUES (?,?,?)",
                        (sid, alias, normalize.normalize_name(alias)))
            accepted += 1
        except Exception as exc:  # FK / integrity
            errors.append(ImportRowError(i, "student_id", str(exc)))
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
