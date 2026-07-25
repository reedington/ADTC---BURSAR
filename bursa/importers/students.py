from bursa import db as dbmod, normalize
from bursa.errors import ImportRowError


def import_students(conn, rows, source_file) -> dict:
    accepted, errors = 0, []
    for i, row in enumerate(rows, start=1):
        sid = (row.get("student_id") or "").strip()
        name = (row.get("name") or "").strip()
        if not sid:
            errors.append(ImportRowError(i, "student_id", "missing"))
            continue
        if not name:
            errors.append(ImportRowError(i, "name", "missing"))
            continue
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT OR IGNORE INTO students "
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
    return {"accepted": accepted, "rejected": len(errors), "errors": errors}
