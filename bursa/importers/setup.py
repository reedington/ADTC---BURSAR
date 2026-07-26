from __future__ import annotations

from bursa import db as dbmod, normalize
from bursa.errors import ImportRowError
from bursa.importers import batches


def _finish(conn, batch_id, accepted, duplicate, errors, original_rows):
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


def import_terms(conn, rows, source_file, column_mapping=None) -> dict:
    original_rows = list(rows)
    rows = batches.mapped_rows(original_rows, column_mapping)
    batch_id = batches.start(conn, source_file, "terms", column_mapping)
    accepted = duplicate = 0
    errors = []
    for i, row in enumerate(rows, start=1):
        term_id = (row.get("term_id") or "").strip()
        session = (row.get("session") or "").strip()
        term_name = (row.get("term_name") or "").strip()
        if not term_id or not session or not term_name:
            errors.append(ImportRowError(i, "term_id/session/term_name", "missing"))
            continue
        if conn.execute("SELECT 1 FROM terms WHERE term_id=?", (term_id,)).fetchone():
            duplicate += 1
            continue
        active = str(row.get("is_active", "0")).strip().lower() in {"1", "true", "yes"}
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT INTO terms(term_id, session, term_name, is_active) VALUES (?,?,?,?)",
                    (term_id, session, term_name, int(active)),
                )
            accepted += 1
        except Exception as exc:
            errors.append(ImportRowError(i, "term_id", str(exc)))
    return _finish(conn, batch_id, accepted, duplicate, errors, original_rows)


def import_guardians(conn, rows, source_file, column_mapping=None) -> dict:
    original_rows = list(rows)
    rows = batches.mapped_rows(original_rows, column_mapping)
    batch_id = batches.start(conn, source_file, "guardians", column_mapping)
    accepted = duplicate = 0
    errors = []
    for i, row in enumerate(rows, start=1):
        guardian_id = (row.get("guardian_id") or "").strip()
        name = (row.get("name") or "").strip()
        if not guardian_id or not name:
            errors.append(ImportRowError(i, "guardian_id/name", "missing"))
            continue
        if conn.execute(
            "SELECT 1 FROM guardians WHERE guardian_id=?", (guardian_id,)
        ).fetchone():
            duplicate += 1
            continue
        phone_suffix = (row.get("phone_suffix") or "").strip() or None
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT INTO guardians "
                    "(guardian_id, name, normalized_name, phone_suffix) VALUES (?,?,?,?)",
                    (guardian_id, name, normalize.normalize_name(name), phone_suffix),
                )
            accepted += 1
        except Exception as exc:
            errors.append(ImportRowError(i, "guardian_id", str(exc)))
    return _finish(conn, batch_id, accepted, duplicate, errors, original_rows)


def import_relationships(conn, rows, source_file, column_mapping=None) -> dict:
    original_rows = list(rows)
    rows = batches.mapped_rows(original_rows, column_mapping)
    batch_id = batches.start(conn, source_file, "relationships", column_mapping)
    accepted = duplicate = 0
    errors = []
    for i, row in enumerate(rows, start=1):
        student_id = (row.get("student_id") or "").strip()
        guardian_id = (row.get("guardian_id") or "").strip()
        if not student_id or not guardian_id:
            errors.append(ImportRowError(i, "student_id/guardian_id", "missing"))
            continue
        if conn.execute(
            "SELECT 1 FROM student_guardians WHERE student_id=? AND guardian_id=?",
            (student_id, guardian_id),
        ).fetchone():
            duplicate += 1
            continue
        try:
            with dbmod.transaction(conn):
                conn.execute(
                    "INSERT INTO student_guardians(student_id, guardian_id) VALUES (?,?)",
                    (student_id, guardian_id),
                )
            accepted += 1
        except Exception as exc:
            errors.append(ImportRowError(i, "student_id/guardian_id", str(exc)))
    return _finish(conn, batch_id, accepted, duplicate, errors, original_rows)
