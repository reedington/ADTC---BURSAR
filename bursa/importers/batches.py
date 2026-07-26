from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4


def start(conn, source_file: str, kind: str, mapping: dict | None = None) -> str:
    batch_id = f"BATCH-{uuid4().hex}"
    conn.execute(
        "INSERT INTO import_batches "
        "(batch_id, source_file, imported_at, kind, status, mapping_json) "
        "VALUES (?,?,?,?,?,?)",
        (
            batch_id,
            source_file,
            datetime.now(timezone.utc).isoformat(),
            kind,
            "processing",
            json.dumps(mapping or {}, sort_keys=True),
        ),
    )
    return batch_id


def mapped_rows(rows, mapping: dict | None):
    """Map canonical field names to source-column names while preserving unmapped fields."""
    if not mapping:
        return [dict(row) for row in rows]
    return [
        {
            **dict(row),
            **{
                canonical: row.get(source, "")
                for canonical, source in mapping.items()
                if source
            },
        }
        for row in rows
    ]


def finish(conn, batch_id, result: dict, rows) -> dict:
    conn.execute(
        "UPDATE import_batches SET accepted=?, rejected=?, duplicate=?, status='complete' "
        "WHERE batch_id=?",
        (
            result.get("accepted", 0),
            result.get("rejected", 0),
            result.get("duplicate", 0),
            batch_id,
        ),
    )
    for error in result.get("errors", []):
        raw = rows[error.row_number - 1] if 0 < error.row_number <= len(rows) else {}
        conn.execute(
            "INSERT INTO import_errors "
            "(batch_id, row_number, field, reason, raw_row_json) VALUES (?,?,?,?,?)",
            (
                batch_id,
                error.row_number,
                error.field,
                error.reason,
                json.dumps(raw, sort_keys=True),
            ),
        )
    result["batch_id"] = batch_id
    return result
