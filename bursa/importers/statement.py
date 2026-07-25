import hashlib
from bursa import db as dbmod, money, normalize, repository as repo
from bursa.errors import ImportRowError
from bursa.models import CanonicalTransaction


def compute_dedup_hash(row, source_file, occurrence) -> str:
    """Collides only on genuine re-imports: by reference when present, else by
    (source_file, canonical row, occurrence-within-file)."""
    ref = normalize.canonicalize_reference(row.get("reference"))
    if ref:
        basis = f"REF:{ref}"
    else:
        basis = "|".join(["ROW", source_file, str(occurrence),
                          row.get("date", ""), row.get("amount", ""),
                          normalize.normalize_name(row.get("payer", "")),
                          " ".join(normalize.narration_tokens(row.get("narration")))])
    return hashlib.sha256(basis.encode()).hexdigest()[:24]


def _content_key(row) -> str:
    return "|".join([row.get("date", ""), row.get("amount", ""),
                     normalize.normalize_name(row.get("payer", "")),
                     " ".join(normalize.narration_tokens(row.get("narration")))])


def import_statement(conn, rows, source_file) -> dict:
    accepted = duplicate = 0
    errors, near_dups = [], []
    seen_occurrence: dict[str, int] = {}

    # Pre-index existing content keys for near-duplicate (no-reference) detection.
    existing = conn.execute(
        "SELECT posted_at, amount_minor, payer_name, narration FROM transactions").fetchall()
    existing_content = set()
    for e in existing:
        existing_content.add("|".join([
            e["posted_at"][:10] if e["posted_at"] else "",
            str(e["amount_minor"]), normalize.normalize_name(e["payer_name"] or ""),
            " ".join(normalize.narration_tokens(e["narration"]))]))

    for i, row in enumerate(rows, start=1):
        if (row.get("direction") or "credit").lower() != "credit":
            continue  # debit rows excluded (FR-002)
        try:
            amount = money.parse_naira(row.get("amount", ""))
        except ValueError as exc:
            errors.append(ImportRowError(i, "amount", str(exc)))
            continue

        # Occurrence index makes reference-less re-imports of the same file idempotent.
        base = f"{source_file}:{_content_key(row)}"
        occ = seen_occurrence.get(base, 0)
        seen_occurrence[base] = occ + 1

        dedup = compute_dedup_hash(row, source_file, occ)
        if repo.find_transaction_by_dedup(conn, dedup) is not None:
            duplicate += 1
            continue

        ref = normalize.canonicalize_reference(row.get("reference"))
        content = "|".join([row.get("date", ""), str(amount),
                            normalize.normalize_name(row.get("payer", "")),
                            " ".join(normalize.narration_tokens(row.get("narration")))])
        is_near_dup = ref is None and content in existing_content
        txn_id = f"TXN-{ref}" if ref else f"TXN-{dedup}"
        tx = CanonicalTransaction(transaction_id=txn_id, source="bank_csv", reference=ref,
            raw_reference=row.get("reference") or None, posted_at=row.get("date", ""),
            payer_name=row.get("payer"), narration=row.get("narration"),
            amount_minor=amount, direction="credit", dedup_hash=dedup)
        with dbmod.transaction(conn):
            repo.insert_transaction(conn, tx)
            if is_near_dup:
                repo.set_routing_state(conn, txn_id, "review")
                near_dups.append(txn_id)
        existing_content.add(content)
        accepted += 1

    return {"accepted": accepted, "duplicate": duplicate, "rejected": len(errors),
            "errors": errors, "near_duplicates": near_dups}
