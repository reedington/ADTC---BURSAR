import json
from dataclasses import dataclass
from bursa.inference.constants import MAX_EXPLANATION_CHARS


@dataclass
class ValidationOutcome:
    ok: bool
    data: dict | None = None
    reason: str | None = None


def validate_output(raw, txn_id, candidate_ids, allowed_codes) -> ValidationOutcome:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ValidationOutcome(False, None, "invalid_json")
    if d.get("transaction_id") != txn_id:
        return ValidationOutcome(False, None, "transaction_id_mismatch")
    if d.get("recommended_action") not in ("auto", "review", "unmatched"):
        return ValidationOutcome(False, None, "bad_action")
    if len(d.get("explanation", "")) > MAX_EXPLANATION_CHARS:
        return ValidationOutcome(False, None, "explanation_too_long")
    allowed_ids, allowed = set(candidate_ids), set(allowed_codes)
    for a in d.get("candidate_allocations", []):
        if a.get("student_id") not in allowed_ids:
            return ValidationOutcome(False, None, "unknown_student_id")
        if not isinstance(a.get("amount_minor"), int) or a["amount_minor"] < 0:
            return ValidationOutcome(False, None, "bad_amount")
        if not set(a.get("reason_codes", [])) <= allowed:
            return ValidationOutcome(False, None, "unknown_reason_code")
    return ValidationOutcome(True, d, None)
