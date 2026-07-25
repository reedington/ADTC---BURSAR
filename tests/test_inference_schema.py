import json
from bursa.inference.schema import validate_output


def _valid(txn="TXN-1", sid="STU-1", code="NAME_ALIAS_MATCH"):
    return json.dumps({"transaction_id": txn, "interpretation": {},
        "candidate_allocations": [{"student_id": sid, "amount_minor": 5000000,
                                   "reason_codes": [code]}],
        "recommended_action": "review", "explanation": "ok", "ambiguities": []})


def test_valid_passes():
    out = validate_output(_valid(), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert out.ok and out.data["recommended_action"] == "review"


def test_unknown_id_rejected():
    out = validate_output(_valid(sid="STU-9999"), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert not out.ok and out.reason == "unknown_student_id"


def test_bad_json_rejected():
    out = validate_output("{not json", "TXN-1", ["STU-1"], ["X"])
    assert not out.ok


def test_bad_action_rejected():
    payload = json.loads(_valid())
    payload["recommended_action"] = "delete_everything"
    out = validate_output(json.dumps(payload), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert not out.ok and out.reason == "bad_action"


def test_over_long_explanation_rejected():
    payload = json.loads(_valid())
    payload["explanation"] = "x" * 5000
    out = validate_output(json.dumps(payload), "TXN-1", ["STU-1"], ["NAME_ALIAS_MATCH"])
    assert not out.ok and out.reason == "explanation_too_long"
