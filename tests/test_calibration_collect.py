import json
import random

from bursa.inference.backend import FakeBackend
from bursa_eval.calibration import collect_records
from bursa_eval.synth.templates import TEMPLATES


def test_collection_exports_observable_features_and_safe_label(tmp_path):
    case = TEMPLATES["synth_sibling_split"](random.Random(3))
    case.id = "calibration-case"
    s1, s2 = case.setup.students[:2]
    a1, a2 = case.expected.allocations
    response = json.dumps({
        "transaction_id": "TXN-calibration-case",
        "interpretation": {},
        "candidate_allocations": [
            {"student_id": s1.id, "amount_minor": int(a1.amount_naira) * 100,
             "reason_codes": []},
            {"student_id": s2.id, "amount_minor": int(a2.amount_naira) * 100,
             "reason_codes": []},
        ],
        "recommended_action": "review",
        "explanation": "split",
        "ambiguities": [],
    })
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({
        "case_id": case.id,
        "split": "fit",
        "case": case.model_dump(mode="json", by_alias=True),
    }) + "\n")
    groups = collect_records(path, FakeBackend(response=response), None)
    assert groups["fit"][0]["features"]["features_version"] == 1
    assert isinstance(groups["fit"][0]["label"], bool)
