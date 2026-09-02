import json

from bursa_eval.general import load_oasst_pairs, project_owned_examples


def test_project_owned_examples_are_balanced_and_deterministic():
    rows = project_owned_examples(12, seed=4)
    assert rows == project_owned_examples(12, seed=4)
    assert {row["category"] for row in rows} == {
        "summarization", "drafting", "analysis", "decision_support"
    }


def test_oasst_filter_pins_ids_and_excludes_non_english(tmp_path):
    rows = [
        {"message_id": "p1", "role": "prompter", "lang": "en",
         "text": "Please explain a sensible fictional operational control."},
        {"message_id": "a1", "parent_id": "p1", "role": "assistant", "lang": "en",
         "rank": 0, "text": "Use documented approvals and reconcile the evidence regularly."},
        {"message_id": "p2", "role": "prompter", "lang": "de", "text": "Eine lange Frage."},
        {"message_id": "a2", "parent_id": "p2", "role": "assistant", "lang": "de",
         "rank": 0, "text": "Eine ausreichend lange Antwort für den Filter."},
    ]
    path = tmp_path / "oasst.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    selected = load_oasst_pairs(path, 1, seed=4, revision="abc")
    assert selected[0]["id"] == "oasst1-a1"
    assert selected[0]["source_revision"] == "abc"
