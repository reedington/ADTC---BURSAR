import json

from bursa_eval.proxy import SUBJECTS, build_proxy


def test_proxy_builder_selects_exact_balanced_rows(tmp_path):
    rows = []
    for subject in SUBJECTS:
        for index in range(3):
            rows.append({
                "subject": subject,
                "question": f"{subject} question {index}",
                "choices": ["one", "two", "three", "four"],
                "answer": index % 4,
            })
    source = tmp_path / "mmlu.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "proxy.jsonl"
    manifest = build_proxy(
        source, output, revision="pinned", seed=1, per_subject=2
    )
    assert len(output.read_text().splitlines()) == 8
    assert manifest["official"] is False
    assert manifest["label"] == "internal_mmlu_enterprise_proxy"
