import json
from pathlib import Path

from bursa_eval.submission_check import check_metadata, run_checks


def _write_valid(root: Path):
    (root / "data/bare").mkdir(parents=True)
    prompts = ["Prompt one", "Prompt two"]
    (root / "data/bare/prompts.jsonl").write_text(
        "\n".join(json.dumps({"id": str(i), "prompt": p, "kind": "visible"})
                  for i, p in enumerate(prompts, 1)) + "\n")
    (root / "metadata.json").write_text(json.dumps({
        "team_id": "team-1",
        "domain": "corporate_enterprise",
        "language_scope": ["en"],
        "african_alpha_claim": False,
        "budget_laptop_claim": True,
        "submitter": {
            "name": "Ada Example",
            "email": "ada@example.test",
            "github_handle": "ada",
        },
        "cross_disciplinary_pairing": {
            "discipline": "finance_accounting",
            "load_bearing": True,
            "description": "Local language interpretation plus formal ledger constraints.",
        },
        "test_prompts": [
            {"prompt_id": "tp_001", "prompt": prompts[0]},
            {"prompt_id": "tp_002", "prompt": prompts[1]},
        ],
        "model": {
            "name": "Qwen3-1.7B-Q4_K_M",
            "runtime": "llama.cpp",
            "quantization": "GGUF Q4_K_M",
            "parameters_estimate": "1.7B",
            "packaging": "binary_bundle",
        },
        "_runtime": {"model_path": "model/qwen3-1.7b-q4_k_m.gguf"},
    }))
    (root / "download_model.sh").write_text("PRIMARY_MODEL_FILE=qwen3-1.7b-q4_k_m.gguf\n")
    for name in ("README.md", "REPORT.md", ".gitignore", "LICENSE"):
        (root / name).write_text("complete\n")


def test_valid_metadata_is_synchronized(tmp_path):
    _write_valid(tmp_path)
    assert check_metadata(tmp_path) == []


def test_identity_and_prompt_placeholders_fail(tmp_path):
    _write_valid(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["team_id"] = ""
    meta["submitter"]["email"] = ""
    meta["test_prompts"] = []
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    codes = {f.code for f in check_metadata(tmp_path)}
    assert {"team_id", "submitter_email", "test_prompts"} <= codes


def test_gate1_requires_profiler_and_media(tmp_path):
    _write_valid(tmp_path)
    codes = {f.code for f in run_checks(tmp_path, gate1=True)}
    assert {"submission_json", "screenshot", "video"} <= codes


def test_visible_prompts_must_match_metadata(tmp_path):
    _write_valid(tmp_path)
    path = tmp_path / "data/bare/prompts.jsonl"
    path.write_text(json.dumps({"id": "1", "prompt": "different", "kind": "visible"}) + "\n")
    assert "prompt_sync" in {f.code for f in check_metadata(tmp_path)}
