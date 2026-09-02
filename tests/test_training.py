import json

import pytest

from bursa_eval.training import (
    c3_decision,
    load_config,
    render_training_text,
    semantic_app_output,
    validate_dataset,
)


class Tokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return f"USER:{messages[0]['content']}\nASSISTANT:{messages[1]['content']}"


def test_locked_training_config():
    config = load_config("configs/training/qwen3-lora.toml")
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["trainer"]["micro_batch_size"] * \
        config["trainer"]["gradient_accumulation_steps"] == 32


def test_render_uses_production_envelope_only_for_app():
    app = render_training_text({
        "id": "a",
        "format": "app",
        "prompt": "<|im_start|>assistant",
        "completion": "{}",
    }, Tokenizer())
    assert app.endswith("{}<|im_end|>")
    chat = render_training_text({
        "id": "c", "format": "chat", "prompt": "Question", "completion": "Answer"
    }, Tokenizer())
    assert chat == "USER:Question\nASSISTANT:Answer"


def test_dataset_must_match_frozen_manifest(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(
        json.dumps({"format": "app", "prompt": "p", "completion": "c"}) + "\n"
        + json.dumps({"format": "chat", "prompt": "p", "completion": "c"}) + "\n"
    )
    val.write_text(json.dumps({"format": "chat", "prompt": "p", "completion": "c"}) + "\n")
    from bursa_eval.repro import sha256_file
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "frozen": True,
        "files": {
            "train.jsonl": sha256_file(train),
            "val.jsonl": sha256_file(val),
        },
    }))
    assert validate_dataset(train, val, manifest)["train_rows"] == 2
    train.write_text("changed")
    with pytest.raises(ValueError, match="manifest"):
        validate_dataset(train, val, manifest)


def test_c3_selects_fine_tuned_only_when_every_gate_passes():
    zero = {
        "bursa_gold": {"exact_allocation_accuracy": .70},
        "adtc": {"accuracy": .60},
        "gates_failed": [],
    }
    candidate = {
        "bursa_gold": {"exact_allocation_accuracy": .75},
        "adtc": {"accuracy": .59},
        "gates_failed": [],
    }
    stress = {
        "valid_json_rate": .999,
        "unsupported_ids": 0,
        "thinking_leaks": 0,
        "incorrect_auto_posts": 0,
        "passes": True,
    }
    decision = c3_decision(
        zero, candidate, stress, zero_bare_score=80, candidate_bare_score=78
    )
    assert decision["ship"] == "fine_tuned"
    candidate["adtc"]["accuracy"] = .55
    decision = c3_decision(
        zero, candidate, stress, zero_bare_score=80, candidate_bare_score=78
    )
    assert decision["ship"] == "zero_shot"
    assert decision["retry_60_40"] is True


def test_semantic_smoke_parser_ignores_explanation_wording():
    first = semantic_app_output(
        'prefix {"candidate_allocations":[{"student_id":"S1","amount_minor":10}],'
        '"recommended_action":"review","explanation":"one"}'
    )
    second = semantic_app_output(
        '{"recommended_action":"review","candidate_allocations":'
        '[{"amount_minor":10,"student_id":"S1"}],"explanation":"two"}'
    )
    assert first == second
