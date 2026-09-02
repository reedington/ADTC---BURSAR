import json
from pathlib import Path

import pytest

from bursa_eval.baseline import (
    QWEN_LARGE,
    QWEN_SMALL,
    _selected_cases,
    c2_decision,
    score_review,
)


def _card(model_id, *, gold, tps, qualifying=True):
    return {
        "provenance": {"model_id": model_id},
        "bursa_gold": {"exact_allocation_accuracy": gold},
        "perf": {
            "generation_tokens_per_second": tps,
            "qualifying_i5": qualifying,
        },
        "gates_failed": [],
    }


def test_c2_speed_tripwire_selects_small(tmp_path):
    models = {
        QWEN_SMALL: _card(QWEN_SMALL, gold=.75, tps=20),
        QWEN_LARGE: _card(QWEN_LARGE, gold=.85, tps=12),
        "gemma-3-1b-it-q4_k_m": _card("gemma-3-1b-it-q4_k_m", gold=.4, tps=16),
        "llama-3.2-3b-instruct-q4_k_m": _card(
            "llama-3.2-3b-instruct-q4_k_m", gold=.5, tps=8
        ),
    }
    for model_id, card in models.items():
        path = tmp_path / model_id
        path.mkdir()
        (path / "scorecard.json").write_text(json.dumps(card))
    scores = {model_id: 70 for model_id in models}
    decision = c2_decision(tmp_path, scores)
    assert decision["selected_model_id"] == QWEN_SMALL
    assert set(decision["four_model_table"]) == set(models)


def test_c2_refuses_non_i5_measurement(tmp_path):
    models = {
        QWEN_SMALL: _card(QWEN_SMALL, gold=.8, tps=20),
        QWEN_LARGE: _card(QWEN_LARGE, gold=.85, tps=15, qualifying=False),
        "gemma": _card("gemma", gold=.4, tps=16),
        "llama": _card("llama", gold=.5, tps=8),
    }
    for model_id, card in models.items():
        path = tmp_path / model_id
        path.mkdir()
        (path / "scorecard.json").write_text(json.dumps(card))
    with pytest.raises(ValueError, match="i5"):
        c2_decision(tmp_path, {model_id: 70 for model_id in models})


def test_blind_rubric_scores_normalize_to_100(tmp_path):
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({
        "review_id": "blind-0001",
        "scores": {
            "correctness": 2,
            "coherence": 2,
            "no_schema_leak": 2,
            "appropriate_abstention": 2,
        },
    }) + "\n")
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"mapping": {"blind-0001": "model-a"}}))
    assert score_review(packet, key) == {"model-a": 100.0}


def test_sealed_test_selection_requires_every_frozen_case(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    source = Path("data/gold/gold-0003-sibling-split-en.yaml").read_text(
        encoding="utf-8"
    )
    (gold / "sibling_split.yaml").write_text(source, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "frozen": True,
        "test_case_ids": ["gold-0003-sibling-split"],
    }), encoding="utf-8")
    selected = _selected_cases(
        str(gold), str(manifest), False, split="test"
    )
    assert [case.id for case in selected] == ["gold-0003-sibling-split"]

    manifest.write_text(json.dumps({
        "frozen": True,
        "test_case_ids": ["gold-0003-sibling-split", "missing"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        _selected_cases(str(gold), str(manifest), False, split="test")
