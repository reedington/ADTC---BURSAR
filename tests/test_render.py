import json
import random
from bursa_eval.synth.render import to_app_format, to_chat_format
from bursa_eval.synth.templates import TEMPLATES
from bursa.pipeline import ALLOWED_CODES


def test_app_format_uses_real_prompt_builder():
    case = TEMPLATES["synth_sibling_split"](random.Random(1))
    ex = to_app_format(case)
    assert ex is not None
    assert "<|im_start|>system" in ex["prompt"] and "/no_think" in ex["prompt"]
    data = json.loads(ex["completion"])
    assert data["transaction_id"] and data["recommended_action"] == "review"
    for a in data["candidate_allocations"]:
        assert a["student_id"] in ex["prompt"]          # target ids are among the shown candidates
        assert set(a["reason_codes"]) <= set(ALLOWED_CODES)   # no train/serve grammar skew


def test_app_format_identity_matches_live_prompt_builder():
    # the rendered prompt IS the live PromptBuilder output for the materialized case
    from bursa import repository as repo, candidates
    from bursa.inference import prompt as prompt_mod
    from bursa.inference.tokens import get_token_counter
    from bursa_eval import loader
    case = TEMPLATES["synth_exact_id"](random.Random(2))
    ex = to_app_format(case)
    conn = loader.materialize(case)
    txn_id = loader.insert_case_transaction(conn, case)
    txn = repo.get_transaction(conn, txn_id)
    cands = candidates.generate(conn, txn)
    live_prompt, _ = prompt_mod.build(txn, cands, get_token_counter(None), ALLOWED_CODES)
    conn.close()
    assert ex["prompt"] == live_prompt


def test_duplicate_blocked_excluded_from_both_renderers():
    case = TEMPLATES["synth_duplicate"](random.Random(1))
    assert to_app_format(case) is None
    assert to_chat_format(case) is None


def test_chat_format_has_no_system_prompt():
    case = TEMPLATES["synth_exact_id"](random.Random(1))
    ex = to_chat_format(case)
    assert ex is not None and "<|im_start|>system" not in ex["prompt"]
