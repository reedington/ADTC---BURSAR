from bursa.inference.backend import FakeBackend
from bursa_eval.harness.baremodel import run_bare_suite, load_bare_prompts, BareRecord


def test_format_leak_detected():
    prompts = [{"id": "d14-1", "prompt": "Summarize this memo.", "kind": "visible"}]
    leaky = FakeBackend(chat_response='{"recommended_action":"review","candidate_allocations":[]}')
    rec = run_bare_suite(prompts, leaky)[0]
    assert rec.format_leak is True        # reconciliation JSON on a generic prompt = leak
    assert rec.valid is True              # non-empty output


def test_clean_generic_answer_has_no_leak():
    prompts = [{"id": "d14-2", "prompt": "Draft a one-line thank-you.", "kind": "visible"}]
    clean = FakeBackend(chat_response="Thank you for your prompt payment this term.")
    rec = run_bare_suite(prompts, clean)[0]
    assert rec.format_leak is False
    assert rec.valid is True
    assert rec.output == "Thank you for your prompt payment this term."


def test_visible_prompts_ship_verbatim():
    prompts = load_bare_prompts("data/bare/prompts.jsonl")
    ids = {p["id"] for p in prompts}
    assert {"d14-visible-1", "d14-visible-2"} <= ids   # the two judge-guaranteed prompts
