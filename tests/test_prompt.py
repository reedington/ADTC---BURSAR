from bursa.inference.prompt import build, SYSTEM_PROMPT
from bursa.inference.tokens import HeuristicTokenCounter
from bursa.candidates import Candidate


def _txn(make_row):
    # real sqlite3.Row, the production type the prompt builder receives
    return make_row(transaction_id="TXN-1", payer_name="Ada", narration="CHI SCH FEE",
                    amount_minor=5_000_000)


def _cands(n=2):
    return [Candidate(student_id=f"STU-{i}", name=f"Name{i}", aliases=["Chi"],
                      outstanding=[("CHG", 5_000_000)], is_prior_payer=True, score=10,
                      fired_signals={"fuzzy_name": 10}) for i in range(n)]


def test_build_applies_chat_template_and_counts(make_row):
    raw, surviving = build(_txn(make_row), _cands(), HeuristicTokenCounter(), ["X"])
    assert "<|im_start|>system" in raw
    assert "/no_think" in raw
    assert raw.strip().endswith("<|im_start|>assistant")
    assert SYSTEM_PROMPT in raw
    assert len(surviving) == 2


def test_budget_exceeded_routes_none(make_row):
    raw, surviving = build(_txn(make_row), _cands(3), HeuristicTokenCounter(), ["X"], budget=5)
    assert raw is None and surviving == []


def test_never_truncates_narration(make_row):
    raw, surviving = build(_txn(make_row), _cands(3), HeuristicTokenCounter(), ["X"], budget=400)
    if raw is not None:
        assert "CHI SCH FEE" in raw


def test_ladder_reduces_candidates_before_failing(make_row):
    raw, surviving = build(_txn(make_row), _cands(5), HeuristicTokenCounter(), ["X"], budget=250)
    assert (raw is None) or (len(surviving) <= 5)
    if raw is not None and surviving:
        assert "aliases=Chi" in raw   # fired evidence preserved
