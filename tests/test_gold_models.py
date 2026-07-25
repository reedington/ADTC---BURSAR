import pytest
from pydantic import ValidationError
from bursa_eval.models import GoldCase, naira_to_minor


def _case(**over):
    base = {
        "id": "g1", "scenario_family": "sibling_split", "language": "en",
        "guardian_family": "okafor", "template_family": "t1",
        "setup": {"term": {"id": "T1", "session": "2025/2026", "name": "second_term"},
                  "students": [{"id": "STU-1", "name": "Chi", "charges": [
                      {"fee_id": "FEE-TUITION", "amount_naira": 40000}]}]},
        "transaction": {"date": "2026-02-14", "amount_naira": 75000},
        "expected": {"outcome": "review",
                     "allocations": [{"student_id": "STU-1", "amount_naira": 40000}]},
    }
    base.update(over)
    return base


def test_valid_case_loads():
    c = GoldCase(**_case())
    assert c.expected.outcome == "review"
    assert c.pool_truth() == ["STU-1"]
    assert not c.is_abstention()


def test_float_amount_rejected():
    with pytest.raises(ValidationError):
        GoldCase(**_case(transaction={"date": "2026-02-14", "amount_naira": 75000.50}))


def test_naira_to_minor_int_and_str():
    assert naira_to_minor(40000) == 4_000_000
    assert naira_to_minor("75000.00") == 7_500_000


def test_outcome_vocabulary_includes_duplicate_blocked():
    c = GoldCase(**_case(expected={"outcome": "duplicate_blocked", "allocations": []}))
    assert c.expected.outcome == "duplicate_blocked"
    assert c.is_abstention()   # empty allocations


def test_pool_must_include_overrides_default():
    c = GoldCase(**_case(expected={"outcome": "review", "allocations": [],
                                   "pool_must_include": ["STU-1", "STU-2"]}))
    assert c.pool_truth() == ["STU-1", "STU-2"]


def test_credits_field_and_unknown_family_rejected():
    c = GoldCase(**_case(expected={"outcome": "review",
        "allocations": [{"student_id": "STU-1", "amount_naira": 40000}],
        "credits": [{"holder": "STU-1", "amount_naira": 5000}]}))
    assert c.expected.credits[0].holder == "STU-1"
    with pytest.raises(ValidationError):
        GoldCase(**_case(scenario_family="not_a_family"))
