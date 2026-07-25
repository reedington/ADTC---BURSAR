from bursa_eval.goldcheck import check_case
from bursa_eval.models import GoldCase


def _base(**over):
    base = {
        "id": "g1", "scenario_family": "sibling_split", "language": "en",
        "guardian_family": "okafor", "template_family": "t1",
        "setup": {"term": {"id": "T1", "session": "2025/2026", "name": "second_term"},
                  "guardians": [{"id": "G1", "name": "Ada Okafor", "phone_suffix": "1049"}],
                  "students": [
                      {"id": "STU-1", "name": "Chidi Okafor", "aliases": ["Chi"], "guardians": ["G1"],
                       "charges": [{"fee_id": "FEE-TUITION", "amount_naira": 40000}]},
                      {"id": "STU-2", "name": "Somto Okafor", "aliases": ["Somto"], "guardians": ["G1"],
                       "charges": [{"fee_id": "FEE-TUITION", "amount_naira": 35000}]}]},
        "transaction": {"reference": "NIP1", "date": "2026-02-14", "amount_naira": 75000,
                        "payer_name": "C N Okafor", "narration": "CHI AND SOMTO"},
        "expected": {"outcome": "review", "allocations": [
            {"student_id": "STU-1", "fee_id": "FEE-TUITION", "amount_naira": 40000},
            {"student_id": "STU-2", "fee_id": "FEE-TUITION", "amount_naira": 35000}]},
    }
    base.update(over)
    return GoldCase(**base)


def test_valid_sibling_split_passes():
    assert check_case(_base()) == []


def test_over_allocation_flagged_by_constraint_engine():
    bad = _base(expected={"outcome": "review", "allocations": [
        {"student_id": "STU-1", "fee_id": "FEE-TUITION", "amount_naira": 40000},
        {"student_id": "STU-2", "fee_id": "FEE-TUITION", "amount_naira": 40000}]})  # 80k > 75k
    problems = check_case(bad)
    assert problems and any("violates" in p for p in problems)


def test_instalment_history_reduces_balance():
    case = _base(
        id="inst", scenario_family="instalment",
        setup={"term": {"id": "T1", "session": "s", "name": "t"},
               "students": [{"id": "STU-1", "name": "Chidi", "charges": [
                   {"fee_id": "FEE-TUITION", "amount_naira": 40000}]}],
               "history": [{"transaction": {"reference": "NIP-OLD", "date": "2026-01-10",
                            "amount_naira": 20000, "payer_name": "x"},
                            "allocations": [{"student_id": "STU-1", "fee_id": "FEE-TUITION",
                                             "amount_naira": 20000}]}]},
        transaction={"reference": "NIP-NEW", "date": "2026-02-01", "amount_naira": 20000},
        expected={"outcome": "auto", "allocations": [
            {"student_id": "STU-1", "fee_id": "FEE-TUITION", "amount_naira": 20000}]})
    assert check_case(case) == []   # 20k prior + 20k now == 40k billed, valid


def test_overpayment_credit_asserted():
    case = _base(
        id="over", scenario_family="overpayment",
        setup={"term": {"id": "T1", "session": "s", "name": "t"},
               "students": [{"id": "STU-1", "name": "Chidi", "charges": [
                   {"fee_id": "FEE-TUITION", "amount_naira": 40000}]}]},
        transaction={"reference": "NIP-OVER", "date": "2026-02-01", "amount_naira": 50000},
        expected={"outcome": "review",
                  "allocations": [{"student_id": "STU-1", "fee_id": "FEE-TUITION", "amount_naira": 40000}],
                  "credits": [{"holder": "STU-1", "amount_naira": 10000}]})
    assert check_case(case) == []   # 40k alloc + 10k credit == 50k txn (conservation holds)


def test_duplicate_blocked_requires_reference_in_history():
    case = _base(
        id="dup", scenario_family="duplicate_reference",
        setup={"term": {"id": "T1", "session": "s", "name": "t"},
               "students": [{"id": "STU-1", "name": "Chidi", "charges": [
                   {"fee_id": "FEE-TUITION", "amount_naira": 40000}]}],
               "history": [{"transaction": {"reference": "NIP-DUP", "date": "2026-01-10",
                            "amount_naira": 40000, "payer_name": "x"},
                            "allocations": [{"student_id": "STU-1", "fee_id": "FEE-TUITION",
                                             "amount_naira": 40000}]}]},
        transaction={"reference": "NIP-DUP", "date": "2026-02-01", "amount_naira": 40000},
        expected={"outcome": "duplicate_blocked", "allocations": []})
    assert check_case(case) == []
