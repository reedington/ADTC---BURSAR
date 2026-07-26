"""goldcheck warns when an authored allocation split diverges from distribute()'s
natural split — otherwise exact_allocation_accuracy is silently unreachable for that case."""
import glob
from bursa_eval.goldcheck import load_case, distribute_divergence, check_case
from bursa_eval.models import (
    GoldCase, Setup, TermSpec, StudentSpec, ChargeSpec, TransactionSpec, Expected, Allocation)


def test_committed_examples_have_no_divergence():
    for p in sorted(glob.glob("data/gold/*.yaml")):
        assert distribute_divergence(load_case(p)) is None, f"{p} unexpectedly diverges"


def _override_case() -> GoldCase:
    # One student, two equal charges; the transaction covers exactly one. distribute() fills
    # the lower charge_id first (FEE-A), but the author overrides the split onto FEE-B.
    return GoldCase(
        id="unit-override", scenario_family="fee_item_split", language="en",
        guardian_family="x", template_family="x",
        setup=Setup(
            term=TermSpec(id="T1", session="2025/2026", name="second_term"),
            students=[StudentSpec(id="STU-X", name="Zed", charges=[
                ChargeSpec(fee_id="FEE-A", amount_naira=10000),
                ChargeSpec(fee_id="FEE-B", amount_naira=10000)])]),
        transaction=TransactionSpec(reference="REF-OVR-1", date="2026-02-15", amount_naira=10000),
        expected=Expected(outcome="review",
                          allocations=[Allocation(student_id="STU-X", fee_id="FEE-B", amount_naira=10000)]))


def test_manual_override_is_flagged():
    case = _override_case()
    assert check_case(case) == []                       # the override is a VALID post
    warn = distribute_divergence(case)
    assert warn is not None and "unit-override" in warn  # ...but exact-allocation is unreachable
