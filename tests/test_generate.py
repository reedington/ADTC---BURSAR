import os
import random
import subprocess
import sys
import pytest
from bursa_eval.synth.generate import generate, PoolExhaustionError
from bursa_eval.goldcheck import check_case
from bursa_eval.models import GoldCase


def _bad_over_allocation(rng):
    # allocates 60k to a 40k charge against a 50k txn -> INV-01 + INV-05 violations
    sid = f"STU-{rng.randint(1, 9999)}"
    return GoldCase(
        id=f"synth-bad-{sid}", scenario_family="name_match", language="en",
        guardian_family=f"synth-bad-{sid}", template_family="synth-bad", provenance="synthetic",
        setup={"term": {"id": "T1", "session": "s", "name": "t"},
               "students": [{"id": sid, "name": "Bad Case",
                             "charges": [{"fee_id": "FEE-TUITION", "amount_naira": 40000}]}]},
        transaction={"reference": f"NIPBAD{sid}", "date": "2026-02-14", "amount_naira": 50000},
        expected={"outcome": "auto",
                  "allocations": [{"student_id": sid, "fee_id": "FEE-TUITION", "amount_naira": 60000}]})


def test_all_generated_valid():
    cases = generate(base_seed=7, n=30)
    assert len(cases) == 30
    for c in cases:
        assert check_case(c) == []


def test_gate_fires_on_invalid_output(monkeypatch):
    from bursa_eval.synth import templates as tmod
    assert check_case(_bad_over_allocation(random.Random(1))) != []   # the gate rejects it directly
    monkeypatch.setitem(tmod.TEMPLATES, "synth_bad", _bad_over_allocation)
    # heavily weight the bad template; generation must emit only valid cases (bad regenerated away)
    cases = generate(base_seed=1, n=15, mix={"synth_exact_id": 1.0, "synth_bad": 9.0})
    assert len(cases) == 15
    assert all("synth-bad" not in c.id for c in cases)


def test_abstention_share_in_band_at_scale():
    cases = generate(base_seed=11, n=500)
    frac = sum(1 for c in cases if c.is_abstention()) / len(cases)
    assert 0.25 <= frac <= 0.35, f"abstention {frac:.0%} out of the 25-30% band"


def test_pool_exhaustion_raises_clearly():
    with pytest.raises(PoolExhaustionError):
        generate(base_seed=1, n=50, max_attempts=5)


def test_two_subprocess_byte_identical():
    def run():
        return subprocess.run([sys.executable, "tests/_gen_subprocess.py"],
                              capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONPATH": "."}).stdout
    out1, out2 = run(), run()
    assert out1 == out2 and len(out1) > 0
