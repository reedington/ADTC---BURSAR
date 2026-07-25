import os
import subprocess
import sys
from bursa_eval.synth.generate import generate
from bursa_eval.goldcheck import check_case


def test_all_generated_valid():
    cases = generate(base_seed=7, n=30)
    assert len(cases) == 30
    for c in cases:
        assert check_case(c) == []


def test_abstention_floor_met_by_construction():
    cases = generate(base_seed=7, n=40)
    abstain = sum(1 for c in cases if c.is_abstention())
    assert abstain >= 0.25 * len(cases)


def test_two_subprocess_byte_identical():
    # process-salt regression guard: two fresh interpreters must emit identical bytes
    def run():
        return subprocess.run([sys.executable, "tests/_gen_subprocess.py"],
                              capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONPATH": "."}).stdout
    out1, out2 = run(), run()
    assert out1 == out2 and len(out1) > 0
