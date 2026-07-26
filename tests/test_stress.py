from bursa_eval.dataset import near_dup_signature
from bursa_eval.stress import MODEL_ONLY_MIX
from bursa_eval.synth.generate import generate


def test_stress_mix_generates_distinct_model_path_cases():
    cases = generate(20260726, 25, mix=MODEL_ONLY_MIX)
    assert len(cases) == 25
    assert len({near_dup_signature(case) for case in cases}) == 25
    assert {case.scenario_family for case in cases} <= {
        "sibling_split", "overpayment", "ambiguous_candidates"
    }
