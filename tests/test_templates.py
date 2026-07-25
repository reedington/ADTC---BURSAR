import random
from bursa_eval.synth.templates import TEMPLATES
from bursa_eval.goldcheck import check_case


def test_every_template_generates_valid_cases():
    for tid, gen in TEMPLATES.items():
        for seed in range(5):
            case = gen(random.Random(seed))
            assert case.provenance == "synthetic"
            assert case.guardian_family.startswith("synth-")
            assert case.template_family.startswith("synth-")
            assert check_case(case) == [], f"{tid} seed {seed}: {check_case(case)}"


def test_abstention_template_has_empty_allocations_and_pool_truth():
    case = TEMPLATES["synth_pidgin_ambiguous"](random.Random(1))
    assert case.is_abstention()
    assert case.expected.pool_must_include
