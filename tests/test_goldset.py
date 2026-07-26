import glob
import yaml
from bursa_eval.goldcheck import load_case, check_case
from bursa_eval.goldnew import render


def test_all_gold_examples_valid():
    """CI gate: every committed gold case must validate (schema + financial invariants)."""
    paths = glob.glob("data/gold/*.yaml")
    assert len(paths) >= 14, "expected at least one worked example for every scenario family"
    for p in paths:
        case = load_case(p)
        assert check_case(case) == [], f"{p} failed validation"


def test_gold_examples_cover_every_required_family():
    from bursa_eval.models import SCENARIO_FAMILIES

    cases = [load_case(p) for p in glob.glob("data/gold/*.yaml")]
    assert {c.scenario_family for c in cases} == SCENARIO_FAMILIES
    assert any(c.language == "pcm" for c in cases)
    assert any(c.is_abstention() for c in cases)


def test_scaffold_renders_loadable_yaml():
    text = render("gold-XXXX-sibling-split", "sibling_split", "en", "hard")
    data = yaml.safe_load(text)
    assert data["scenario_family"] == "sibling_split"
    assert data["language"] == "en" and data["difficulty"] == "hard"
    assert "setup" in data and "expected" in data
