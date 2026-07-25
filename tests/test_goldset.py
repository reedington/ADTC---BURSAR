import glob
import yaml
from bursa_eval.goldcheck import load_case, check_case
from bursa_eval.goldnew import render


def test_all_gold_examples_valid():
    """CI gate: every committed gold case must validate (schema + financial invariants)."""
    paths = glob.glob("data/gold/*.yaml")
    assert len(paths) >= 3, "expected at least the three worked examples"
    for p in paths:
        case = load_case(p)
        assert check_case(case) == [], f"{p} failed validation"


def test_scaffold_renders_loadable_yaml():
    text = render("gold-XXXX-sibling-split", "sibling_split", "en", "hard")
    data = yaml.safe_load(text)
    assert data["scenario_family"] == "sibling_split"
    assert data["language"] == "en" and data["difficulty"] == "hard"
    assert "setup" in data and "expected" in data
