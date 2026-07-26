import glob
import json
from bursa.inference.backend import FakeBackend
from bursa_eval.goldcheck import load_case
from bursa_eval.harness.runner import run_gold_suite
from bursa_eval.harness.scorecard import build_scorecard, write_run, diff_runs, sidebyside, main


def _gold_cases():
    return [load_case(p) for p in sorted(glob.glob("data/gold/*.yaml"))]


def test_scorecard_has_gates_and_provenance():
    recs = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    sc = build_scorecard(recs, provenance={"git_commit": "abc123", "model_path": None,
                                            "model_sha256": None, "backend": "fake", "seeds": {}})
    assert "incorrect_auto_posts" in sc["bursa_gold"]
    assert "duplicate_blocked_rate" in sc["bursa_gold"]
    assert sc["provenance"]["git_commit"] == "abc123"
    assert "gates_failed" in sc


def test_write_and_diff_two_identical_runs(tmp_path):
    a = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    write_run(str(tmp_path / "A"), a, build_scorecard(a, provenance={}))
    b = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    write_run(str(tmp_path / "B"), b, build_scorecard(b, provenance={}))
    assert diff_runs(str(tmp_path / "A"), str(tmp_path / "B")) == []


def test_diff_detects_regression(tmp_path):
    good = run_gold_suite(_gold_cases(), FakeBackend(response="{}"))
    write_run(str(tmp_path / "A"), good, build_scorecard(good, provenance={}))
    # Corrupt B's exact case so a passing metric flips to False.
    write_run(str(tmp_path / "B"), good, build_scorecard(good, provenance={}))
    recs = [json.loads(l) for l in open(str(tmp_path / "B" / "records.jsonl")) if l.strip()]
    for r in recs:
        if r["exact_alloc_hit"] is True:
            r["exact_alloc_hit"] = False
    with open(str(tmp_path / "B" / "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    diff = diff_runs(str(tmp_path / "A"), str(tmp_path / "B"))
    assert any(d["metric"] == "exact_alloc_hit" for d in diff)


def test_sidebyside_compares_two_runs(tmp_path):
    from bursa_eval.harness.baremodel import BareRecord
    from dataclasses import asdict
    zero = build_scorecard([], bare_records=[BareRecord("bare_model", "p1", "Draft?", "zero out", True, False)],
                           provenance={})
    cand = build_scorecard([], bare_records=[BareRecord("bare_model", "p1", "Draft?", "cand out", True, False)],
                           provenance={})
    write_run(str(tmp_path / "Z"), [], zero)
    write_run(str(tmp_path / "C"), [], cand)
    md = sidebyside(str(tmp_path / "Z"), str(tmp_path / "C"))
    assert "zero out" in md and "cand out" in md and "Draft?" in md


def test_main_smoke_writes_artifacts(tmp_path):
    code = main(["--backend", "fake", "--out", str(tmp_path), "--label", "smoke"])
    run = tmp_path / "smoke"
    assert (run / "scorecard.json").exists()
    assert (run / "records.jsonl").exists()
    sc = json.loads((run / "scorecard.json").read_text())
    assert "gates_failed" in sc
    assert sc["provenance"]["backend"] == "fake"
    assert code == 0   # committed gold set passes both gates
    # The committed suite contains an exact auto-post and a duplicate-reference case, so both
    # money-safety gates are genuinely exercised rather than vacuously green.
    assert "incorrect_auto_posts" not in sc["gates_not_exercised"]
    assert "duplicate_blocked_rate" not in sc["gates_not_exercised"]
