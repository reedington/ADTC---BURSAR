import os
import random
import pytest
from bursa_eval.dataset import near_dup_signature, split, build, freeze, load_manifest
from bursa_eval.models import GoldCase
from bursa_eval.synth.templates import TEMPLATES


def _gold(fam, tfam, sid="STU-1", prov="team_authored"):
    # distinct name + narration per sid so cases are NOT accidental near-duplicates
    return GoldCase(id=f"c-{fam}-{tfam}-{sid}", scenario_family="name_match", language="en",
        guardian_family=fam, template_family=tfam, provenance=prov,
        setup={"term": {"id": "T1", "session": "s", "name": "t"},
               "students": [{"id": sid, "name": f"Name {sid}",
                             "charges": [{"fee_id": "FEE-TUITION", "amount_naira": 1000}]}]},
        transaction={"reference": f"R{sid}", "date": "2026-01-01", "amount_naira": 1000,
                     "narration": f"pay {sid}"},
        expected={"outcome": "auto",
                  "allocations": [{"student_id": sid, "fee_id": "FEE-TUITION", "amount_naira": 1000}]})


def test_synthetic_only_in_train():
    synth = TEMPLATES["synth_exact_id"](random.Random(1))
    golds = [_gold(f"g{i}", f"t{i}", f"STU-{i}") for i in range(9)]
    result = split(golds + [synth], base_seed=1)
    assert synth.id in result["train"]
    assert synth.id not in result["val"] and synth.id not in result["test"]


def test_no_family_straddles_boundary():
    a = _gold("shared", "ta", "STU-A")
    b = _gold("shared", "tb", "STU-B")     # shares guardian_family "shared" with a
    others = [_gold(f"g{i}", f"t{i}", f"STU-{i}") for i in range(8)]
    result = split([a, b] + others, base_seed=1)
    where = {cid: s for s, ids in result.items() for cid in ids}
    assert where[a.id] == where[b.id]


def test_pinned_test_stays_and_new_train_edge_raises():
    golds = [_gold(f"g{i}", f"t{i}", f"STU-{i}") for i in range(12)]
    first = split(golds, base_seed=1)
    pinned = {"val": first["val"], "test": first["test"]}
    assert first["test"], "expected a non-empty test split"
    again = split(golds, base_seed=1, pinned=pinned)
    assert set(again["test"]) >= set(pinned["test"])
    tcase_fam = next(g.guardian_family for g in golds if g.id in pinned["test"])
    intruder = _gold(tcase_fam, "t-new", "STU-NEW")   # new edge toward a pinned test case
    with pytest.raises(ValueError):
        split(golds + [intruder], base_seed=1, pinned=pinned)


def test_realized_ratios_reasonable():
    golds = [_gold(f"g{i}", f"t{i}", f"STU-{i}") for i in range(20)]
    r = split(golds, base_seed=3)
    assert len(r["train"]) >= 10 and r["val"] and r["test"]


def test_freeze_and_reload_manifest(tmp_path):
    p = str(tmp_path / "manifest.json")
    freeze({"train": ["a"], "val": ["b"], "test": ["c"]}, path=p,
           base_seed=1, synth_config_hash="x", gold_count=3, synth_count=0)
    m = load_manifest(p)
    assert m["frozen"] and m["test_case_ids"] == ["c"] and m["val_case_ids"] == ["b"]


def test_build_produces_splits_jsonl_and_coverage(tmp_path):
    out = str(tmp_path / "build")
    result = build(base_seed=5, n_synth=12, gold_dir="data/gold", out_dir=out)
    assert result["gold"] >= 14 and result["synth"] == 12
    for name in ("train", "val", "test"):
        assert os.path.exists(f"{out}/{name}.jsonl")
    for name in ("val", "test"):            # synthetic (synth-*) never leaves train
        assert all(not cid.startswith("synth-") for cid in result["assignment"][name])
    assert "train" in result["coverage"]


def test_release_build_enforces_mixture_and_calibration_isolation(tmp_path):
    import json
    from bursa_eval.dataset import build_release

    messages = []
    for index in range(40):
        messages.extend([
            {
                "message_id": f"p{index}",
                "role": "prompter",
                "lang": "en",
                "text": f"Explain fictional operational control number {index} clearly.",
            },
            {
                "message_id": f"a{index}",
                "parent_id": f"p{index}",
                "role": "assistant",
                "lang": "en",
                "rank": 0,
                "text": f"Control {index} should have an owner, evidence, and review date.",
            },
        ])
    oasst = tmp_path / "oasst.jsonl"
    oasst.write_text("\n".join(json.dumps(row) for row in messages))
    out = tmp_path / "release"
    manifest = build_release(
        base_seed=9,
        n_synth=12,
        n_calibration=6,
        out_dir=str(out),
        oasst_path=str(oasst),
        oasst_revision="pinned",
        require_final_gold=False,
    )
    assert manifest["mixture"]["general_fraction_target"] == 0.30
    total = (
        manifest["reconciliation_train_rows"] + manifest["general_train_rows"]
    )
    assert abs(manifest["general_train_rows"] / total - 0.30) < 0.02
    assert manifest["calibration_fit_count"] == 4
    assert manifest["calibration_threshold_count"] == 2
    assert manifest["split_leakage_errors"] == []
    assert manifest["synthetic_rendered_row_count"] == 24
    synthetic_rows = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text().splitlines()
        if json.loads(line).get("provenance") == "synthetic"
    ]
    assert len(synthetic_rows) == 24
    assert all(
        {row["format"] for row in synthetic_rows if row["case_id"] == case_id}
        == {"app", "chat"}
        for case_id in {row["case_id"] for row in synthetic_rows}
    )
    second_out = tmp_path / "release-again"
    second = build_release(
        base_seed=9,
        n_synth=12,
        n_calibration=6,
        out_dir=str(second_out),
        oasst_path=str(oasst),
        oasst_revision="pinned",
        require_final_gold=False,
    )
    assert second["files"] == manifest["files"]
