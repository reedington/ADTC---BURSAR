import json

import pytest

from bursa.calibrator import FEATURE_KEYS, ModelConfidencePolicy
from bursa_eval.calibration import fit_artifact, verify_artifact


def _records(count=60):
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        value = 1.0 if positive else 0.0
        rows.append({
            "case_id": f"c-{index}",
            "label": positive,
            "features": {
                "features_version": 1,
                **{key: value for key in FEATURE_KEYS},
            },
        })
    return rows


def test_fit_is_deterministic_and_portable():
    records = _records()
    threshold = _records(200)
    first = fit_artifact(
        records,
        threshold,
        model_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )
    second = fit_artifact(
        records,
        threshold,
        model_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )
    assert first.to_dict() == second.to_dict()
    assert first.auto_enabled is True
    assert verify_artifact(first, threshold)["passed"] is True
    assert ModelConfidencePolicy(first).score(records[0]["features"]) > 0.9


def test_insufficient_classes_keeps_auto_disabled():
    rows = _records(20)
    artifact = fit_artifact(
        rows,
        rows,
        model_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )
    assert artifact.auto_enabled is False


def test_fit_requires_held_out_threshold_records():
    with pytest.raises(ValueError, match="threshold selection"):
        fit_artifact(
            _records(),
            [],
            model_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
        )
