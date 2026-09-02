import itertools
import json
import pytest
from bursa.calibrator import (
    CalibrationArtifact,
    CalibrationError,
    FEATURE_KEYS,
    ModelConfidencePolicy,
)
from bursa.models import RecommendedAction


def test_v1_always_review_even_on_maximal_features():
    pol = ModelConfidencePolicy()
    strong = {"name_alias_similarity": 1.0, "guardian_relationship": 1,
              "amount_to_balance_agreement": 1.0, "historical_payer_consistency": 1,
              "candidate_separation": 1.0, "llm_ranking_consistency": 1,
              "constraint_validation_result": 1}
    assert pol.route(strong) == RecommendedAction.REVIEW
    assert 0.0 <= pol.score(strong) <= 1.0
    assert pol.score(strong) == 1.0   # maximal score...


def test_v1_never_auto_across_the_feature_space():
    # exhaustive over binary features + score extremes: v1 must NEVER return auto/unmatched.
    pol = ModelConfidencePolicy()
    keys = ["guardian_relationship", "historical_payer_consistency",
            "llm_ranking_consistency", "constraint_validation_result"]
    for combo in itertools.product([0, 1], repeat=len(keys)):
        for sim in (0.0, 0.5, 1.0):
            feats = dict(zip(keys, combo))
            feats.update({"name_alias_similarity": sim, "amount_to_balance_agreement": sim,
                          "candidate_separation": sim})
            assert pol.route(feats) == RecommendedAction.REVIEW   # ...but routing is always review


def _artifact(*, intercept=10.0, auto_enabled=True, model_sha="a" * 64):
    return CalibrationArtifact(
        schema_version=1,
        features_version=1,
        model_sha256=model_sha,
        feature_names=FEATURE_KEYS,
        means=(0.0,) * len(FEATURE_KEYS),
        scales=(1.0,) * len(FEATURE_KEYS),
        intercept=intercept,
        coefficients=(0.0,) * len(FEATURE_KEYS),
        auto_threshold=0.90,
        review_threshold=0.65,
        auto_enabled=auto_enabled,
        fit_positive_count=25,
        fit_negative_count=25,
        seed=3407,
        source_manifest_sha256="b" * 64,
        threshold_case_count=200,
        threshold_negative_count=100,
        threshold_false_positive_count=0,
    )


def test_fitted_policy_routes_by_locked_bands():
    features = {"features_version": 1}
    assert ModelConfidencePolicy(_artifact(intercept=10)).route(features) == \
        RecommendedAction.AUTO
    assert ModelConfidencePolicy(_artifact(intercept=0)).route(features) == \
        RecommendedAction.UNMATCHED
    review = _artifact(intercept=1.0, auto_enabled=False)
    assert ModelConfidencePolicy(review).route(features) == RecommendedAction.REVIEW


def test_artifact_file_rejects_model_hash_mismatch(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_artifact().to_dict()))
    with pytest.raises(CalibrationError, match="different model"):
        ModelConfidencePolicy.from_file(path, expected_model_sha256="c" * 64)


def test_incompatible_runtime_features_fail_closed_to_review():
    policy = ModelConfidencePolicy(_artifact())
    assert policy.route({"features_version": 999}) == RecommendedAction.REVIEW


def test_artifact_cannot_enable_auto_with_insufficient_fit_counts():
    value = _artifact().to_dict()
    value["fit_positive_count"] = 24
    with pytest.raises(CalibrationError, match="25 positive"):
        CalibrationArtifact.from_dict(value)


def test_artifact_rejects_truthy_string_auto_flag():
    value = _artifact().to_dict()
    value["auto_enabled"] = "false"
    with pytest.raises(CalibrationError, match="boolean"):
        CalibrationArtifact.from_dict(value)
