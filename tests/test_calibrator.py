import itertools
from bursa.calibrator import ModelConfidencePolicy
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
