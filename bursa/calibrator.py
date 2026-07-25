from bursa.models import RecommendedAction

# v1 unweighted-average score, recorded for Phase-3 training; NOT used for routing yet.
_SCORE_KEYS = ("name_alias_similarity", "guardian_relationship", "amount_to_balance_agreement",
               "historical_payer_consistency", "candidate_separation", "llm_ranking_consistency",
               "constraint_validation_result")


class ModelConfidencePolicy:
    """Model-path confidence seam. v1 is UNTRAINED: it records a provisional score but always
    routes to review, so the model never auto-posts (zero-false-auto-post holds trivially).
    Phase 3 replaces route()'s internals with the logistic model fit on recorded features."""

    def score(self, features: dict) -> float:
        vals = [float(features.get(k, 0)) for k in _SCORE_KEYS]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def route(self, features: dict) -> RecommendedAction:
        return RecommendedAction.REVIEW
