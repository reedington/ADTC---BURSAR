from typing import Protocol
from bursa.models import Proposal, RecommendedAction


class ConfidencePolicy(Protocol):
    def route(self, proposal: Proposal) -> RecommendedAction: ...


class RuleBasedConfidencePolicy:
    """Phase-1: the deterministic matcher already decided; pass through.
    The D11 logistic calibrator replaces this class behind the same interface."""

    def route(self, proposal: Proposal) -> RecommendedAction:
        return proposal.recommended_action
