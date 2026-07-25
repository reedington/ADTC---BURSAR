from bursa import matcher, distribute, ledger, repository as repo
from bursa.config import Config
from bursa.confidence import RuleBasedConfidencePolicy
from bursa.errors import InvariantViolation
from bursa.models import RecommendedAction


def reconcile(conn, txn_id, config: Config | None = None) -> str:
    config = config or Config()
    txn = repo.get_transaction(conn, txn_id)
    proposal = matcher.match(conn, txn)
    action = RuleBasedConfidencePolicy().route(proposal)

    if action == RecommendedAction.AUTO and config.auto_post_enabled:
        proposed = []
        for line in proposal.lines:
            events, _ = distribute.distribute(conn, txn_id, line.student_id,
                                              line.amount_minor, "engine")
            proposed.extend(events)
        try:
            ledger.post(conn, txn_id, proposed, "engine", "deterministic", txn_id, "auto")
            repo.set_routing_state(conn, txn_id, "auto")
            return "auto"
        except InvariantViolation:
            repo.set_routing_state(conn, txn_id, "review")  # INV-10
            return "review"

    state = ("review" if action in (RecommendedAction.AUTO, RecommendedAction.REVIEW)
             else "unmatched")
    repo.set_routing_state(conn, txn_id, state)
    return state
