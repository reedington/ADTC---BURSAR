import re
from bursa import repository as repo, projections as proj
from bursa.models import Proposal, ProposalLine, RecommendedAction
from bursa.reasoncodes import ReasonCode


def match(conn, txn) -> Proposal:
    txn_id = txn["transaction_id"]
    narration = txn["narration"] or ""

    # Rule 1: exact imported student id token in the narration.
    for token in re.findall(r"[A-Za-z]+-?\d+", narration):
        candidate = token.upper()
        if repo.student_exists(conn, candidate):
            outstanding = sum(proj.charge_balance(conn, c["charge_id"])
                              for c in repo.charges_for_student(conn, candidate))
            codes = [ReasonCode.EXACT_STUDENT_ID]
            if outstanding == txn["amount_minor"]:
                codes.append(ReasonCode.EXACT_OUTSTANDING_BALANCE)
            return Proposal(transaction_id=txn_id, source="deterministic",
                lines=[ProposalLine(student_id=candidate, amount_minor=txn["amount_minor"],
                                    reason_codes=codes)],
                recommended_action=RecommendedAction.AUTO,
                explanation="Exact student ID present in narration.")

    # No deterministic candidate -> unmatched (Phase-2 LLM path handles ambiguity).
    return Proposal(transaction_id=txn_id, source="deterministic", lines=[],
        recommended_action=RecommendedAction.UNMATCHED,
        explanation="No deterministic candidate; awaiting model path.")
