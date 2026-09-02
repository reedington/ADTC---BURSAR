import json
from bursa import repository as repo, candidates
from bursa.reasoncodes import ReasonCode as RC
from bursa.inference import prompt as prompt_mod
from bursa.inference.tokens import get_token_counter
from bursa_eval import loader
from bursa_eval.models import naira_to_minor

# Map each family to reason codes that ACTUALLY exist in ReasonCode (== ALLOWED_CODES), so the
# training target is grammar-valid and there is no train/serve skew.
FAMILY_REASON_CODES = {
    "name_match": [RC.EXACT_STUDENT_ID],
    "nickname_initials": [RC.MODEL_RANKED],
    "guardian_surname_differs": [RC.SINGLE_GUARDIAN_STUDENT],
    "sibling_split": [RC.SINGLE_GUARDIAN_STUDENT, RC.EXACT_OUTSTANDING_BALANCE, RC.MODEL_RANKED],
    "instalment": [RC.EXACT_OUTSTANDING_BALANCE],
    "underpayment": [RC.MODEL_RANKED],
    "overpayment": [RC.EXACT_OUTSTANDING_BALANCE],
    "fee_item_split": [RC.MODEL_RANKED],
    "known_payer": [RC.KNOWN_PAYER_MAPPING],
    "ambiguous_candidates": [RC.AMBIGUOUS_CANDIDATES],
    "no_candidate": [RC.NO_CANDIDATE],
    "ocr_substitution": [RC.MODEL_RANKED],
    "injection": [RC.MODEL_RANKED],
}


def _target_json(case, txn_id):
    codes = [str(c) for c in FAMILY_REASON_CODES.get(case.scenario_family, [])]
    allocs = [{"student_id": a.student_id, "amount_minor": naira_to_minor(a.amount_naira),
               "reason_codes": codes} for a in case.expected.allocations]
    return json.dumps({
        "transaction_id": txn_id,
        "interpretation": {"payer_name": case.transaction.payer_name or "",
                           "student_mentions": [], "term": case.setup.term.name,
                           "fee_types": sorted({a.fee_id for a in case.expected.allocations if a.fee_id}),
                           "payment_intent": case.scenario_family},
        "candidate_allocations": allocs,
        "recommended_action": case.expected.outcome,
        "explanation": case.expected.rationale, "ambiguities": []})


def to_app_format(case):
    """Constrained-JSON training example via the REAL Agent M path (no reimplemented prompt).
    Returns None for duplicate_blocked (import-layer) or when the correct student(s) were not
    pooled (a pool-recall gap, not a valid target)."""
    if case.expected.outcome == "duplicate_blocked":
        return None
    from bursa.pipeline import ALLOWED_CODES
    conn = loader.materialize(case)
    try:
        txn_id = loader.insert_case_transaction(conn, case)
        txn = repo.get_transaction(conn, txn_id)
        cands = candidates.generate(conn, txn)
        raw_prompt, surviving = prompt_mod.build(txn, cands, get_token_counter(None), ALLOWED_CODES)
        if raw_prompt is None:
            return None
        surviving_ids = {c.student_id for c in surviving}
        if not all(a.student_id in surviving_ids for a in case.expected.allocations):
            return None
        return {"prompt": raw_prompt, "completion": _target_json(case, txn_id)}
    finally:
        conn.close()


def to_chat_format(case):
    """Bare-model conversational example — no system prompt, self-contained scenario.
    Excluded (like app-format) for duplicate_blocked: import-layer behaviour the bare model
    must never learn to perform."""
    if case.expected.outcome == "duplicate_blocked":
        return None
    lines = [f"A bank transfer of NGN {case.transaction.amount_naira} arrived"
             f" (payer: {case.transaction.payer_name}, narration: \"{case.transaction.narration}\").",
             "Students and balances:"]
    for s in case.setup.students:
        bals = ", ".join(f"{c.fee_id} {c.amount_naira}" for c in s.charges)
        al = f" (aka {', '.join(s.aliases)})" if s.aliases else ""
        lines.append(f"- {s.id} {s.name}{al}: {bals}")
    lines.append("Who should this be allocated to, and how? If unclear, say it needs review.")
    # Keep this as a plain user turn. The training runner applies the selected
    # base model's embedded chat template, preserving bare-chat behavior.
    prompt = "\n".join(lines)
    answer = case.expected.rationale + " Recommended action: " + case.expected.outcome + "."
    return {"prompt": prompt, "completion": answer}
