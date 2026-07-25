import json
from datetime import datetime, timezone
from bursa import matcher, distribute, ledger, repository as repo, candidates, features, constraints
from bursa.config import Config
from bursa.calibrator import ModelConfidencePolicy
from bursa.errors import InvariantViolation
from bursa.models import RecommendedAction
from bursa.reasoncodes import ReasonCode
from bursa.inference import prompt as prompt_mod, grammar as grammar_mod, schema as schema_mod
from bursa.inference.backend import BackendTransportError
from bursa.inference.run import run_inference
from bursa.inference.tokens import get_token_counter
from bursa.inference.constants import OUTPUT_MAX, CONTEXT_CAP, SAFETY_MARGIN

ALLOWED_CODES = [c.value for c in ReasonCode]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _store_review(conn, txn_id, source, reason, explanation, features_json=None):
    pid = f"PROP-{txn_id}-{source}"
    full = f"{reason}: {explanation}" if reason else explanation
    repo.insert_proposal(conn, pid, txn_id, source, RecommendedAction.REVIEW, None, full, _now())
    if features_json is not None:
        conn.execute("UPDATE proposals SET features = ? WHERE proposal_id = ?",
                     (features_json, pid))
    repo.set_routing_state(conn, txn_id, "review")
    return "review"


def reconcile(conn, txn_id, config: Config | None = None, backend=None,
              tokenizer_path: str | None = None) -> str:
    config = config or Config()
    txn = repo.get_transaction(conn, txn_id)

    # Exact deterministic path (Phase 1) — never reinterpreted by the model.
    p = matcher.match(conn, txn)
    if p.recommended_action == RecommendedAction.AUTO:
        if config.auto_post_enabled:
            proposed = []
            for line in p.lines:
                events, _ = distribute.distribute(conn, txn_id, line.student_id,
                                                  line.amount_minor, "engine")
                proposed.extend(events)
            try:
                ledger.post(conn, txn_id, proposed, "engine", "deterministic", txn_id, "auto")
                repo.set_routing_state(conn, txn_id, "auto")
                return "auto"
            except InvariantViolation:
                return _store_review(conn, txn_id, "deterministic", "invariant", "auto->review")
        return _store_review(conn, txn_id, "deterministic", "flag_off", "auto disabled")

    # Non-exact -> model path. With no backend, fall back to Phase-1 behaviour (unmatched),
    # so existing Phase-1 pipeline tests (which call reconcile without a backend) stay green.
    if backend is None:
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"
    cands = candidates.generate(conn, txn)
    if not cands:
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"

    counter = get_token_counter(tokenizer_path)
    raw_prompt, surviving = prompt_mod.build(txn, cands, counter, ALLOWED_CODES)
    if raw_prompt is None:
        return _store_review(conn, txn_id, "llm", ReasonCode.PROMPT_BUDGET_EXCEEDED, "budget exceeded")

    ids = [c.student_id for c in surviving]
    grammar = grammar_mod.build_grammar(txn_id, ids, ALLOWED_CODES)
    n_predict = min(OUTPUT_MAX, CONTEXT_CAP - counter.count(raw_prompt) - SAFETY_MARGIN)
    try:
        raw = run_inference(backend, raw_prompt, grammar, n_predict)
    except BackendTransportError:
        return _store_review(conn, txn_id, "llm", ReasonCode.INFERENCE_UNAVAILABLE, "backend down")

    # Content-invalid -> review immediately (NO retry; temp 0 is deterministic).
    outcome = schema_mod.validate_output(raw, txn_id, ids, ALLOWED_CODES)
    if not outcome.ok:
        return _store_review(conn, txn_id, "llm", ReasonCode.SCHEMA_INVALID, outcome.reason)

    data = outcome.data
    allocs = data.get("candidate_allocations", [])
    chosen_id = allocs[0]["student_id"] if allocs else None
    dry_ok = True
    if chosen_id:
        events = []
        for a in allocs:
            evs, _ = distribute.distribute(conn, txn_id, a["student_id"], a["amount_minor"], "model")
            events.extend(evs)
        dry_ok = constraints.validate(conn, txn, events).ok   # dry-run = feature, NOT a post
    feats = features.extract(txn, surviving, data, dry_ok, chosen_id,
                             budget_shed=(len(surviving) < len(cands)))
    ModelConfidencePolicy().route(feats)   # v1 -> review
    return _store_review(conn, txn_id, "llm", ReasonCode.MODEL_RANKED,
                         data.get("explanation", ""), json.dumps(feats))
