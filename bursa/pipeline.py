from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from bursa import (
    candidates,
    constraints,
    db as dbmod,
    distribute,
    features,
    ledger,
    matcher,
    projections,
    repository as repo,
)
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


def _evidence_snapshot(txn) -> dict:
    return {
        key: txn[key]
        for key in (
            "transaction_id",
            "source",
            "reference",
            "raw_reference",
            "posted_at",
            "payer_name",
            "narration",
            "amount_minor",
            "direction",
        )
    }


def _candidate_snapshot(items) -> list[dict]:
    return [asdict(item) for item in items]


def _deterministic_candidates(conn, txn, proposal) -> list:
    items = candidates.generate(conn, txn)
    present = {item.student_id for item in items}
    for line in proposal.lines:
        if line.student_id in present:
            continue
        student = conn.execute(
            "SELECT * FROM students WHERE student_id=?", (line.student_id,)
        ).fetchone()
        items.append(
            candidates.Candidate(
                student_id=line.student_id,
                name=student["name"],
                outstanding=[
                    (charge["charge_id"], projections.charge_balance(conn, charge["charge_id"]))
                    for charge in repo.charges_for_student(conn, line.student_id)
                    if projections.charge_balance(conn, charge["charge_id"]) > 0
                ],
                score=10,
                fired_signals={"exact_student_id": 10},
            )
        )
    return items


def _store_review(
    conn,
    txn,
    source,
    reason,
    explanation,
    *,
    candidate_items=None,
    raw_output=None,
    model_data=None,
    feature_values=None,
    confidence=None,
    allocations=None,
):
    txn_id = txn["transaction_id"]
    pid = f"PROP-{uuid4().hex}"
    full = f"{reason}: {explanation}" if reason else explanation
    repo.supersede_pending_proposals(conn, txn_id)
    repo.insert_proposal(
        conn,
        pid,
        txn_id,
        source,
        RecommendedAction.REVIEW,
        confidence,
        full,
        _now(),
        features=feature_values,
        candidates=_candidate_snapshot(candidate_items or []),
        evidence=_evidence_snapshot(txn),
        raw_output=raw_output,
        failure_reason=str(reason) if reason else None,
        ambiguities=(model_data or {}).get("ambiguities", []),
        allocations=allocations or (model_data or {}).get("candidate_allocations", []),
    )
    repo.set_routing_state(conn, txn_id, "review")
    return "review"


@dataclass
class ModelPathResult:
    """The model path's RAW result. reconcile() consumes it for routing/storage; the eval
    harness consumes it for scoring. Never posts. No harness-specific fields or mode flags."""
    surviving: list
    budget_shed: bool
    data: dict | None
    failure: str | None          # None | no_candidates | prompt_budget | transport | schema
    schema_reason: str | None
    dry_ok: bool
    chosen_id: str | None
    model_events: list = field(default_factory=list)
    raw_output: str | None = None


def run_model_path(conn, txn, backend, tokenizer_path: str | None = None) -> ModelPathResult:
    """Run the non-exact model path through the real serving components and return the model's
    raw validated output plus the charge-grain events its allocations distribute to."""
    txn_id = txn["transaction_id"]
    cands = candidates.generate(conn, txn)
    if not cands:
        return ModelPathResult([], False, None, "no_candidates", None, True, None)

    counter = get_token_counter(tokenizer_path)
    raw_prompt, surviving = prompt_mod.build(txn, cands, counter, ALLOWED_CODES)
    budget_shed = len(surviving) < len(cands)
    if raw_prompt is None:
        return ModelPathResult(surviving, budget_shed, None, "prompt_budget", None, True, None)

    ids = [c.student_id for c in surviving]
    grammar = grammar_mod.build_grammar(txn_id, ids, ALLOWED_CODES)
    n_predict = min(OUTPUT_MAX, CONTEXT_CAP - counter.count(raw_prompt) - SAFETY_MARGIN)
    try:
        raw = run_inference(backend, raw_prompt, grammar, n_predict)
    except BackendTransportError:
        return ModelPathResult(surviving, budget_shed, None, "transport", None, True, None)

    outcome = schema_mod.validate_output(raw, txn_id, ids, ALLOWED_CODES)
    if not outcome.ok:
        return ModelPathResult(
            surviving, budget_shed, None, "schema", outcome.reason, True, None,
            raw_output=raw,
        )

    data = outcome.data
    allocs = data.get("candidate_allocations", [])
    chosen_id = allocs[0]["student_id"] if allocs else None
    model_events, dry_ok = [], True
    if chosen_id:
        for a in allocs:
            evs, _ = distribute.distribute(conn, txn_id, a["student_id"], a["amount_minor"], "model")
            model_events.extend(evs)
        dry_ok = constraints.validate(conn, txn, model_events).ok   # dry-run = feature, NOT a post
    return ModelPathResult(
        surviving,
        budget_shed,
        data,
        None,
        None,
        dry_ok,
        chosen_id,
        model_events,
        raw,
    )


def reconcile(conn, txn_id, config: Config | None = None, backend=None,
              tokenizer_path: str | None = None) -> str:
    config = config or Config()
    txn = repo.get_transaction(conn, txn_id)
    if txn is None:
        raise KeyError(txn_id)
    # Reconciliation is idempotent once the transaction funds a live ledger event. A retry can
    # refresh an unposted proposal, but it can never create a second posting for the same money.
    if projections.txn_used(conn, txn_id) > 0:
        return txn["routing_state"]

    # Exact deterministic path (Phase 1) — never reinterpreted by the model.
    p = matcher.match(conn, txn)
    if p.recommended_action == RecommendedAction.AUTO:
        original_allocations = [line.model_dump() for line in p.lines]
        exact_candidates = _deterministic_candidates(conn, txn, p)
        financially_safe = all(
            line.amount_minor
            <= sum(
                max(0, projections.charge_balance(conn, charge["charge_id"]))
                for charge in repo.charges_for_student(conn, line.student_id)
            )
            for line in p.lines
        )
        if not financially_safe:
            return _store_review(
                conn,
                txn,
                "deterministic",
                ReasonCode.OVERPAYMENT_REQUIRES_REVIEW,
                "The exact student ID is known, but credit creation requires a bursar decision.",
                candidate_items=exact_candidates,
                allocations=original_allocations,
            )
        if config.auto_post_enabled:
            proposed = []
            for line in p.lines:
                events, _ = distribute.distribute(
                    conn,
                    txn_id,
                    line.student_id,
                    line.amount_minor,
                    "engine",
                    create_credit=False,
                )
                proposed.extend(events)
            proposal_id = f"PROP-{uuid4().hex}"
            repo.supersede_pending_proposals(conn, txn_id)
            repo.insert_proposal(
                conn,
                proposal_id,
                txn_id,
                "deterministic",
                RecommendedAction.AUTO,
                1.0,
                p.explanation,
                _now(),
                candidates=_candidate_snapshot(exact_candidates),
                evidence=_evidence_snapshot(txn),
                allocations=original_allocations,
            )
            try:
                with dbmod.transaction(conn):
                    ledger.post_within_transaction(
                        conn,
                        txn_id,
                        proposed,
                        "engine",
                        "deterministic",
                        txn_id,
                        "auto",
                    )
                    repo.record_proposal_decision(
                        conn, proposal_id, "approve", "engine", _now(),
                        original_allocations, 0,
                    )
                    repo.set_routing_state(conn, txn_id, "auto")
                return "auto"
            except InvariantViolation as exc:
                conn.execute(
                    "UPDATE proposals SET status='superseded', failure_reason=? "
                    "WHERE proposal_id=?",
                    (",".join(exc.violations), proposal_id),
                )
                return _store_review(
                    conn,
                    txn,
                    "deterministic",
                    "invariant",
                    "Automatic posting was blocked by the constraint engine.",
                    candidate_items=exact_candidates,
                    allocations=original_allocations,
                )
        return _store_review(
            conn,
            txn,
            "deterministic",
            "flag_off",
            "Automatic posting is disabled.",
            candidate_items=exact_candidates,
            allocations=original_allocations,
        )

    # Non-exact -> model path. Candidate-bearing transactions must remain reviewable even when
    # the local model is not installed or temporarily unavailable.
    if backend is None:
        available_candidates = candidates.generate(conn, txn)
        if not available_candidates:
            repo.set_routing_state(conn, txn_id, "unmatched")
            return "unmatched"
        return _store_review(
            conn,
            txn,
            "llm",
            ReasonCode.INFERENCE_UNAVAILABLE,
            "Local inference is unavailable; inspect the deterministic evidence.",
            candidate_items=available_candidates,
        )

    r = run_model_path(conn, txn, backend, tokenizer_path)
    if r.failure == "no_candidates":
        repo.set_routing_state(conn, txn_id, "unmatched")
        return "unmatched"
    if r.failure == "prompt_budget":
        return _store_review(
            conn, txn, "llm", ReasonCode.PROMPT_BUDGET_EXCEEDED, "Prompt budget exceeded.",
            candidate_items=r.surviving,
        )
    if r.failure == "transport":
        return _store_review(
            conn, txn, "llm", ReasonCode.INFERENCE_UNAVAILABLE, "Local inference backend is down.",
            candidate_items=r.surviving,
        )
    if r.failure == "schema":   # content-invalid -> review immediately (NO retry; temp 0 is deterministic)
        return _store_review(
            conn, txn, "llm", ReasonCode.SCHEMA_INVALID, r.schema_reason,
            candidate_items=r.surviving, raw_output=r.raw_output,
        )

    feats = features.extract(txn, r.surviving, r.data, r.dry_ok, r.chosen_id, budget_shed=r.budget_shed)
    policy = ModelConfidencePolicy()
    policy.route(feats)   # v1 -> review
    return _store_review(
        conn,
        txn,
        "llm",
        ReasonCode.MODEL_RANKED,
        r.data.get("explanation", ""),
        candidate_items=r.surviving,
        raw_output=r.raw_output,
        model_data=r.data,
        feature_values=feats,
        confidence=policy.score(feats),
    )
