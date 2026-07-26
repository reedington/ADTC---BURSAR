import sqlite3
import time
from dataclasses import dataclass, field
from bursa import matcher, distribute, repository as repo
from bursa.models import RecommendedAction
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.models import GoldCase


@dataclass
class CaseRecord:
    suite: str
    case_id: str
    family: str
    language: str
    difficulty: str | None
    would_auto_post: bool
    correct_action: bool | None
    top1_hit: bool | None
    exact_alloc_hit: bool | None
    model_abstains: bool | None
    abstention_hit: bool | None
    true_abstention: bool
    pool_recall_hit: bool | None
    valid_json: bool | None
    unsupported_id: bool | None
    dry_run_ok: bool | None
    dup_blocked: bool | None
    timings: dict = field(default_factory=dict)


def _events_set(events):
    return {(e.student_id, e.charge_id, e.amount_minor) for e in events}


def _base(case: GoldCase, **kw) -> dict:
    d = dict(suite="bursa_gold", case_id=case.id, family=case.scenario_family,
             language=case.language, difficulty=case.difficulty,
             would_auto_post=False, correct_action=None, top1_hit=None,
             exact_alloc_hit=None, model_abstains=None, abstention_hit=None,
             true_abstention=case.is_abstention(), pool_recall_hit=None,
             valid_json=None, unsupported_id=None, dry_run_ok=None, dup_blocked=None,
             timings={})
    d.update(kw)
    return d


def _duplicate_is_blocked(conn, case: GoldCase) -> bool:
    """True if the import/dedup layer refuses the case transaction (money-doubling block)."""
    try:
        loader.insert_case_transaction(conn, case)
        return False   # insert succeeded => NOT blocked => a leak
    except sqlite3.IntegrityError:
        return True     # dedup / append-only refused the duplicate


def evaluate_case(case: GoldCase, backend, tokenizer_path=None) -> CaseRecord:
    t0 = time.perf_counter()
    conn = loader.materialize(case)
    try:
        if case.expected.outcome == "duplicate_blocked":
            blocked = _duplicate_is_blocked(conn, case)
            return CaseRecord(**_base(case, dup_blocked=blocked, correct_action=blocked,
                                      timings={"total": time.perf_counter() - t0}))

        txn_id = loader.insert_case_transaction(conn, case)
        txn = repo.get_transaction(conn, txn_id)
        expected_events = _events_set(loader.build_expected_events(conn, case, txn_id))
        primary = case.expected.allocations[0].student_id if case.expected.allocations else None

        p = matcher.match(conn, txn)
        if p.recommended_action == RecommendedAction.AUTO:
            model_events = []
            for line in p.lines:
                evs, _ = distribute.distribute(conn, txn_id, line.student_id, line.amount_minor, "engine")
                model_events.extend(evs)
            hit = _events_set(model_events) == expected_events
            top1 = (p.lines[0].student_id == primary) if primary else None
            return CaseRecord(**_base(case, would_auto_post=True, exact_alloc_hit=hit,
                                      correct_action=(case.expected.outcome == "auto"), top1_hit=top1,
                                      timings={"total": time.perf_counter() - t0}))

        # Non-exact -> model path (raw output scored, never posted).
        r = run_model_path(conn, txn, backend, tokenizer_path)
        pool_ids = [c.student_id for c in r.surviving]
        truth = case.pool_truth()
        pool_recall = (set(truth) <= set(pool_ids)) if truth else None

        if r.failure is not None:
            model_abstains = (r.failure == "no_candidates")
            valid_json = False if r.failure == "schema" else None
            unsupported = (r.failure == "schema" and r.schema_reason == "unknown_student_id")
            return CaseRecord(**_base(
                case, pool_recall_hit=pool_recall, valid_json=valid_json, unsupported_id=unsupported,
                model_abstains=model_abstains, abstention_hit=(model_abstains == case.is_abstention()),
                dry_run_ok=r.dry_ok, timings={"total": time.perf_counter() - t0}))

        data = r.data
        action = data.get("recommended_action")
        allocs = data.get("candidate_allocations", [])
        model_abstains = (not allocs) or (action == "unmatched")
        top1 = (r.chosen_id == primary) if primary else (r.chosen_id is None)
        exact = _events_set(r.model_events) == expected_events
        return CaseRecord(**_base(
            case, valid_json=True, unsupported_id=False, dry_run_ok=r.dry_ok,
            model_abstains=model_abstains, abstention_hit=(model_abstains == case.is_abstention()),
            correct_action=(action == case.expected.outcome), top1_hit=top1,
            exact_alloc_hit=exact, pool_recall_hit=pool_recall,
            timings={"total": time.perf_counter() - t0}))
    finally:
        conn.close()


def run_gold_suite(cases, backend, tokenizer_path=None):
    return [evaluate_case(c, backend, tokenizer_path) for c in cases]
