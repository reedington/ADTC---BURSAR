from bursa_eval.harness.runner import CaseRecord
from bursa_eval.harness.metrics import compute_metrics, evaluate_gates


def _rec(**kw):
    base = dict(suite="bursa_gold", case_id="c", family="name_match", language="en",
                difficulty="easy", would_auto_post=False, correct_action=None, top1_hit=None,
                exact_alloc_hit=None, model_abstains=None, abstention_hit=None,
                true_abstention=False, pool_recall_hit=None, valid_json=None,
                unsupported_id=None, dry_run_ok=None, dup_blocked=None, timings={})
    base.update(kw)
    return CaseRecord(**base)


def test_incorrect_auto_post_trips_gate():
    recs = [_rec(would_auto_post=True, exact_alloc_hit=False)]   # auto-posted WRONG money
    m = compute_metrics(recs)
    assert m["incorrect_auto_posts"] == 1
    assert "incorrect_auto_posts" in evaluate_gates(m)           # -> non-zero exit


def test_unblocked_duplicate_trips_gate():
    recs = [_rec(dup_blocked=True), _rec(dup_blocked=False)]     # one leak
    m = compute_metrics(recs)
    assert m["duplicate_blocked_rate"] < 1.0
    assert "duplicate_blocked_rate" in evaluate_gates(m)         # -> non-zero exit


def test_all_gates_pass_on_clean_run():
    recs = [_rec(would_auto_post=True, exact_alloc_hit=True), _rec(dup_blocked=True)]
    assert evaluate_gates(compute_metrics(recs)) == []


def test_review_with_allocations_is_not_an_abstention():
    # Sibling split routed to review, WITH allocations -> must NOT count as a model abstention.
    rec = _rec(model_abstains=False, abstention_hit=True, true_abstention=False,
               top1_hit=True, exact_alloc_hit=True, correct_action=True)
    m = compute_metrics([rec])
    # No true-abstentions present, so recall is undefined; precision has no abstainers.
    assert m["abstention_recall"] is None
    assert m["abstention_precision"] is None
    assert m["action_accuracy"] == 1.0
    assert rec.model_abstains is False   # the record itself never mislabels it


def test_action_accuracy_and_top1_aggregate():
    recs = [_rec(correct_action=True, top1_hit=True), _rec(correct_action=False, top1_hit=False)]
    m = compute_metrics(recs)
    assert m["action_accuracy"] == 0.5
    assert m["top1_student_accuracy"] == 0.5
