import json
from copy import deepcopy
from bursa.inference.backend import FakeBackend
from bursa_eval.harness.runner import evaluate_case
from bursa_eval.goldcheck import load_case
from bursa_eval.models import naira_to_minor, HistoryEntry


def _keyed_fake(mapping):
    """FakeBackend whose response depends on the txn_id embedded in the prompt."""
    def respond(raw_prompt, grammar, n_predict):
        for txn_id, payload in mapping.items():
            if txn_id in raw_prompt:
                return json.dumps(payload)
        return "{}"
    return FakeBackend(response=respond)


def test_exact_case_records_auto_and_correct_allocation():
    case = load_case("data/gold/gold-0001-exact-id-en.yaml")
    rec = evaluate_case(case, FakeBackend(response="{}"))
    assert rec.would_auto_post is True
    assert rec.exact_alloc_hit is True          # deterministic allocation matches expected
    assert rec.pool_recall_hit is None          # exact path builds no pool
    assert rec.valid_json is None


def test_model_case_scores_top1_and_pool_recall():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    student = case.expected.allocations[0].student_id
    amount = case.expected.allocations[0].amount_naira
    txn_id = f"TXN-{case.transaction.reference}"
    payload = {"transaction_id": txn_id,
               "recommended_action": "review",
               "candidate_allocations": [
                   {"student_id": student, "amount_minor": naira_to_minor(amount),
                    "reason_codes": ["MODEL_RANKED"]}],
               "explanation": "nickname"}
    rec = evaluate_case(case, _keyed_fake({txn_id: payload}))
    assert rec.would_auto_post is False
    assert rec.top1_hit is True
    assert rec.exact_alloc_hit is True
    assert rec.pool_recall_hit is True
    assert rec.valid_json is True
    assert rec.model_abstains is False
    assert rec.correct_action is True           # model action "review" == expected outcome "review"


def test_duplicate_blocked_evaluated_at_import_layer():
    case = deepcopy(load_case("data/gold/gold-0002-nickname-en.yaml"))
    # Forge a duplicate_blocked scenario: put the case reference into history, expect a block.
    case.expected.outcome = "duplicate_blocked"
    case.setup.history.append(HistoryEntry(transaction=case.transaction, allocations=[]))
    rec = evaluate_case(case, FakeBackend(response="{}"))
    assert rec.dup_blocked is True              # import layer refused the double-post
    assert rec.valid_json is None               # model never ran
