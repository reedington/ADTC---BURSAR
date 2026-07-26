import json
from bursa import repository as repo
from bursa.inference.backend import FakeBackend
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.goldcheck import load_case
from bursa_eval.models import naira_to_minor


def _model_json(txn_id, student_id, amount_minor):
    return json.dumps({
        "transaction_id": txn_id,
        "recommended_action": "review",
        "candidate_allocations": [
            {"student_id": student_id, "amount_minor": amount_minor, "reason_codes": ["MODEL_RANKED"]}
        ],
        "explanation": "matched by nickname",
    })


def test_run_model_path_returns_raw_data_and_events():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    conn = loader.materialize(case)
    txn_id = loader.insert_case_transaction(conn, case)
    txn = repo.get_transaction(conn, txn_id)
    student = case.expected.allocations[0].student_id
    amount = naira_to_minor(case.expected.allocations[0].amount_naira)
    backend = FakeBackend(response=_model_json(txn_id, student, amount))

    r = run_model_path(conn, txn, backend)

    assert r.failure is None
    assert r.data["candidate_allocations"][0]["student_id"] == student
    assert r.chosen_id == student
    assert [c.student_id for c in r.surviving]  # pool built
    assert r.model_events  # distributed charge-grain events recovered
    conn.close()


def test_run_model_path_no_candidates_is_flagged():
    case = load_case("data/gold/gold-0002-nickname-en.yaml")
    conn = loader.materialize(case)
    txn_id = loader.insert_case_transaction(conn, case)
    # Wipe candidate sources so generate() returns [] (mutate BEFORE fetching txn).
    # Student name and txn narration/payer must be DISJOINT, and the guardian phone
    # suffix nulled so the reference digits can't seed a phone-suffix candidate.
    conn.execute("DELETE FROM student_aliases")
    conn.execute("UPDATE students SET name = 'zzzzzz', normalized_name = 'zzzzzz'")
    conn.execute("UPDATE guardians SET phone_suffix = NULL")
    # Amount must not equal any outstanding charge balance (kills the exact_balance signal).
    conn.execute("UPDATE transactions SET narration = 'qqqqqq', payer_name = 'qqqqqq', "
                 "amount_minor = 777 WHERE transaction_id = ?", (txn_id,))
    txn = repo.get_transaction(conn, txn_id)
    r = run_model_path(conn, txn, FakeBackend(response="{}"))
    assert r.failure == "no_candidates"
    conn.close()
