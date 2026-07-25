from bursa.features import extract, FEATURES_VERSION
from bursa.candidates import Candidate


def _cands():
    return [Candidate(student_id="STU-1", name="Chidi", aliases=["Chi"],
                      outstanding=[("CHG", 5_000_000)], is_prior_payer=True, score=10),
            Candidate(student_id="STU-2", name="Bola", score=4)]


def test_extract_shape_and_version(make_row):
    # real sqlite3.Row, not a dict — guards the txn field-accessor contract
    txn = make_row(transaction_id="TXN-1", payer_name="Chidi", narration="chi",
                   amount_minor=5_000_000)
    feats = extract(txn, _cands(), {"candidate_allocations": [{"student_id": "STU-1",
                    "amount_minor": 5_000_000}]}, dry_ok=True, chosen_id="STU-1")
    assert feats["features_version"] == FEATURES_VERSION
    assert feats["constraint_validation_result"] == 1
    assert feats["llm_ranking_consistency"] == 1          # chose the top candidate
    assert 0.0 <= feats["name_alias_similarity"] <= 1.0
    assert feats["candidate_count"] == 2
    assert feats["amount_to_balance_agreement"] == 1.0    # exact balance match


def test_dry_run_failure_recorded(make_row):
    txn = make_row(transaction_id="TXN-2", payer_name="x", narration="y", amount_minor=1)
    feats = extract(txn, _cands(), {"candidate_allocations": []}, dry_ok=False, chosen_id=None)
    assert feats["constraint_validation_result"] == 0
