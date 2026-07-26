import json
from bursa import pipeline, ledger, repository as repo
from bursa.inference.backend import FakeBackend, BackendTransportError
from bursa.models import CanonicalTransaction


def _setup(db):
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, narration, amount=5_000_000, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, payer_name="Ada", dedup_hash=dedup))
    return tid


def _model_json(txn_id, sid):
    return json.dumps({"transaction_id": txn_id, "interpretation": {},
        "candidate_allocations": [{"student_id": sid, "amount_minor": 5_000_000,
                                   "reason_codes": []}],
        "recommended_action": "review", "explanation": "match", "ambiguities": []})


class _CountingBackend:
    def __init__(self, resp):
        self.calls = 0
        self.resp = resp

    def generate(self, p, g, n):
        self.calls += 1
        return self.resp


def test_model_path_routes_review_and_stores_proposal(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "chi fees")                       # fuzzy/alias -> candidate, not exact
    backend = FakeBackend(response=_model_json(tid, "STU-1"))
    assert pipeline.reconcile(db, tid, backend=backend) == "review"
    p = db.execute("SELECT source, features FROM proposals WHERE transaction_id=?", (tid,)).fetchone()
    assert p["source"] == "llm"
    assert json.loads(p["features"])["features_version"] == 1
    snapshot = db.execute(
        "SELECT candidate_snapshot_json, raw_output_json FROM proposals "
        "WHERE transaction_id=?", (tid,)
    ).fetchone()
    assert json.loads(snapshot["candidate_snapshot_json"])[0]["student_id"] == "STU-1"
    assert json.loads(snapshot["raw_output_json"])["transaction_id"] == tid
    assert db.execute(
        "SELECT COUNT(*) FROM proposal_allocations WHERE proposal_id="
        "(SELECT proposal_id FROM proposals WHERE transaction_id=?)", (tid,)
    ).fetchone()[0] == 1
    assert repo.live_events(db, transaction_id=tid) == []   # nothing posted (v1)


def test_transport_failure_routes_review(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "chi fees", tid="TXN-2", dedup="h2")
    backend = FakeBackend(raises=BackendTransportError("down"))
    assert pipeline.reconcile(db, tid, backend=backend) == "review"


def test_content_invalid_is_not_retried(db, seeded_term_student_fee):
    _setup(db)
    tid = _txn(db, "chi fees", tid="TXN-4", dedup="h4")
    backend = _CountingBackend('{"transaction_id":"WRONG"}')   # content-invalid (mismatch)
    assert pipeline.reconcile(db, tid, backend=backend) == "review"
    assert backend.calls == 1                                   # NO retry on content-invalid


def test_no_candidate_unmatched(db, seeded_term_student_fee):
    _setup(db)
    # amount matches no outstanding balance, and no name/alias/guardian -> genuinely no candidates
    tid = _txn(db, "zzz", amount=1234, tid="TXN-3", dedup="h3")
    assert pipeline.reconcile(db, tid, backend=FakeBackend(response="{}")) == "unmatched"
