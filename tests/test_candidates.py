from bursa import candidates, ledger
from bursa.models import CanonicalTransaction
from bursa import repository as repo


def _mk(db):
    db.execute("INSERT INTO students VALUES ('STU-2','Bola Bello','bola bello','JSS1','T1')")
    db.execute("INSERT INTO guardians VALUES ('G1','Ada Okafor','ada okafor','1234')")
    db.execute("INSERT INTO student_guardians VALUES ('STU-1','G1')")
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    ledger.create_charge(db, "CHG-1", "STU-1", "FEE-TUITION", "T1", 5_000_000, "i", "f", "B")


def _txn(db, narration, payer, amount=5_000_000, tid="TXN-1", dedup="h1"):
    repo.insert_transaction(db, CanonicalTransaction(transaction_id=tid, source="bank_csv",
        posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount, direction="credit",
        narration=narration, payer_name=payer, dedup_hash=dedup))
    return repo.get_transaction(db, tid)


def test_alias_pools_candidate(db, seeded_term_student_fee):
    _mk(db)
    txn = _txn(db, "CHI SCH FEE", "Ada Okafor")
    ids = [c.student_id for c in candidates.generate(db, txn)]
    assert "STU-1" in ids


def test_fuzzy_misspelling_pools(db, seeded_term_student_fee):
    _mk(db)
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chidi','chidi')")
    txn = _txn(db, "chidy fees", "someone", tid="TXN-2", dedup="h2")
    ids = [c.student_id for c in candidates.generate(db, txn)]
    assert "STU-1" in ids


def test_capped_at_five_deterministic(db, seeded_term_student_fee):
    _mk(db)
    for i in range(8):
        db.execute(f"INSERT INTO students VALUES ('STU-A{i}','Chi Test{i}','chi test{i}','JSS1','T1')")
    txn = _txn(db, "chi", "x")
    out = candidates.generate(db, txn)
    assert len(out) <= candidates.MAX_CANDIDATES
    # deterministic: a second call yields the identical ordering
    out2 = candidates.generate(db, txn)
    assert [c.student_id for c in out] == [c.student_id for c in out2]
    # ordering is (score DESC, student_id ASC)
    keys = [(-c.score, c.student_id) for c in out]
    assert keys == sorted(keys)


def test_sibling_rides_in(db, seeded_term_student_fee):
    # STU-1 and STU-3 share guardian G1; narration names only STU-1's alias
    db.execute("INSERT INTO guardians VALUES ('G1','Ada','ada','1')")
    db.execute("INSERT INTO students VALUES ('STU-3','Uche','uche','JSS1','T1')")
    db.execute("INSERT INTO student_guardians VALUES ('STU-1','G1')")
    db.execute("INSERT INTO student_guardians VALUES ('STU-3','G1')")
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    txn = _txn(db, "chi", "x", tid="TXN-5", dedup="h5")
    ids = [c.student_id for c in candidates.generate(db, txn)]
    assert "STU-1" in ids and "STU-3" in ids
