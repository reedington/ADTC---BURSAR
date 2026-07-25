import pytest
from bursa.db import connect, init_db, transaction


@pytest.fixture
def db(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def seeded_term_student_fee(db):
    with transaction(db):
        db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
        db.execute("INSERT INTO students VALUES ('STU-1','Chi Okafor','chi okafor','JSS1','T1')")
        db.execute("INSERT INTO fee_items VALUES ('FEE-TUITION','Tuition','T1',10)")
        db.execute("INSERT INTO charges VALUES ('CHG-1','STU-1','FEE-TUITION','T1')")
    return "CHG-1"


@pytest.fixture
def seeded_ledger_event(db, seeded_term_student_fee):
    cur = db.execute(
        "INSERT INTO ledger_events "
        "(event_type, charge_id, student_id, fee_id, amount_minor, actor, source, "
        " evidence_ref, decision_path, created_at) "
        "VALUES ('charge_created','CHG-1','STU-1','FEE-TUITION',5000000,'importer',"
        "'fees_csv','BATCH-1','import','2026-01-01T00:00:00+00:00')")
    return cur.lastrowid
