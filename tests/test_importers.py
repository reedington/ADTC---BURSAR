from bursa.importers import students, fees
from bursa import projections as proj


def test_import_students_reports_row_errors(db):
    db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
    rows = [
        {"student_id": "STU-1", "name": "Chi Okafor", "class": "JSS1", "term_id": "T1"},
        {"student_id": "", "name": "No ID", "class": "JSS1", "term_id": "T1"},
    ]
    res = students.import_students(db, rows, "students.csv")
    assert res["accepted"] == 1 and res["rejected"] == 1
    assert res["errors"][0].field == "student_id"


def test_import_fees_creates_charge_and_billing_event_atomically(db):
    db.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
    db.execute("INSERT INTO students VALUES ('STU-1','Chi','chi','JSS1','T1')")
    rows = [{"fee_id": "FEE-TUITION", "fee_name": "Tuition", "priority": "10",
             "student_id": "STU-1", "term_id": "T1", "amount": "50,000"}]
    res = fees.import_fees(db, rows, "fees.csv")
    assert res["accepted"] == 1
    charge = db.execute("SELECT charge_id FROM charges WHERE student_id='STU-1'").fetchone()
    assert charge is not None
    assert proj.charge_billed(db, charge["charge_id"]) == 5_000_000


def test_no_charge_without_billing_event(db):
    orphans = db.execute(
        "SELECT charge_id FROM charges c WHERE NOT EXISTS "
        "(SELECT 1 FROM ledger_events e WHERE e.charge_id = c.charge_id "
        " AND e.event_type='charge_created')").fetchall()
    assert orphans == []
