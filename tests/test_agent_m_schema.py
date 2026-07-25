from bursa.reasoncodes import ReasonCode


def test_new_reason_codes_exist():
    assert ReasonCode.PROMPT_BUDGET_EXCEEDED == "PROMPT_BUDGET_EXCEEDED"
    assert ReasonCode.INFERENCE_UNAVAILABLE == "INFERENCE_UNAVAILABLE"
    assert ReasonCode.SCHEMA_INVALID == "SCHEMA_INVALID"


def test_student_aliases_table(db, seeded_term_student_fee):
    db.execute("INSERT INTO student_aliases VALUES ('STU-1','Chi','chi')")
    row = db.execute("SELECT normalized_alias FROM student_aliases WHERE student_id='STU-1'").fetchone()
    assert row["normalized_alias"] == "chi"


def test_proposals_has_features_column(db):
    cols = [r["name"] for r in db.execute("PRAGMA table_info(proposals)").fetchall()]
    assert "features" in cols
