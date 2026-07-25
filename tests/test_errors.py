from bursa.errors import ImportRowError, InvariantViolation


def test_import_row_error_carries_context():
    e = ImportRowError(row_number=4, field="amount", reason="not a number")
    assert e.row_number == 4
    assert e.field == "amount"
    assert "not a number" in str(e)


def test_invariant_violation_lists_codes():
    e = InvariantViolation(["INV-01", "INV-05"])
    assert e.violations == ["INV-01", "INV-05"]
    assert "INV-01" in str(e)
