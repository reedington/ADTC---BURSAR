import pytest
from bursa.money import parse_naira, format_naira, MINOR_UNITS_PER_NAIRA


@pytest.mark.parametrize("text,minor", [
    ("75000", 7_500_000),
    ("75,000", 7_500_000),
    ("₦75,000.00", 7_500_000),
    ("75000.5", 7_500_050),
    ("0", 0),
    ("  1,234.99 ", 123_499),
])
def test_parse_naira_ok(text, minor):
    assert parse_naira(text) == minor


@pytest.mark.parametrize("bad", ["", "abc", "1.234", "-5", "1e3", "12,34,5"])
def test_parse_naira_rejects(bad):
    with pytest.raises(ValueError):
        parse_naira(bad)


def test_format_naira():
    assert format_naira(7_500_000) == "₦75,000.00"
    assert format_naira(0) == "₦0.00"
    assert format_naira(123_499) == "₦1,234.99"


def test_round_trip():
    assert format_naira(parse_naira("₦1,234.99")) == "₦1,234.99"


def test_no_floats_stored():
    assert isinstance(parse_naira("75000.50"), int)
