from decimal import Decimal, InvalidOperation

MINOR_UNITS_PER_NAIRA = 100


def parse_naira(value: str) -> int:
    """Parse a naira string to integer minor units (kobo).

    Decimal is used ONLY at this boundary; the result is always an int.
    Rejects empty, non-numeric, negative, exponent, or >2-decimal input.
    """
    if not isinstance(value, str):
        raise ValueError("money must be parsed from a string")
    cleaned = value.strip().replace("₦", "").replace(",", "").strip()
    if cleaned == "" or "e" in cleaned.lower():
        raise ValueError(f"invalid money value: {value!r}")
    if "." in cleaned and len(cleaned.split(".")[1]) > 2:
        raise ValueError(f"too many decimal places: {value!r}")
    # Reject grouping mistakes like "12,34,5".
    if "," in value:
        whole = value.strip().replace("₦", "").split(".")[0]
        groups = whole.split(",")
        if len(groups) > 1 and (len(groups[0]) == 0 or len(groups[0]) > 3
                                or any(len(g) != 3 for g in groups[1:])):
            raise ValueError(f"invalid thousands grouping: {value!r}")
    try:
        dec = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"invalid money value: {value!r}")
    if dec < 0:
        raise ValueError(f"money cannot be negative: {value!r}")
    minor = (dec * MINOR_UNITS_PER_NAIRA).to_integral_value()
    return int(minor)


def format_naira(minor: int) -> str:
    """Format integer minor units as a naira display string."""
    if not isinstance(minor, int):
        raise ValueError("format_naira requires an int (minor units)")
    naira, kobo = divmod(abs(minor), MINOR_UNITS_PER_NAIRA)
    sign = "-" if minor < 0 else ""
    return f"{sign}₦{naira:,}.{kobo:02d}"


def format_naira_input(minor: int) -> str:
    """Return a plain decimal string for an editable NGN field without using float."""
    if not isinstance(minor, int):
        raise ValueError("format_naira_input requires an int (minor units)")
    naira, kobo = divmod(abs(minor), MINOR_UNITS_PER_NAIRA)
    sign = "-" if minor < 0 else ""
    return f"{sign}{naira}.{kobo:02d}"
