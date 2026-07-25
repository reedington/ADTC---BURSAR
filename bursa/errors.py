class ImportRowError(Exception):
    """A single CSV row failed validation; the rest of the file continues."""

    def __init__(self, row_number: int, field: str, reason: str):
        self.row_number = row_number
        self.field = field
        self.reason = reason
        super().__init__(f"row {row_number}: {field}: {reason}")


class InvariantViolation(Exception):
    """One or more ledger invariants would be broken by a proposed posting."""

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))
