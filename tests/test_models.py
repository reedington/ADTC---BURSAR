import pytest
from pydantic import ValidationError
from bursa.models import (EventType, RoutingState, RecommendedAction,
                          CanonicalTransaction, ProposalLine, LedgerEventInput)


def test_enums_are_strings():
    assert EventType.ALLOCATION == "allocation"
    assert RoutingState.REVIEW == "review"
    assert RecommendedAction.AUTO == "auto"


def test_transaction_amount_must_be_int():
    with pytest.raises(ValidationError):
        CanonicalTransaction(transaction_id="TXN-1", source="bank_csv",
                             posted_at="2026-02-14T09:32:00+01:00",
                             amount_minor=1000.5, direction="credit",
                             dedup_hash="h1")


def test_proposal_line_defaults_reason_codes():
    line = ProposalLine(student_id="STU-1", amount_minor=5000)
    assert line.reason_codes == []


def test_ledger_event_input_requires_provenance_fields():
    with pytest.raises(ValidationError):
        LedgerEventInput(event_type=EventType.ALLOCATION, amount_minor=100)
