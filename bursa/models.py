from enum import StrEnum
from pydantic import BaseModel, Field, StrictInt


class EventType(StrEnum):
    CHARGE_CREATED = "charge_created"
    ALLOCATION = "allocation"
    REVERSAL = "reversal"
    CREDIT_GRANT = "credit_grant"
    CREDIT_APPLICATION = "credit_application"


class RoutingState(StrEnum):
    NEW = "new"
    AUTO = "auto"
    REVIEW = "review"
    UNMATCHED = "unmatched"


class RecommendedAction(StrEnum):
    AUTO = "auto"
    REVIEW = "review"
    UNMATCHED = "unmatched"


class CanonicalTransaction(BaseModel):
    transaction_id: str
    source: str
    reference: str | None = None
    raw_reference: str | None = None
    posted_at: str
    payer_name: str | None = None
    narration: str | None = None
    amount_minor: StrictInt
    direction: str  # "credit" | "debit"
    dedup_hash: str
    batch_id: str | None = None


class ProposalLine(BaseModel):
    student_id: str
    amount_minor: StrictInt
    reason_codes: list[str] = Field(default_factory=list)


class Proposal(BaseModel):
    transaction_id: str
    source: str  # "deterministic" | "llm"
    lines: list[ProposalLine] = Field(default_factory=list)
    recommended_action: RecommendedAction
    confidence: float | None = None
    explanation: str = ""


class LedgerEventInput(BaseModel):
    event_type: EventType
    amount_minor: StrictInt
    actor: str
    source: str
    evidence_ref: str
    decision_path: str
    transaction_id: str | None = None
    charge_id: str | None = None
    student_id: str | None = None
    fee_id: str | None = None
    holder: str | None = None
    reverses_event_id: int | None = None
