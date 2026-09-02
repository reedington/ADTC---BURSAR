from typing import Annotated, Literal
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from bursa import money

# The 13 required reconciliation scenario families (MODEL_ARCHITECTURE §9.2), plus the
# injection family (malicious/irrelevant narration) that the plan's Prime Directive 6 requires.
SCENARIO_FAMILIES = {
    "name_match", "guardian_surname_differs", "nickname_initials", "sibling_split",
    "instalment", "underpayment", "overpayment", "fee_item_split", "duplicate_reference",
    "known_payer", "ambiguous_candidates", "no_candidate", "ocr_substitution", "injection",
}
LANGUAGES = {"en", "pcm", "yo", "ha", "ig"}


def _reject_float(v):
    # kobo safety (INV-03 extended to the data): an IEEE float must never carry money.
    if isinstance(v, float):
        raise ValueError("amount must be an integer or quoted string, not a float")
    return v


NairaAmount = Annotated[int | str, BeforeValidator(_reject_float)]


def naira_to_minor(v) -> int:
    """Convert an authored naira amount (int or str) to integer minor units via money.py."""
    return money.parse_naira(str(v))


class ChargeSpec(BaseModel):
    fee_id: str
    amount_naira: NairaAmount


class GuardianSpec(BaseModel):
    id: str
    name: str
    phone_suffix: str | None = None


class StudentSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    name: str
    student_class: str | None = Field(None, alias="class")
    aliases: list[str] = []
    guardians: list[str] = []
    charges: list[ChargeSpec] = []


class TransactionSpec(BaseModel):
    reference: str | None = None
    date: str
    amount_naira: NairaAmount
    payer_name: str | None = None
    narration: str | None = None


class Allocation(BaseModel):
    student_id: str
    fee_id: str | None = None
    amount_naira: NairaAmount


class Credit(BaseModel):
    holder: str
    amount_naira: NairaAmount


class HistoryEntry(BaseModel):
    transaction: TransactionSpec
    allocations: list[Allocation] = []


class TermSpec(BaseModel):
    id: str
    session: str
    name: str


class Setup(BaseModel):
    term: TermSpec
    guardians: list[GuardianSpec] = []
    students: list[StudentSpec] = []
    history: list[HistoryEntry] = []


class Expected(BaseModel):
    outcome: Literal["auto", "review", "unmatched", "duplicate_blocked"]
    allocations: list[Allocation] = []
    credits: list[Credit] = []
    pool_must_include: list[str] | None = None
    rationale: str = ""


class ReviewMetadata(BaseModel):
    status: Literal["draft", "reviewed"]
    reviewer: str | None = None
    reviewed_at: str | None = None
    content_sha256: str | None = None


class GoldCase(BaseModel):
    id: str
    scenario_family: str
    language: str
    difficulty: Literal["easy", "medium", "hard"] | None = None
    guardian_family: str
    template_family: str
    provenance: Literal["team_authored", "synthetic", "draft"] = "team_authored"
    review: ReviewMetadata | None = None
    setup: Setup
    transaction: TransactionSpec
    expected: Expected

    @field_validator("scenario_family")
    @classmethod
    def _known_family(cls, v):
        if v not in SCENARIO_FAMILIES:
            raise ValueError(f"unknown scenario_family '{v}'; one of {sorted(SCENARIO_FAMILIES)}")
        return v

    @field_validator("language")
    @classmethod
    def _known_language(cls, v):
        if v not in LANGUAGES:
            raise ValueError(f"unknown language '{v}'; one of {sorted(LANGUAGES)}")
        return v

    def pool_truth(self) -> list[str]:
        """Students the candidate generator must surface (pool-recall truth)."""
        if self.expected.pool_must_include is not None:
            return self.expected.pool_must_include
        return [a.student_id for a in self.expected.allocations]

    def is_abstention(self) -> bool:
        return len(self.expected.allocations) == 0
