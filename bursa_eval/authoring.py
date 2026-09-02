"""Draft and human-review workflow for the expanded authored gold set."""
from __future__ import annotations

import copy
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bursa_eval.goldcheck import check_case, load_case
from bursa_eval.models import GoldCase, SCENARIO_FAMILIES
from bursa_eval.repro import canonical_sha256


LEGACY_CASE_COUNT = 14
LEGACY_CASE_IDS = {
    f"gold-{index:04d}-{suffix}"
    for index, suffix in (
        (1, "exact-id"),
        (2, "nickname"),
        (3, "sibling-split"),
        (4, "ambiguous-abstain"),
        (5, "guardian-surname"),
        (6, "instalment"),
        (7, "underpayment"),
        (8, "overpayment"),
        (9, "fee-item-split"),
        (10, "duplicate-reference"),
        (11, "known-payer"),
        (12, "no-candidate"),
        (13, "ocr-substitution"),
        (14, "injection"),
    )
}
TARGET_CASE_COUNT = 154
TARGET_LANGUAGES = {"en": 90, "pcm": 40, "yo": 24}
TARGET_ABSTENTIONS = 42
TARGET_PER_FAMILY = 11
_DRAFT_MARKER = "REVIEW_DRAFT"


def review_content(case_or_data: GoldCase | dict) -> dict:
    data = (
        case_or_data.model_dump(mode="json", by_alias=True)
        if isinstance(case_or_data, GoldCase)
        else copy.deepcopy(case_or_data)
    )
    data.pop("review", None)
    return data


def review_checksum(case_or_data: GoldCase | dict) -> str:
    return canonical_sha256(review_content(case_or_data))


def _replace_ids(data: dict, family_index: int, variant: int) -> None:
    prefix = f"D{family_index:02d}{variant:02d}"
    original_reference = data["transaction"].get("reference")
    new_reference = f"NIP-D-{prefix}"
    student_map = {
        student["id"]: f"STU-{prefix}{index:02d}"
        for index, student in enumerate(data["setup"].get("students", []), 1)
    }
    guardian_map = {
        guardian["id"]: f"G-{prefix}{index:02d}"
        for index, guardian in enumerate(data["setup"].get("guardians", []), 1)
    }
    for student in data["setup"].get("students", []):
        student["id"] = student_map[student["id"]]
        student["guardians"] = [
            guardian_map.get(value, value) for value in student.get("guardians", [])
        ]
    for guardian in data["setup"].get("guardians", []):
        guardian["id"] = guardian_map[guardian["id"]]
    for history_index, history in enumerate(data["setup"].get("history", []), 1):
        history_reference = history["transaction"].get("reference")
        history["transaction"]["reference"] = (
            new_reference
            if history_reference and history_reference == original_reference
            else f"NIP-H-{prefix}-{history_index:02d}"
        )
        for allocation in history.get("allocations", []):
            allocation["student_id"] = student_map[allocation["student_id"]]
    for allocation in data["expected"].get("allocations", []):
        allocation["student_id"] = student_map[allocation["student_id"]]
    for credit in data["expected"].get("credits", []):
        credit["holder"] = student_map.get(
            credit["holder"], guardian_map.get(credit["holder"], credit["holder"])
        )
    if data["expected"].get("pool_must_include") is not None:
        data["expected"]["pool_must_include"] = [
            student_map[value] for value in data["expected"]["pool_must_include"]
        ]
    data["transaction"]["reference"] = new_reference


def _localize_draft(data: dict, language: str, variant: int) -> None:
    original = data["transaction"].get("narration") or ""
    if language == "pcm":
        prefix = "Abeg, na school fees matter"
    elif language == "yo":
        prefix = "Jọ̀wọ́, owó ilé ẹ̀kọ́ ni èyí"
    else:
        prefix = "School payment evidence"
    data["transaction"]["narration"] = (
        f"{prefix}. {original} {_DRAFT_MARKER}_{variant:02d}"
    )
    data["expected"]["rationale"] = (
        f"{data['expected'].get('rationale', '')} "
        f"[{_DRAFT_MARKER}: reviewer must rewrite and verify this variant.]"
    ).strip()


def build_drafts(
    gold_dir: str | Path = "data/gold", *, seed: int = 3407
) -> list[GoldCase]:
    base = [load_case(path) for path in sorted(Path(gold_dir).glob("*.yaml"))]
    by_family = {case.scenario_family: case for case in base}
    if set(by_family) != SCENARIO_FAMILIES:
        missing = sorted(SCENARIO_FAMILIES - set(by_family))
        raise ValueError(f"one base case per scenario family is required; missing={missing}")

    languages = ["yo"] * 24 + ["pcm"] * 39 + ["en"] * 77
    random.Random(seed).shuffle(languages)
    slots = [
        (family, variant)
        for variant in range(1, 11)
        for family in sorted(SCENARIO_FAMILIES)
    ]
    drafts = []
    additional_abstentions = 9
    for slot_index, ((family, variant), language) in enumerate(zip(slots, languages)):
        source = by_family[family]
        data = source.model_dump(mode="json", by_alias=True, exclude_none=True)
        data["id"] = f"gold-draft-{slot_index + 1:04d}-{family}"
        data["language"] = language
        data["guardian_family"] = f"draft-{family}-{variant:02d}"
        data["template_family"] = f"draft-{family}-{variant:02d}"
        data["provenance"] = "draft"
        data["review"] = {"status": "draft"}
        _replace_ids(data, sorted(SCENARIO_FAMILIES).index(family) + 1, variant)
        _localize_draft(data, language, variant)
        if (
            additional_abstentions
            and source.expected.allocations
            and family != "name_match"
        ):
            data["expected"]["outcome"] = "review"
            data["expected"]["allocations"] = []
            data["expected"]["credits"] = []
            data["expected"]["pool_must_include"] = [
                student["id"] for student in data["setup"].get("students", [])
            ]
            data["expected"]["rationale"] += (
                " Drafted as a true abstention; reviewer must add and verify conflicting evidence."
            )
            additional_abstentions -= 1
        draft = GoldCase.model_validate(data)
        problems = check_case(draft)
        if problems:
            raise ValueError(f"{draft.id}: invalid generated draft: {problems}")
        drafts.append(draft)
    if additional_abstentions:
        raise AssertionError("draft planner failed to allocate the abstention target")
    return drafts


def write_drafts(
    drafts: list[GoldCase], draft_dir: str | Path = "data/gold_drafts"
) -> list[Path]:
    target = Path(draft_dir)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.glob("*.yaml")):
        raise FileExistsError(f"{target} already contains draft YAML files")
    paths = []
    for case in drafts:
        path = target / f"{case.id}.yaml"
        path.write_text(
            yaml.safe_dump(
                case.model_dump(mode="json", by_alias=True, exclude_none=True),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def promote_draft(
    draft_path: str | Path,
    gold_dir: str | Path,
    *,
    reviewer: str,
    reviewed_at: str | None = None,
) -> Path:
    source = Path(draft_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    case = GoldCase.model_validate(raw)
    if case.provenance != "draft" or not case.review or case.review.status != "draft":
        raise ValueError("only a draft case can be promoted")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    if _DRAFT_MARKER in (case.transaction.narration or "") or _DRAFT_MARKER in (
        case.expected.rationale or ""
    ):
        raise ValueError("reviewer must remove every REVIEW_DRAFT marker before promotion")
    raw["provenance"] = "team_authored"
    raw["review"] = {
        "status": "reviewed",
        "reviewer": reviewer,
        "reviewed_at": reviewed_at or datetime.now(timezone.utc).isoformat(),
    }
    reviewed = GoldCase.model_validate(raw)
    raw["review"]["content_sha256"] = review_checksum(reviewed)
    reviewed = GoldCase.model_validate(raw)
    problems = check_case(reviewed)
    if problems:
        raise ValueError(f"reviewed case is invalid: {problems}")
    target = Path(gold_dir) / source.name
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    source.unlink()
    return target


def validate_authored_set(
    cases: list[GoldCase], *, require_final: bool = False
) -> dict:
    counts = {
        "total": len(cases),
        "languages": {
            language: sum(case.language == language for case in cases)
            for language in ("en", "pcm", "yo")
        },
        "abstentions": sum(case.is_abstention() for case in cases),
        "families": {
            family: sum(case.scenario_family == family for case in cases)
            for family in sorted(SCENARIO_FAMILIES)
        },
    }
    errors = []
    for case in cases:
        errors.extend(f"{case.id}: {problem}" for problem in check_case(case))
        if case.review and case.review.status == "reviewed":
            if case.review.content_sha256 != review_checksum(case):
                errors.append(f"{case.id}: review checksum mismatch")
            if not case.review.reviewer or not case.review.reviewed_at:
                errors.append(f"{case.id}: reviewed case lacks reviewer metadata")
        if case.language == "yo" and (
            not case.review or case.review.status != "reviewed"
        ):
            errors.append(f"{case.id}: Yoruba case lacks reviewed sign-off")
    if require_final:
        if counts["total"] != TARGET_CASE_COUNT:
            errors.append(f"expected {TARGET_CASE_COUNT} cases, got {counts['total']}")
        if counts["languages"] != TARGET_LANGUAGES:
            errors.append(
                f"expected language counts {TARGET_LANGUAGES}, got {counts['languages']}"
            )
        if counts["abstentions"] != TARGET_ABSTENTIONS:
            errors.append(
                f"expected {TARGET_ABSTENTIONS} abstentions, got {counts['abstentions']}"
            )
        bad_families = {
            key: value
            for key, value in counts["families"].items()
            if value != TARGET_PER_FAMILY
        }
        if bad_families:
            errors.append(f"expected 11 cases per family, got {bad_families}")
        for case in cases:
            if case.id in LEGACY_CASE_IDS:
                continue
            if not case.review or case.review.status != "reviewed":
                errors.append(f"{case.id}: new authored case is not reviewed")
    return {**counts, "errors": errors, "valid": not errors}
