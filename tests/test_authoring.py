from bursa_eval.authoring import (
    TARGET_ABSTENTIONS,
    TARGET_LANGUAGES,
    build_drafts,
    validate_authored_set,
)


def test_draft_plan_hits_locked_expansion_counts():
    drafts = build_drafts()
    assert len(drafts) == 140
    languages = {
        language: sum(case.language == language for case in drafts)
        for language in TARGET_LANGUAGES
    }
    assert languages == {"en": 77, "pcm": 39, "yo": 24}
    existing_abstentions = 3
    assert sum(case.is_abstention() for case in drafts) + existing_abstentions == \
        TARGET_ABSTENTIONS
    per_family = {
        family: sum(case.scenario_family == family for case in drafts)
        for family in {case.scenario_family for case in drafts}
    }
    assert set(per_family.values()) == {10}
    assert all(case.review.status == "draft" for case in drafts)


def test_final_review_gate_does_not_depend_on_case_order():
    report = validate_authored_set(
        list(reversed(build_drafts())), require_final=True
    )
    assert any(
        "new authored case is not reviewed" in error
        for error in report["errors"]
    )
