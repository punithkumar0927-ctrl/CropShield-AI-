from src.cropshield.treatment_guide import (
    SAFE_PRACTICE_WARNINGS,
    build_action_plan,
    build_verified_checklist,
)


def test_action_plan_maps_only_verified_management_guidance():
    guidance = {
        "immediate_steps": ["Inspect plants."],
        "management": ["Improve spacing."],
        "monitoring": ["Check for new symptoms."],
    }

    plan = build_action_plan(guidance)

    assert plan["Do immediately"] == ["Inspect plants."]
    assert plan["Do this week"] == ["Improve spacing."]
    assert plan["Monitor continuously"] == ["Check for new symptoms."]
    assert any("Do not apply pesticides" in item for item in plan["Avoid doing"])


def test_action_plan_without_verified_guidance_stays_generic_and_escalates():
    plan = build_action_plan(None)

    assert len(plan["Do immediately"]) == 1
    assert "qualified local agricultural expert" in plan["Do this week"][0]
    assert "rapidly" in plan["Contact an expert if"][0]


def test_checklist_contains_only_explicitly_verified_guidance_fields():
    actions = build_verified_checklist(
        {
            "immediate_steps": ["Isolate a symptomatic sample for expert review."],
            "management": ["Use only reviewed guidance."],
            "monitoring": ["Inspect weekly."],
            "cultural_practices": ["Remove crop debris if source-supported."],
            "chemical_options": ["Only verified, locally approved option."],
        }
    )

    assert actions == [
        ("Immediate actions", "Isolate a symptomatic sample for expert review."),
        ("Cultural practices", "Remove crop debris if source-supported."),
        ("Verified chemical options", "Only verified, locally approved option."),
        ("Other verified management", "Use only reviewed guidance."),
        ("Monitoring", "Inspect weekly."),
    ]
    assert build_verified_checklist(None) == []
    plan = build_action_plan(None)
    assert all(warning in plan["Avoid doing"] for warning in SAFE_PRACTICE_WARNINGS)
