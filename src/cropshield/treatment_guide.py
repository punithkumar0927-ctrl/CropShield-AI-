from __future__ import annotations

from typing import Any

SAFE_PRACTICE_WARNINGS = [
    "Do not use an unverified pesticide or assume a dose from this model result.",
    "Do not spray without checking the current product label, local approval, protective equipment, and restrictions.",
    "Do not delay expert consultation when symptoms are severe or spreading.",
]


def build_verified_checklist(guidance: dict[str, Any] | None) -> list[tuple[str, str]]:
    if guidance is None:
        return []
    actions: list[tuple[str, str]] = []
    for category, field in (
        ("Immediate actions", "immediate_steps"),
        ("Cultural practices", "cultural_practices"),
        ("Biological options", "biological_options"),
        ("Verified chemical options", "chemical_options"),
        ("Other verified management", "management"),
        ("Monitoring", "monitoring"),
        ("Follow-up inspection", "follow_up_inspection"),
    ):
        for action in guidance.get(field, []):
            actions.append((category, action))
    return actions


def build_action_plan(guidance: dict[str, Any] | None) -> dict[str, list[str]]:
    """Organize verified knowledge into a plan without adding disease-specific remedies."""
    if guidance is None:
        return {
            "Do immediately": [
                "Record the visible symptoms and crop; do not choose treatment based only on this model result."
            ],
            "Do this week": [
                "Seek confirmation from a qualified local agricultural expert before choosing a treatment."
            ],
            "Monitor continuously": [
                "Record whether symptoms are spreading or new plants are affected."
            ],
            "Avoid doing": [
                *SAFE_PRACTICE_WARNINGS,
                "Do not apply pesticides based only on this model result or mix products.",
            ],
            "Contact an expert if": [
                "Symptoms spread rapidly, damage is severe, or identification remains uncertain."
            ],
        }
    return {
        "Do immediately": list(guidance["immediate_steps"]),
        "Do this week": list(guidance["management"]),
        "Monitor continuously": list(guidance["monitoring"]),
        "Avoid doing": [
            *SAFE_PRACTICE_WARNINGS,
            *guidance.get("what_not_to_do", []),
            "Do not apply pesticides based only on this model result.",
            "Do not mix products; follow current local regulations and the product label.",
        ],
        "Contact an expert if": [
            "Symptoms spread rapidly, damage is severe, the crop is valuable or commercial, "
            "or the identification remains uncertain."
        ],
    }
