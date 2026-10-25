from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldRiskAssessment:
    level: str
    score: int
    factors: tuple[str, ...]
    affected_fraction: float


def assess_field_risk(
    affected_leaves: int,
    inspected_leaves: int,
    nearby_affected_plants: bool | None,
    prolonged_leaf_wetness: bool | None,
    overhead_irrigation: bool | None,
    rapid_symptom_spread: bool | None = None,
) -> FieldRiskAssessment:
    """Produce a transparent, generic field-risk prompt from user-reported conditions."""
    if inspected_leaves < 1:
        raise ValueError("The number of inspected leaves must be at least one.")
    if not 0 <= affected_leaves <= inspected_leaves:
        raise ValueError("Affected leaves must be between zero and the inspected leaf count.")

    fraction = affected_leaves / inspected_leaves
    score = 0
    factors: list[str] = []
    if fraction >= 0.25:
        score += 2
        factors.append("At least one quarter of inspected leaves are affected.")
    elif fraction >= 0.05:
        score += 1
        factors.append("Some inspected leaves are affected.")
    if nearby_affected_plants:
        score += 2
        factors.append("You reported nearby plants with similar symptoms.")
    if prolonged_leaf_wetness:
        score += 1
        factors.append("You reported prolonged leaf wetness or humid weather.")
    if overhead_irrigation:
        score += 1
        factors.append("You reported overhead irrigation that can wet foliage.")
    if rapid_symptom_spread:
        score += 2
        factors.append("You reported rapid symptom spread; inspect nearby plants and seek expert advice.")

    level = "Elevated" if score >= 4 else "Moderate" if score >= 2 else "Low"
    if not factors:
        factors.append("No additional spread-favoring conditions were reported.")
    return FieldRiskAssessment(level, score, tuple(factors), fraction)
