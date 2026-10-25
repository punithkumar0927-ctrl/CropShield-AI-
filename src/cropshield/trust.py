from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DETECTION_THRESHOLDS_PATH
from .image_quality import ImageQualityAssessment

DEFAULT_THRESHOLDS = {
    "high_confidence": 0.8,
    "medium_confidence": 0.55,
    "high_margin": 0.25,
    "medium_margin": 0.1,
    "minimum_leaf_coverage": 0.01,
    "high_leaf_coverage": 0.2,
    "minimum_blur_score": 20.0,
    "minimum_heatmap_leaf_focus": 0.5,
}


def load_trust_thresholds(path: str | Path = DETECTION_THRESHOLDS_PATH) -> dict[str, float]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file_handle:
        document = json.load(file_handle)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"Unsupported detection thresholds format: {source}")
    values = document.copy()
    values.pop("schema_version")
    if set(values) != set(DEFAULT_THRESHOLDS):
        raise ValueError(f"Detection thresholds must define exactly {sorted(DEFAULT_THRESHOLDS)}.")
    for name, default in DEFAULT_THRESHOLDS.items():
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Detection threshold {name!r} must be numeric.")
        if value < 0 or (value > 1 and default <= 1):
            raise ValueError(f"Detection threshold {name!r} is outside its valid range.")
    if values["medium_confidence"] > values["high_confidence"]:
        raise ValueError("Medium confidence threshold cannot exceed high confidence threshold.")
    if values["medium_margin"] > values["high_margin"]:
        raise ValueError("Medium margin threshold cannot exceed high margin threshold.")
    if values["minimum_leaf_coverage"] > values["high_leaf_coverage"]:
        raise ValueError("Minimum leaf coverage cannot exceed high leaf coverage.")
    return {name: float(value) for name, value in values.items()}


def assess_trust(
    confidence: float,
    margin: float,
    quality: ImageQualityAssessment,
    validation_threshold: float | None,
    heatmap_leaf_focus: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    if not 0 <= confidence <= 1 or not 0 <= margin <= 1:
        raise ValueError("Confidence and prediction margin must be between 0 and 1.")
    if validation_threshold is not None and not 0 <= validation_threshold <= 1:
        raise ValueError("Validation threshold must be between 0 and 1.")
    if heatmap_leaf_focus is not None and not 0 <= heatmap_leaf_focus <= 1:
        raise ValueError("Heatmap leaf-focus ratio must be between 0 and 1.")

    if not quality.acceptable or quality.leaf_coverage_fraction <= limits["minimum_leaf_coverage"]:
        return {
            "level": "Cannot Assess",
            "explanation": (
                "Cannot assess: no clear leaf-like region or image quality is too poor. "
                "Retake the photo before interpreting a model result."
            ),
        }

    calibrated = validation_threshold is not None
    below_validation_threshold = calibrated and confidence < float(validation_threshold)
    quality_limited = (
        quality.blur_score < limits["minimum_blur_score"]
        or quality.glare_fraction > 0.15
        or quality.brightness < 35
        or quality.brightness > 235
        or quality.leaf_coverage_fraction < limits["minimum_leaf_coverage"]
    )
    background_attention = (
        heatmap_leaf_focus is not None
        and heatmap_leaf_focus < limits["minimum_heatmap_leaf_focus"]
    )
    if (
        confidence >= limits["high_confidence"]
        and margin >= limits["high_margin"]
        and quality.leaf_coverage_fraction >= limits["high_leaf_coverage"]
        and not quality_limited
        and not below_validation_threshold
        and not background_attention
        and calibrated
    ):
        level = "High"
        explanation = "High trust: good image quality and a strong prediction above its validation threshold."
    elif (
        confidence >= limits["medium_confidence"]
        and margin >= limits["medium_margin"]
        and not below_validation_threshold
        and not background_attention
    ):
        level = "Medium"
        explanation = "Medium trust: prediction is possible, but image quality or model certainty is moderate."
    else:
        level = "Low"
        explanation = "Low trust: the model is uncertain or the photo quality is limited. Take another photo or seek expert review."
    if not calibrated:
        explanation += " No validation-derived confidence threshold is available; High trust is withheld."
    if background_attention:
        explanation += (
            " The attention map focuses mostly on the background outside the color-segmented leaf-like region."
        )
    return {"level": level, "explanation": explanation}
