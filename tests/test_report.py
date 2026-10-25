import pytest
from PIL import Image

from src.cropshield.report import build_prediction_report


def test_report_contains_prediction_and_safe_guidance():
    result = {
        "status": "ok",
        "predicted_class": "Tomato___<script>",
        "confidence": 0.8,
        "top_predictions": [{"class_name": "Tomato___<script>", "probability": 0.8}],
    }
    report = build_prediction_report(result, None, 0.25)

    assert "Preliminary Plant Image Report" in report
    assert "No verified disease-specific guidance" in report
    assert "&lt;script&gt;" in report
    assert "<script>" not in report
    assert "0.250 seconds" in report


def test_report_rejects_failed_prediction():
    with pytest.raises(ValueError, match="successful prediction"):
        build_prediction_report({"status": "invalid_image"}, None, 0.2)


def test_report_explains_uncertain_prediction_threshold():
    result = {
        "status": "ok",
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.6,
        "confidence_threshold": 0.8,
        "is_uncertain": True,
        "top_predictions": [],
    }
    report = build_prediction_report(result, None, 0.2)

    assert "below the validation-derived confidence threshold" in report
    assert "80.0%" in report


def test_report_includes_image_and_does_not_mislabel_entropy_as_threshold_failure():
    result = {
        "status": "ok",
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.6,
        "is_uncertain": True,
        "scope_warning": True,
        "normalized_entropy": 0.9,
        "top_prediction_margin": 0.03,
        "top_predictions": [],
    }
    report = build_prediction_report(result, None, 0.2, image=Image.new("RGB", (16, 12), "green"))

    assert "data:image/jpeg;base64," in report
    assert "Prediction ambiguity requires extra caution." in report
    assert "This condition may not be covered by the current model" in report
    assert "below the validation-derived confidence threshold" not in report
    assert "Prioritized action plan" in report


def test_report_includes_trust_explanation_and_model_file():
    result = {
        "status": "ok",
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.82,
        "trust_level": "Medium",
        "trust_explanation": "Model output is preliminary.",
        "model_file_name": "crop_model.keras",
    }

    report = build_prediction_report(result, None, 0.2)

    assert "Trust level (heuristic):</strong> Medium" in report
    assert "Model output is preliminary." in report
    assert "crop_model.keras" in report
