from datetime import date

import pytest
from PIL import Image

from src.cropshield.expert_request import build_expert_request


def test_expert_request_is_local_export_with_explicit_photo_and_safe_html():
    observation = {
        "id": 4,
        "observed_on": date(2026, 10, 25).isoformat(),
        "record_type": "ai_prediction",
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.73,
        "notes": "<script>alert(1)</script>",
    }

    html = build_expert_request(
        "Tomato 1",
        "North field",
        [observation],
        4,
        "Spots are increasing.",
        "Low — expert review requested",
        location="Optional location",
        include_selected_photo=True,
        image=Image.new("RGB", (12, 12), "green"),
    )

    assert "does not send it anywhere" in html
    assert "No Grad-CAM heatmap was generated" in html
    assert "Tomato___Early_blight" in html
    assert "data:image/jpeg;base64," in html
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_expert_request_requires_real_observation_and_symptom_text():
    with pytest.raises(ValueError, match="Select an observation"):
        build_expert_request("Plant", "Field", [], 1, "Symptoms", "Preliminary — not verified")

    observation = {
        "id": 1,
        "observed_on": "2026-10-25",
        "record_type": "user_observation",
        "predicted_class": None,
        "confidence": None,
        "notes": "User note.",
    }
    with pytest.raises(ValueError, match="Observed symptoms"):
        build_expert_request("Plant", "Field", [observation], 1, " ", "Preliminary — not verified")
