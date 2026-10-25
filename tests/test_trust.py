import numpy as np
from PIL import Image
import pytest

from src.cropshield.image_quality import assess_image_quality
from src.cropshield.trust import assess_trust, load_trust_thresholds


def _good_leaf_photo():
    pixels = np.full((300, 400, 3), 80, dtype=np.uint8)
    pixels[40:260, 70:330] = (35, 150, 45)
    pixels[80:220:2, 100:300:2] = (70, 190, 70)
    return Image.fromarray(pixels)


def test_trust_high_requires_calibrated_confidence_and_good_photo():
    quality = assess_image_quality(_good_leaf_photo())

    result = assess_trust(0.94, 0.6, quality, 0.7, heatmap_leaf_focus=0.8)

    assert result["level"] == "High"
    assert "High trust" in result["explanation"]


def test_trust_withholds_high_without_calibration_and_warns_for_background_attention():
    quality = assess_image_quality(_good_leaf_photo())

    uncalibrated = assess_trust(0.99, 0.8, quality, None)
    background = assess_trust(0.99, 0.8, quality, 0.7, heatmap_leaf_focus=0.1)

    assert uncalibrated["level"] == "Medium"
    assert "High trust is withheld" in uncalibrated["explanation"]
    assert background["level"] == "Low"
    assert "background" in background["explanation"]


def test_trust_returns_cannot_assess_for_blank_image():
    quality = assess_image_quality(Image.new("RGB", (300, 300), color="white"))

    result = assess_trust(0.99, 0.8, quality, 0.7)

    assert result["level"] == "Cannot Assess"


def test_detection_threshold_configuration_is_loaded_and_validated(tmp_path):
    thresholds = load_trust_thresholds()
    assert thresholds["high_confidence"] == pytest.approx(0.8)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"schema_version": 1, "high_confidence": 1.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_trust_thresholds(invalid_path)
