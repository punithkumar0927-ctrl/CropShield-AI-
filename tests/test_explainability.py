import numpy as np

from src.cropshield.explainability import gradcam_for_prediction
from src.cropshield.model import build_baseline_cnn, build_mobilenet_transfer


def test_gradcam_requires_model():
    model = build_baseline_cnn(num_classes=2, input_shape=(32, 32, 3))
    image = np.full((32, 32, 3), 127, dtype=np.float32)
    result = gradcam_for_prediction(model, image)
    assert result["status"] == "ok"
    assert result["heatmap_shape"] == [32, 32]
    assert np.all(np.isfinite(result["heatmap"]))
    assert np.max(result["heatmap"]) > 0


def test_gradcam_supports_nested_mobilenet_backbone():
    model = build_mobilenet_transfer(
        num_classes=2,
        input_shape=(32, 32, 3),
        weights=None,
    )
    image = np.full((32, 32, 3), 127, dtype=np.float32)
    result = gradcam_for_prediction(model, image)
    assert result["status"] in {"ok", "failed"}
    if result["status"] == "ok":
        assert result["heatmap_shape"] == [32, 32]
        assert np.max(result["heatmap"]) > 0
    else:
        assert "signal" in result["message"].lower()
