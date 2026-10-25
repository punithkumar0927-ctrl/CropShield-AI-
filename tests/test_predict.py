import io

import numpy as np
from PIL import Image

from src.cropshield.model import save_labels
from src.cropshield.predict import predict_image_file, predict_uploaded_file


class FakeModel:
    input_shape = (None, 32, 32, 3)
    output_shape = (None, 2)

    def __init__(self):
        self.seen_image = None

    def predict(self, image, verbose=0):
        self.seen_image = image
        return np.asarray([[0.25, 0.75]], dtype=np.float32)


def _upload() -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), color=(255, 0, 0)).save(buffer, format="PNG")
    buffer.name = "leaf.png"
    buffer.type = "image/png"
    buffer.seek(0)
    return buffer


def test_missing_model_returns_clear_message(tmp_path):
    result = predict_image_file(
        tmp_path / "missing.png",
        model_path=tmp_path / "missing.keras",
        labels_path=tmp_path / "labels.json",
    )
    assert result["status"] == "model_unavailable"
    assert "trained cropshield model" in result["message"].lower()


def test_prediction_response_and_raw_rgb_preprocessing(tmp_path):
    labels_path = tmp_path / "labels.json"
    save_labels(["Crop___healthy", "Crop___rust"], labels_path)
    model = FakeModel()
    result = predict_uploaded_file(_upload(), labels_path=labels_path, model=model)
    assert result["status"] == "ok"
    assert result["predicted_class"] == "Crop___rust"
    assert result["image_size"] == [20, 10]
    assert len(result["top_predictions"]) == 2
    assert np.isclose(model.seen_image.max(), 255.0)
    assert model.seen_image.shape == (1, 32, 32, 3)
    assert result["normalized_entropy"] > 0.8
    assert "high_prediction_entropy" in result["uncertainty_reasons"]
    assert result["scope_warning"] is True


def test_prediction_rejects_label_output_mismatch(tmp_path):
    labels_path = tmp_path / "labels.json"
    save_labels(["only-one-label"], labels_path)
    result = predict_uploaded_file(_upload(), labels_path=labels_path, model=FakeModel())
    assert result["status"] == "invalid_model"
    assert "does not match" in result["message"]


def test_prediction_is_flagged_below_validation_threshold(tmp_path):
    labels_path = tmp_path / "labels.json"
    save_labels(["Crop___healthy", "Crop___rust"], labels_path)
    result = predict_uploaded_file(
        _upload(),
        labels_path=labels_path,
        model=FakeModel(),
        confidence_threshold=0.8,
    )
    assert result["status"] == "ok"
    assert result["predicted_class"] == "Crop___rust"
    assert result["is_uncertain"] is True
    assert result["confidence_threshold"] == 0.8


def test_prediction_accepts_real_webp_upload(tmp_path):
    labels_path = tmp_path / "labels.json"
    save_labels(["Crop___healthy", "Crop___rust"], labels_path)
    webp = io.BytesIO()
    Image.new("RGB", (256, 256), color=(25, 100, 30)).save(webp, format="WEBP")
    webp.name = "field.webp"
    webp.type = "image/webp"
    webp.seek(0)

    result = predict_uploaded_file(webp, labels_path=labels_path, model=FakeModel())

    assert result["status"] == "ok"
    assert result["predicted_class"] == "Crop___rust"


def test_array_prediction_reuses_real_model_and_label_validation():
    from src.cropshield.predict import predict_image_array

    model = FakeModel()
    result = predict_image_array(
        np.full((256, 256, 3), 120, dtype=np.float32),
        ["Crop___healthy", "Crop___rust"],
        model,
    )

    assert result["status"] == "ok"
    assert result["predicted_class"] == "Crop___rust"
