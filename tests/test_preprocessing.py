import io

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.cropshield.preprocessing import ensure_finite_probabilities, preprocess_for_model, validate_upload


def test_preprocess_for_model_resizes_rgb_and_preserves_expected_pixel_range():
    arr = np.full((32, 48, 3), 128, dtype=np.uint8)
    output = preprocess_for_model(arr, (16, 20))
    expected = tf.image.resize(arr.astype(np.float32), (16, 20), method="bilinear").numpy()
    assert output.shape == (1, 16, 20, 3)
    assert output.dtype == np.float32
    assert 127.0 <= float(output.max()) <= 129.0
    np.testing.assert_allclose(output[0], expected)


def test_preprocess_rejects_invalid_array_shape():
    with pytest.raises(ValueError, match="RGB"):
        preprocess_for_model(np.zeros((32, 32), dtype=np.uint8))


def test_probabilities_are_normalized():
    clean = ensure_finite_probabilities(np.array([0.2, 0.8], dtype=np.float32))
    assert np.all(np.isfinite(clean))
    assert np.isclose(clean.sum(), 1.0)


def test_invalid_probabilities_are_not_replaced_with_fake_values():
    with pytest.raises(ValueError, match="invalid probabilities"):
        ensure_finite_probabilities(np.array([0.2, np.nan], dtype=np.float32))


def test_validate_upload_rejects_unsupported_extension():
    buffer = io.BytesIO(b"not an image")
    buffer.name = "bad.txt"
    ok, message, _, _ = validate_upload(buffer)
    assert ok is False
    assert "valid image file" in message


def test_validate_upload_checks_decodability_and_dimensions():
    buffer = io.BytesIO()
    Image.new("RGB", (12, 10), color="green").save(buffer, format="PNG")
    buffer.name = "leaf.png"
    buffer.seek(0)
    valid, message, width, height = validate_upload(buffer)
    assert valid is True
    assert message == "OK"
    assert (width, height) == (12, 10)

    corrupt = io.BytesIO(b"not an image")
    corrupt.name = "leaf.jpg"
    valid, message, _, _ = validate_upload(corrupt)
    assert valid is False
    assert "blank or corrupted" in message.lower()


def test_validate_upload_accepts_webp_and_enforces_configured_file_size():
    webp = io.BytesIO()
    Image.new("RGB", (224, 224), color="green").save(webp, format="WEBP")
    webp.name = "leaf.webp"
    webp.type = "image/webp"
    webp.seek(0)
    valid, message, width, height = validate_upload(webp)
    assert valid is True
    assert message == "OK"
    assert (width, height) == (224, 224)

    oversized = io.BytesIO(b"x" * (10 * 1024 * 1024 + 1))
    oversized.name = "leaf.png"
    valid, message, _, _ = validate_upload(oversized)
    assert valid is False
    assert "10MB" in message


def test_validate_upload_rejects_mismatched_extension_and_mime():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="green").save(buffer, format="PNG")
    buffer.name = "leaf.jpg"
    buffer.type = "image/jpeg"
    buffer.seek(0)
    valid, message, _, _ = validate_upload(buffer)
    assert valid is False
    assert "extension" in message.lower()

    buffer.name = "leaf.png"
    buffer.type = "image/jpeg"
    buffer.seek(0)
    valid, message, _, _ = validate_upload(buffer)
    assert valid is False
    assert "mime" in message.lower()
