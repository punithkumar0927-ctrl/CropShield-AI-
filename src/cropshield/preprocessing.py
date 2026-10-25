from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from .config import ALLOWED_EXTENSIONS, DEFAULT_IMAGE_SIZE, MAX_IMAGE_PIXELS, MAX_UPLOAD_SIZE_MB

LOGGER = logging.getLogger(__name__)


def validate_upload(file_obj: Any, max_size_mb: int = MAX_UPLOAD_SIZE_MB) -> Tuple[bool, str, int, int]:
    if file_obj is None:
        return False, "No file uploaded.", 0, 0
    content_type = getattr(file_obj, "type", None)
    if content_type and content_type.lower() not in {
        "image/jpeg", "image/png", "image/jpg", "image/webp"
    }:
        return False, "Please upload a valid image file. Use JPG, JPEG, PNG, or WEBP.", 0, 0
    if hasattr(file_obj, "name"):
        suffix = Path(str(file_obj.name)).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return False, "Please upload a valid image file. Use JPG, JPEG, PNG, or WEBP.", 0, 0
    try:
        file_obj.seek(0, io.SEEK_END)
        size_bytes = file_obj.tell()
        file_obj.seek(0)
    except (AttributeError, OSError, io.UnsupportedOperation):
        return False, "Uploaded file could not be read.", 0, 0
    if size_bytes <= 0:
        return False, "Uploaded file is empty.", 0, 0
    max_size_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_size_bytes:
        return False, f"Image exceeds the {max_size_mb}MB limit.", 0, 0
    try:
        file_obj.seek(0)
        with Image.open(file_obj) as image:
            width, height = image.size
            image_format = image.format
            extension = Path(str(getattr(file_obj, "name", ""))).suffix.lower()
            expected_format = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
            }.get(extension)
            if expected_format and image_format != expected_format:
                return False, "File extension does not match the image content.", 0, 0
            expected_mime = {
                "JPEG": {"image/jpeg", "image/jpg"},
                "PNG": {"image/png"},
                "WEBP": {"image/webp"},
            }.get(image_format)
            if content_type and expected_mime and content_type.lower() not in expected_mime:
                return False, "MIME type does not match the image content.", 0, 0
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                return False, "Image dimensions are invalid or exceed the 40-megapixel limit.", 0, 0
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                return False, "Unsupported image content. Use a JPG, JPEG, PNG, or WEBP image.", 0, 0
            image.verify()
        return True, "OK", width, height
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        LOGGER.warning("Rejected an uploaded image that could not be decoded.")
        return False, "Image appears blank or corrupted.", 0, 0


def load_image_array(image_buffer: Any, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    with Image.open(image_buffer) as image:
        rgb_image = np.asarray(image.convert("RGB"), dtype=np.float32)
    return tf.image.resize(rgb_image, target_size, method="bilinear").numpy()


def preprocess_for_model(image_array: np.ndarray, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image_array = np.asarray(image_array)
    if image_array.ndim != 3 or image_array.shape[2] not in {3, 4}:
        raise ValueError("Expected an image array with 3 (RGB) or 4 (RGBA) channels.")
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    if image_array.shape[:2] != input_size:
        image_array = tf.image.resize(
            np.clip(image_array, 0, 255).astype(np.float32),
            input_size,
            method="bilinear",
        ).numpy()
    image_array = np.asarray(image_array, dtype=np.float32)
    if not np.all(np.isfinite(image_array)):
        raise ValueError("Image contains non-finite pixel values.")
    return np.expand_dims(image_array, axis=0)


def ensure_finite_probabilities(probabilities: np.ndarray) -> np.ndarray:
    arr = np.asarray(probabilities, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Expected a non-empty, one-dimensional probability vector.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Model returned invalid probabilities.")
    total = arr.sum()
    if total <= 0 or not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError("Model output is not a normalized probability vector.")
    arr = arr / total
    return arr.astype(np.float32)
