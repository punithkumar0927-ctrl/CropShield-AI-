from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Dict

import numpy as np
from PIL import Image
import tensorflow as tf

from .config import DEFAULT_IMAGE_SIZE, LABELS_PATH, MODEL_PATH
from .model import load_labels, load_model
from .preprocessing import ensure_finite_probabilities, preprocess_for_model, validate_upload

LOGGER = logging.getLogger(__name__)


def _prepare_model_output(model: Any, labels: list[str]) -> bool:
    output_shape = model.output_shape
    if isinstance(output_shape, list) or not output_shape or output_shape[-1] != len(labels):
        return False
    return True


def _expected_image_size(model: Any) -> tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("Model must accept one RGB image tensor.")
    height, width, channels = input_shape[1:]
    if channels != 3:
        raise ValueError("Model must accept RGB image input.")
    if height is None or width is None:
        return DEFAULT_IMAGE_SIZE
    return int(height), int(width)


def _prediction_result(
    model: Any,
    labels: list[str],
    arr: np.ndarray,
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    processed = preprocess_for_model(arr, _expected_image_size(model))
    output = np.asarray(model.predict(processed, verbose=0))
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("The model must return one probability vector for a single image.")
    probs = ensure_finite_probabilities(output[0])
    if len(probs) != len(labels):
        raise ValueError("Model output dimensions do not match the saved class labels.")
    top_index = int(np.argmax(probs))
    confidence = float(probs[top_index])
    top_three = sorted(
        [(idx, float(prob)) for idx, prob in enumerate(probs)],
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    entropy = (
        float(-np.sum(probs * np.log2(np.maximum(probs, np.finfo(np.float32).tiny))) / np.log2(len(probs)))
        if len(probs) > 1
        else 0.0
    )
    margin = float(top_three[0][1] - top_three[1][1]) if len(top_three) > 1 else float(top_three[0][1])
    uncertainty_reasons = []
    if confidence_threshold is not None and confidence < confidence_threshold:
        uncertainty_reasons.append("confidence_below_validation_threshold")
    if entropy >= 0.8:
        uncertainty_reasons.append("high_prediction_entropy")
    if margin < 0.1:
        uncertainty_reasons.append("small_top_prediction_margin")
    return {
        "status": "ok",
        "image_size": [int(arr.shape[1]), int(arr.shape[0])],
        "predicted_class": labels[top_index],
        "confidence": round(confidence, 4),
        "is_uncertain": bool(uncertainty_reasons),
        "uncertainty_reasons": uncertainty_reasons,
        "normalized_entropy": round(entropy, 4),
        "top_prediction_margin": round(margin, 4),
        "scope_warning": bool(uncertainty_reasons),
        "confidence_threshold": confidence_threshold,
        "top_predictions": [
            {"class_name": labels[idx], "probability": round(float(prob), 4)} for idx, prob in top_three
        ],
        "all_probabilities": [
            {"class_name": labels[idx], "probability": round(float(prob), 4)}
            for idx, prob in enumerate(probs)
        ],
    }


def predict_image_array(
    image_array: np.ndarray,
    labels: list[str],
    model: Any,
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    if model is None or not labels:
        return {
            "status": "model_unavailable",
            "message": "No trained CropShield model is available. Please train the model first.",
            "predictions": [],
        }
    if not _prepare_model_output(model, labels):
        return {
            "status": "invalid_model",
            "message": "The trained model output does not match the saved class labels.",
            "predictions": [],
        }
    try:
        return _prediction_result(model, labels, np.asarray(image_array, dtype=np.float32), confidence_threshold)
    except (ValueError, RuntimeError, tf.errors.OpError) as exc:
        LOGGER.exception("Prediction failed for an in-memory image.")
        return {"status": "invalid_model", "message": str(exc), "predictions": []}


def predict_image_file(
    image_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    labels_path: str | Path = LABELS_PATH,
    model: Any = None,
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    model = model if model is not None else load_model(model_path)
    labels = load_labels(labels_path)
    if model is None or not labels:
        return {
            "status": "model_unavailable",
            "message": "No trained CropShield model is available. Please train the model first.",
            "predictions": [],
        }
    try:
        image_file = Path(image_path)
        if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return {
                "status": "invalid_image",
                "message": "Unsupported file type. Use JPG, JPEG, PNG, or WEBP.",
                "predictions": [],
            }
        with image_file.open("rb") as file_handle:
            valid, message, _width, _height = validate_upload(file_handle)
    except OSError:
        return {"status": "invalid_image", "message": "Image file could not be read.", "predictions": []}
    if not valid:
        return {"status": "invalid_image", "message": message, "predictions": []}
    if not _prepare_model_output(model, labels):
        return {
            "status": "invalid_model",
            "message": "The trained model output does not match the saved class labels.",
            "predictions": [],
        }
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except (OSError, ValueError):
        return {
            "status": "invalid_image",
            "message": "The uploaded image could not be decoded. Please upload a valid JPG, JPEG, or PNG file.",
            "predictions": [],
        }
    try:
        return _prediction_result(model, labels, arr, confidence_threshold)
    except (ValueError, RuntimeError, tf.errors.OpError) as exc:
        LOGGER.exception("Prediction failed for image file %s.", Path(image_path).name)
        return {"status": "invalid_model", "message": str(exc), "predictions": []}


def predict_uploaded_file(
    uploaded_file: Any,
    model_path: str | Path = MODEL_PATH,
    labels_path: str | Path = LABELS_PATH,
    model: Any = None,
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    model = model if model is not None else load_model(model_path)
    labels = load_labels(labels_path)
    if model is None or not labels:
        return {
            "status": "model_unavailable",
            "message": "No trained CropShield model is available. Please train the model first.",
            "predictions": [],
        }
    if uploaded_file is None:
        return {"status": "invalid_image", "message": "Please upload a valid JPG, JPEG, or PNG image.", "predictions": []}
    valid, message, _width, _height = validate_upload(uploaded_file)
    if not valid:
        return {"status": "invalid_image", "message": message, "predictions": []}
    if not _prepare_model_output(model, labels):
        return {
            "status": "invalid_model",
            "message": "The trained model output does not match the saved class labels.",
            "predictions": [],
        }
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    except (OSError, ValueError):
        return {"status": "invalid_image", "message": "The uploaded image could not be decoded.", "predictions": []}
    try:
        return _prediction_result(model, labels, arr, confidence_threshold)
    except (ValueError, RuntimeError, tf.errors.OpError) as exc:
        LOGGER.exception("Prediction failed for uploaded image.")
        return {"status": "invalid_model", "message": str(exc), "predictions": []}
