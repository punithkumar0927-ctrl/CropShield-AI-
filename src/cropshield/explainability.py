from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from .model import load_model
from .preprocessing import preprocess_for_model


def find_last_conv_layer(model: tf.keras.Model) -> Optional[tf.keras.layers.Layer]:
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    return conv_layers[-1] if conv_layers else None


def gradcam_for_prediction(model: tf.keras.Model, image_array: np.ndarray, class_index: int | None = None) -> Dict[str, Any]:
    if model is None:
        return {"status": "model_unavailable", "message": "Explainability requires a trained model."}
    conv_layer = find_last_conv_layer(model)
    activation_layer = conv_layer
    if activation_layer is None:
        activation_layer = next(
            (layer for layer in model.layers if layer.name == "gradcam_activations"),
            None,
        )
    if activation_layer is None:
        return {
            "status": "unsupported",
            "message": "This model does not expose a compatible convolutional activation for Grad-CAM.",
        }

    try:
        input_size = (int(model.input_shape[1]), int(model.input_shape[2]))
        output_shape = model.output_shape
        if (
            isinstance(output_shape, list)
            or len(output_shape) != 2
            or output_shape[-1] is None
        ):
            return {"status": "unsupported", "message": "The model output or requested class index is invalid."}
        if class_index is not None and not 0 <= class_index < output_shape[-1]:
            return {"status": "unsupported", "message": "The requested class index is outside model outputs."}
        model_input = preprocess_for_model(image_array, input_size)
        image_tensor = tf.convert_to_tensor(model_input, dtype=tf.float32)
        _ = model(image_tensor, training=False)
        gradient_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[activation_layer.output, model.layers[-1].output],
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return {"status": "unsupported", "message": f"Grad-CAM is not supported for this model: {exc}"}
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradient_model(image_tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        target = predictions[:, class_index]
    grads = tape.gradient(target, conv_outputs)
    if grads is None or not bool(tf.reduce_any(tf.not_equal(grads, 0)).numpy()):
        return {"status": "failed", "message": "Non-zero gradients were unavailable for the selected class."}
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap_max = tf.reduce_max(heatmap)
    if not bool(tf.math.is_finite(heatmap_max).numpy()) or float(heatmap_max.numpy()) <= 0:
        return {"status": "failed", "message": "The activation map had no positive class-specific signal."}
    heatmap = heatmap / heatmap_max
    heatmap_np = heatmap[0].numpy()
    original_size = (int(image_array.shape[1]), int(image_array.shape[0]))
    if heatmap_np.shape[:2] != image_array.shape[:2]:
        heatmap_np = (
            np.array(
                Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(original_size),
                dtype=np.float32,
            )
            / 255.0
        )
    if not np.all(np.isfinite(heatmap_np)):
        return {"status": "failed", "message": "Heatmap generation produced invalid values."}
    return {
        "status": "ok",
        "class_index": int(class_index),
        "heatmap_shape": list(heatmap_np.shape),
        "heatmap": heatmap_np,
    }


def overlay_heatmap(image_array: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.array(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    image = np.asarray(image_array).astype(np.float32)
    overlay = image.copy()
    overlay = overlay.astype(np.float32)
    overlay = overlay * 0.6 + heatmap_color.astype(np.float32) * 0.4
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_explanation(
    image_array: np.ndarray,
    model_path: str,
    class_index: int | None = None,
    model: tf.keras.Model | None = None,
) -> Dict[str, Any]:
    model = model if model is not None else load_model(model_path)
    if model is None:
        return {"status": "model_unavailable", "message": "No model is available for explanation."}
    result = gradcam_for_prediction(model, image_array, class_index=class_index)
    if result["status"] != "ok":
        return result
    overlay = overlay_heatmap(image_array, result["heatmap"])
    return {
        "status": "ok",
        "heatmap": result["heatmap"],
        "overlay": overlay,
        "class_index": result["class_index"],
    }
