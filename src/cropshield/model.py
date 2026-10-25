from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import threading
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from .config import DEFAULT_IMAGE_SIZE, DEFAULT_LEARNING_RATE, LABELS_PATH, MODEL_PATH, TFLITE_MODEL_PATH

LOGGER = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_baseline_cnn(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (*DEFAULT_IMAGE_SIZE, 3),
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.Rescaling(scale=1.0 / 255.0),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class TFLiteModel:
    def __init__(self, interpreter: tf.lite.Interpreter) -> None:
        self._interpreter = interpreter
        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()
        if len(self._input_details) != 1 or len(self._output_details) != 1:
            raise ValueError("TFLite model must have exactly one input and one output tensor.")
        input_shape = tuple(int(size) for size in self._input_details[0]["shape"])
        output_shape = tuple(int(size) for size in self._output_details[0]["shape"])
        if len(input_shape) != 4 or len(output_shape) != 2:
            raise ValueError("TFLite model must accept one batched image and return class probabilities.")
        if input_shape[0] != 1 or output_shape[0] != 1 or input_shape[-1] != 3:
            raise ValueError("TFLite model must accept one RGB image at a time.")
        if input_shape[1] <= 0 or input_shape[2] <= 0 or output_shape[-1] <= 0:
            raise ValueError("TFLite model input and output dimensions must be fixed and non-empty.")
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._interpreter.allocate_tensors()
        self._inference_lock = threading.Lock()

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        value = np.asarray(batch)
        if value.shape != self.input_shape:
            raise ValueError(f"TFLite model expects input shape {self.input_shape}, got {value.shape}.")
        detail = self._input_details[0]
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = detail["quantization"]
            if scale <= 0:
                raise ValueError("Quantized TFLite input is missing a valid scale.")
            limits = np.iinfo(dtype)
            value = np.clip(np.rint(value / scale + zero_point), limits.min, limits.max).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            value = value.astype(dtype)
        else:
            raise ValueError(f"Unsupported TFLite input type: {dtype}.")
        with self._inference_lock:
            self._interpreter.set_tensor(detail["index"], value)
            self._interpreter.invoke()
            output_detail = self._output_details[0]
            output = self._interpreter.get_tensor(output_detail["index"])
        output_dtype = output_detail["dtype"]
        if np.issubdtype(output_dtype, np.integer):
            scale, zero_point = output_detail["quantization"]
            if scale <= 0:
                raise ValueError("Quantized TFLite output is missing a valid scale.")
            output = (output.astype(np.float32) - zero_point) * scale
        return np.asarray(output, dtype=np.float32)


def load_tflite_model(path: str | Path = TFLITE_MODEL_PATH) -> TFLiteModel | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    return TFLiteModel(interpreter)


def build_mobilenet_transfer(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (*DEFAULT_IMAGE_SIZE, 3),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "MobileNetV2 pretrained weights could not be loaded. Check network access or retry; "
            "the project does not silently substitute random weights."
        ) from exc
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Activation("linear", name="gradcam_activations")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(labels: list[str], path: str | Path = LABELS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def load_labels(path: str | Path = LABELS_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, str) and item for item in data):
        raise ValueError(f"Invalid class-label file: {path}")
    if len(set(data)) != len(data):
        raise ValueError(f"Class-label file contains duplicate labels: {path}")
    return data


def save_model(model: tf.keras.Model, path: str | Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: str | Path = MODEL_PATH) -> tf.keras.Model | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        LOGGER.error("Could not load model artifact %s: %s", model_path.name, exc)
        return None


def model_summary(model: tf.keras.Model) -> Dict[str, int | str]:
    return {
        "trainable_params": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
        "total_params": int(model.count_params()),
        "layers": len(model.layers),
        "name": model.name,
    }
