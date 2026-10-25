import numpy as np
import pytest
import tensorflow as tf

from scripts.export_tflite import export_tflite_model
from src.cropshield.model import load_tflite_model
from src.cropshield.predict import _prediction_result


def test_exported_tflite_model_runs_local_prediction(tmp_path):
    keras_path = tmp_path / "small_model.keras"
    tflite_path = tmp_path / "small_model.tflite"
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4, 4, 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.save(keras_path)

    export_tflite_model(keras_path, tflite_path)
    lite_model = load_tflite_model(tflite_path)
    assert lite_model is not None
    result = _prediction_result(lite_model, ["Tomato___healthy", "Tomato___example"], np.ones((8, 8, 3)) * 128)

    assert result["status"] == "ok"
    assert result["predicted_class"] in {"Tomato___healthy", "Tomato___example"}
    assert sum(row["probability"] for row in result["top_predictions"]) == pytest.approx(1.0)


def test_tflite_export_requires_an_existing_trained_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained Keras model"):
        export_tflite_model(tmp_path / "missing.keras", tmp_path / "model.tflite")
