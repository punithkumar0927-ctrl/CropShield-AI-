from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropshield.config import MODEL_PATH, TFLITE_MODEL_PATH  # noqa: E402


def export_tflite_model(
    model_path: str | Path = MODEL_PATH,
    output_path: str | Path = TFLITE_MODEL_PATH,
) -> Path:
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"Trained Keras model was not found: {source}")
    model = tf.keras.models.load_model(source)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(tflite_bytes)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained CropShield model to TensorFlow Lite.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a trusted trained .keras model.")
    parser.add_argument("--output", type=Path, default=TFLITE_MODEL_PATH, help="Output .tflite path.")
    args = parser.parse_args()
    output_path = export_tflite_model(args.model, args.output)
    print(f"TFLite model exported to: {output_path}")


if __name__ == "__main__":
    main()
