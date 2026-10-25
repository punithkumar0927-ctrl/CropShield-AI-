from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropshield.config import ARTIFACTS_DIR, LABELS_PATH, METRICS_PATH, MODELS_DIR, REPORTS_DIR  # noqa: E402
from src.cropshield.dataset import detect_dataset_root  # noqa: E402
from src.cropshield.evaluate import (  # noqa: E402
    evaluate_model,
    load_metrics,
    save_confusion_matrix_png,
    save_metrics,
    save_model_comparison,
)
from src.cropshield.model import load_labels, load_model  # noqa: E402
from src.cropshield.train import build_dataset_from_records  # noqa: E402


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_model_type(requested: str | None) -> str:
    if requested:
        return requested
    config_path = MODELS_DIR / "model_config.json"
    if config_path.is_file():
        model_type = _read_json(config_path).get("model_type")
        if model_type in {"baseline", "mobilenet"}:
            return model_type
    existing = [
        model_type
        for model_type in ("baseline", "mobilenet")
        if (MODELS_DIR / f"{model_type}_best.keras").is_file()
    ]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            "Both trained model types exist but models/model_config.json does not identify the latest model. "
            "Specify --model-type baseline or --model-type mobilenet."
        )
    raise FileNotFoundError("No trained model is available under models/. Run scripts/train_model.py first.")


def _load_saved_test_records(manifest_path: Path, dataset_root: Path, labels: list[str]) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Saved split manifest not found at {manifest_path}. Retrain the model to create a held-out test split."
        )
    manifest = pd.read_csv(manifest_path)
    if not {"path", "label", "split"}.issubset(manifest.columns):
        raise ValueError(f"Split manifest is missing required path, label, or split columns: {manifest_path}")
    if manifest["path"].duplicated().any():
        raise ValueError("Split manifest contains duplicate image paths; refusing an ambiguous evaluation split.")
    test_rows = manifest.loc[manifest["split"] == "test", ["path", "label"]]
    if test_rows.empty:
        raise ValueError("The saved held-out test split is empty.")

    root = dataset_root.resolve()
    records: list[dict[str, str]] = []
    for path_value, label_value in test_rows.itertuples(index=False, name=None):
        image_path = Path(str(path_value)).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image recorded in the saved split is missing: {image_path}")
        if not image_path.is_relative_to(root):
            raise ValueError(f"Saved test image is outside the selected dataset root: {image_path}")
        label = str(label_value)
        if label not in labels:
            raise ValueError(f"Test split label '{label}' is missing from the model's saved class mapping.")
        records.append({"path": str(image_path), "label": label})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CropShield model using its persisted held-out test split."
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="Folder-per-class dataset directory.")
    parser.add_argument("--model-type", choices=["baseline", "mobilenet"], default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit .keras model path.")
    parser.add_argument("--labels-path", type=str, default=None, help="Optional JSON class-label file.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size; defaults to training config.")
    args = parser.parse_args()

    model_type = _resolve_model_type(args.model_type)
    dataset_root = (
        detect_dataset_root(Path(args.dataset_root).expanduser())
        if args.dataset_root
        else detect_dataset_root()
    )
    if dataset_root is None:
        raise FileNotFoundError("Dataset not found. Supply --dataset-root pointing to the training dataset.")
    dataset_root = dataset_root.resolve()

    model_path = Path(args.model_path).expanduser() if args.model_path else MODELS_DIR / f"{model_type}_best.keras"
    if not model_path.is_file() and args.model_path is None:
        generic_model = MODELS_DIR / "crop_model.keras"
        if generic_model.is_file():
            model_path = generic_model
    model = load_model(model_path)
    if model is None:
        raise FileNotFoundError(f"No readable trained model found at {model_path}.")

    labels_path = (
        Path(args.labels_path).expanduser()
        if args.labels_path
        else MODELS_DIR / f"{model_type}_class_labels.json"
    )
    if not labels_path.is_file() and args.labels_path is None:
        labels_path = LABELS_PATH
    labels = load_labels(labels_path)
    if not labels:
        raise FileNotFoundError(f"No class labels found at {labels_path}.")
    if len(model.output_shape) != 2 or model.output_shape[-1] != len(labels):
        raise ValueError("Model output dimensions do not match the saved class-label file.")

    config_path = MODELS_DIR / f"{model_type}_config.json"
    if not config_path.is_file():
        config_path = MODELS_DIR / "model_config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    image_size = tuple(int(value) for value in config.get("image_size", [model.input_shape[1], model.input_shape[2]]))
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError(f"Invalid image size in model config: {image_size}.")
    batch_size = args.batch_size or int(config.get("batch_size", 32))
    seed = int(config.get("seed", 42))
    if batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")

    manifest_path = ARTIFACTS_DIR / "dataset_split.csv"
    if not manifest_path.is_file():
        manifest_path = PROJECT_ROOT / "data" / "split_manifest.csv"
    test_records = _load_saved_test_records(manifest_path, dataset_root, labels)
    test_dataset = build_dataset_from_records(
        test_records,
        labels,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    result = evaluate_model(model, test_dataset, labels)
    evaluated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "evaluated_at": evaluated_at,
        "model_file": model_path.name,
        "test_image_count": result["image_count"],
        "number_of_classes": len(labels),
        "image_size": list(image_size),
        "batch_size": batch_size,
        "random_seed": seed,
        "dataset_path": str(dataset_root),
        "test_split_file": str(manifest_path.resolve()),
        "test_split_method": "Saved seed-based stratified 70/15/15 split after exact duplicate removal",
    }
    result.update(
        {
            "model_type": model_type,
            "model_file": model_path.name,
            "class_names": labels,
            "test_split": metadata["test_split_method"],
            "metadata": metadata,
            "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
            "evaluation_device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        }
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_png(result["confusion_matrix"], REPORTS_DIR / "confusion_matrix.png")
    (REPORTS_DIR / "classification_report.json").write_text(
        json.dumps(
            {"model_type": model_type, "metadata": metadata, "report": result["per_class"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "class_names": labels,
                "matrix": result["confusion_matrix"],
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    existing = load_metrics(METRICS_PATH)
    existing_models = existing.get("models", {}) if existing.get("status") == "evaluated" else {}
    if not isinstance(existing_models, dict):
        raise ValueError(f"Existing evaluation report has an invalid models object: {METRICS_PATH}")
    existing_models[model_type] = result
    stored = {
        "status": "evaluated",
        "models": existing_models,
        "updated_model": model_type,
        "updated_at": evaluated_at,
        "test_split": metadata["test_split_method"],
    }
    save_metrics(stored, METRICS_PATH)
    save_metrics(stored, REPORTS_DIR / "evaluation_metrics.json")
    save_model_comparison(existing_models, ARTIFACTS_DIR / "model_comparison.json")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
