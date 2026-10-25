from __future__ import annotations

import json

import numpy as np
from PIL import Image

from src.cropshield import train as train_module
from src.cropshield.evaluate import evaluate_model
from src.cropshield.model import load_labels, load_model
from src.cropshield.predict import predict_image_file


def _make_tiny_dataset(root):
    rng = np.random.default_rng(23)
    for class_index, class_name in enumerate(("Leaf___healthy", "Leaf___rust")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(8):
            image = np.zeros((40, 40, 3), dtype=np.uint8)
            image[:, :, class_index] = 120 + image_index
            image += rng.integers(0, 20, image.shape, dtype=np.uint8)
            Image.fromarray(image).save(class_dir / f"{image_index}.png")


def test_baseline_training_and_held_out_evaluation_end_to_end(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    data_dir = tmp_path / "data"
    dataset_root.mkdir()
    models_dir.mkdir()
    artifacts_dir.mkdir()
    reports_dir.mkdir()
    data_dir.mkdir()
    _make_tiny_dataset(dataset_root)
    monkeypatch.setattr(train_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(train_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(train_module, "DATA_DIR", data_dir)

    training_result = train_module.train_model(
        dataset_root=dataset_root,
        model_type="baseline",
        image_size=(32, 32),
        batch_size=4,
        epochs=1,
        learning_rate=1e-3,
    )
    assert training_result["split_counts"] == {"train": 10, "validation": 2, "test": 4}
    assert training_result["confidence_policy"]["validation_count"] == 2
    assert (models_dir / "baseline_best.keras").exists()
    assert (artifacts_dir / "dataset_split.csv").exists()
    assert (data_dir / "split_manifest.csv").exists()
    with (reports_dir / "training_history.json").open(encoding="utf-8") as history_file:
        saved_history = json.load(history_file)
    assert saved_history["epochs_completed"] == 1
    assert saved_history["best_epoch"] == 1
    assert saved_history["early_stopping"]["enabled"] is True
    with (models_dir / "baseline_config.json").open(encoding="utf-8") as config_file:
        saved_config = json.load(config_file)
    assert saved_config["confidence_policy"] == training_result["confidence_policy"]

    model = load_model(models_dir / "baseline_best.keras")
    labels = load_labels(models_dir / "class_labels.json")
    _train_ds, _validation_ds, test_ds, split_labels, _mapping, _splits = train_module.build_datasets(
        dataset_root,
        image_size=(32, 32),
        batch_size=4,
    )
    assert labels == split_labels
    result = evaluate_model(model, test_ds, labels)
    assert result["image_count"] == 4
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["latency_ms_per_image"] >= 0.0
    prediction = predict_image_file(
        dataset_root / "Leaf___healthy" / "0.png",
        model_path=models_dir / "baseline_best.keras",
        labels_path=models_dir / "class_labels.json",
        model=model,
    )
    assert prediction["status"] == "ok"
    assert len(prediction["top_predictions"]) == 2
