from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    MODELS_DIR,
    REPORTS_DIR,
)
from .dataset import collect_image_records, detect_dataset_root, iter_class_dirs, stratified_split
from .evaluate import select_confidence_threshold
from .model import build_baseline_cnn, build_mobilenet_transfer, save_labels, save_model, set_seeds

LOGGER = logging.getLogger(__name__)


def ensure_supported_layout(dataset_root: Path) -> list[Path]:
    class_dirs = iter_class_dirs(dataset_root)
    if not class_dirs:
        raise ValueError(
            "Unsupported dataset layout. Expected a folder per class with JPG, JPEG, or PNG images."
        )
    return class_dirs


def _make_tf_dataset(
    records: list[dict[str, object]],
    label_to_index: dict[str, int],
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [str(record["path"]) for record in records]
    labels = [label_to_index[str(record["label"])] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, method="bilinear")
        return tf.cast(image, tf.float32), label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_dataset_from_records(
    records: list[dict[str, object]],
    labels: list[str],
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int = DEFAULT_SEED,
) -> tf.data.Dataset:
    """Build an unshuffled dataset from an explicit saved split manifest."""
    if not records:
        raise ValueError("Cannot build a dataset from an empty split.")
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_labels = sorted({str(record["label"]) for record in records} - set(label_to_index))
    if unknown_labels:
        raise ValueError(f"Split manifest contains labels absent from the model mapping: {unknown_labels}.")
    return _make_tf_dataset(
        records,
        label_to_index,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )


def build_datasets(
    dataset_root: Path,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    class_names: list[str] | None = None,
):
    ensure_supported_layout(dataset_root)
    records = collect_image_records(dataset_root, class_names=class_names)
    splits = stratified_split(records, seed=seed)
    labels = sorted({str(record["label"]) for record in records})
    if len(labels) < 2:
        raise ValueError("Training requires at least two selected classes.")
    class_to_index = {label: index for index, label in enumerate(labels)}
    datasets = {
        name: (
            _make_tf_dataset(
                split_records,
                class_to_index,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            )
            if name == "train"
            else build_dataset_from_records(
                split_records,
                labels,
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
            )
        )
        for name, split_records in splits.items()
    }
    return datasets["train"], datasets["validation"], datasets["test"], labels, class_to_index, splits


def train_model(
    dataset_root: Optional[str | Path] = None,
    model_type: str = "mobilenet",
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    class_names: list[str] | None = None,
):
    if dataset_root is not None:
        provided_path = Path(dataset_root).expanduser()
        dataset_path = detect_dataset_root(provided_path)
        if dataset_path is None:
            raise ValueError(
                f"Unsupported dataset layout at {provided_path}. Expected class folders directly or under raw/color, color, PlantVillage, or train."
            )
    else:
        dataset_path = detect_dataset_root()
    if dataset_path is None:
        raise FileNotFoundError(
            "No PlantVillage dataset was found. Download it or point --dataset-root to an extracted folder-per-class dataset."
        )
    dataset_path = dataset_path.resolve()
    if model_type not in {"baseline", "mobilenet"}:
        raise ValueError("model_type must be 'baseline' or 'mobilenet'.")
    if batch_size < 1 or epochs < 1 or learning_rate <= 0 or min(image_size) < 1:
        raise ValueError("batch_size, epochs, learning_rate, and image dimensions must be positive.")
    if model_type == "mobilenet" and min(image_size) < 32:
        raise ValueError("MobileNetV2 requires image dimensions of at least 32 pixels.")
    set_seeds(DEFAULT_SEED)
    train_ds, val_ds, _test_ds, labels, class_to_index, splits = build_datasets(
        dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
    )
    num_classes = len(labels)
    if model_type == "baseline":
        model = build_baseline_cnn(
            num_classes=num_classes,
            input_shape=(*image_size, 3),
            learning_rate=learning_rate,
        )
    else:
        model = build_mobilenet_transfer(
            num_classes=num_classes,
            input_shape=(*image_size, 3),
            learning_rate=learning_rate,
        )

    train_counts = np.bincount(
        [class_to_index[str(record["label"])] for record in splits["train"]],
        minlength=num_classes,
    )
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.asarray([class_to_index[str(record["label"])] for record in splits["train"]]),
    )
    class_weights = {index: float(weight) for index, weight in enumerate(class_weights_values)}
    if int(train_counts.max()) < 2 * int(train_counts.min()):
        class_weights = None

    with tempfile.TemporaryDirectory(prefix=f".{model_type}-", dir=MODELS_DIR) as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "best.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            shuffle=False,
            verbose=1,
        )
        validation_losses = np.asarray(history.history.get("val_loss", []), dtype=np.float64)
        if not np.any(np.isfinite(validation_losses)) or not checkpoint_path.exists():
            raise RuntimeError("Training did not produce a valid best checkpoint from validation data.")
        best_model = tf.keras.models.load_model(checkpoint_path)

    validation_labels: list[int] = []
    validation_probabilities: list[np.ndarray] = []
    for images, batch_labels in val_ds:
        outputs = best_model(images, training=False)
        validation_probabilities.append(np.asarray(outputs.numpy()))
        validation_labels.extend(np.asarray(batch_labels).astype(int).tolist())
    if not validation_probabilities:
        raise RuntimeError("Validation data was empty; confidence threshold cannot be calibrated.")
    confidence_policy = select_confidence_threshold(
        validation_labels,
        np.concatenate(validation_probabilities, axis=0),
    )

    model_path = MODELS_DIR / f"{model_type}_best.keras"
    save_model(best_model, model_path)
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(ARTIFACTS_DIR / f"{model_type}_training_history.csv", index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "training_history.json").open("w", encoding="utf-8") as history_file:
        json.dump(
            {
                "model_type": model_type,
                "epochs": history.history,
                "epochs_completed": len(history.epoch),
                "best_epoch": int(np.argmin(validation_losses)) + 1,
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "stopped_early": bool(callbacks[0].stopped_epoch > 0),
                    "stopped_epoch": int(callbacks[0].stopped_epoch) + 1
                    if callbacks[0].stopped_epoch > 0
                    else None,
                    "monitor": "val_loss",
                    "restore_best_weights": True,
                },
            },
            history_file,
            indent=2,
        )
    split_manifest = [
        {**record, "split": split_name}
        for split_name, split_records in splits.items()
        for record in split_records
    ]
    pd.DataFrame(split_manifest).to_csv(ARTIFACTS_DIR / "dataset_split.csv", index=False)
    pd.DataFrame(split_manifest).to_csv(DATA_DIR / "split_manifest.csv", index=False)
    save_labels(labels, MODELS_DIR / f"{model_type}_class_labels.json")
    save_labels(labels, MODELS_DIR / "class_labels.json")
    save_model(best_model, MODELS_DIR / "crop_model.keras")
    shutil.copyfile(model_path, MODELS_DIR / "cropshield_model.keras")
    config = {
        "model_type": model_type,
        "model_version": f"{model_type}-seed{DEFAULT_SEED}",
        "image_size": list(image_size),
        "normalization": "raw RGB 0..255 input; model graph applies architecture-specific preprocessing",
        "class_names": labels,
        "selected_classes": labels,
        "seed": DEFAULT_SEED,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "pretrained_weights": "imagenet" if model_type == "mobilenet" else None,
        "class_weighting": class_weights is not None,
        "confidence_policy": confidence_policy,
    }
    with (MODELS_DIR / f"{model_type}_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    with (MODELS_DIR / "model_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    with (MODELS_DIR / "training_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    preprocessing_config = {
        "input_size": list(image_size),
        "color_mode": "RGB",
        "resize_method": "bilinear",
        "input_pixel_range": [0, 255],
        "architecture_preprocessing": (
            "MobileNetV2 preprocess_input is embedded in the trained model."
            if model_type == "mobilenet"
            else "No architecture-specific normalization."
        ),
    }
    with (MODELS_DIR / "preprocessing_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(preprocessing_config, config_file, indent=2)
    if model_type == "mobilenet":
        shutil.copyfile(model_path, MODELS_DIR / "cropshield_mobilenetv2.keras")
    LOGGER.info(
        "Finished training %s on %d unique valid images.",
        model_type,
        sum(len(split_records) for split_records in splits.values()),
    )
    return {
        "model_type": model_type,
        "labels": labels,
        "history": history.history,
        "best_validation_loss": float(np.min(validation_losses[np.isfinite(validation_losses)])),
        "confidence_policy": confidence_policy,
        "split_counts": {name: len(values) for name, values in splits.items()},
    }
