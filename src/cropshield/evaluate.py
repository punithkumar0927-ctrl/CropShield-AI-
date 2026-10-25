from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from .config import ARTIFACTS_DIR, METRICS_PATH


def safe_accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(metrics.accuracy_score(y_true, y_pred))


def compute_classification_report(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict[str, Any]:
    if not labels or len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Evaluation requires labels and equal, non-empty true and predicted values.")
    report = metrics.classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    precision = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": safe_accuracy_score(y_true, y_pred),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix": metrics.confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
    }


def evaluate_model(model: Any, dataset: Any, labels: List[str]) -> Dict[str, Any]:
    """Evaluate once on the held-out test dataset and measure local inference latency."""
    true_values: list[int] = []
    predicted_values: list[int] = []
    probability_values: list[np.ndarray] = []
    elapsed_seconds = 0.0
    image_count = 0
    for images, batch_labels in dataset:
        start = time.perf_counter()
        model_output = model(images, training=False)
        probabilities = np.asarray(
            model_output.numpy() if hasattr(model_output, "numpy") else model_output
        )
        elapsed_seconds += time.perf_counter() - start
        true_values.extend(np.asarray(batch_labels).astype(int).tolist())
        predicted_values.extend(np.argmax(probabilities, axis=1).astype(int).tolist())
        probability_values.append(probabilities)
        image_count += len(batch_labels)
    if not image_count:
        raise ValueError("The held-out test dataset is empty.")
    result = compute_classification_report(true_values, predicted_values, labels)
    result["image_count"] = image_count
    result["inference_seconds"] = round(elapsed_seconds, 6)
    result["latency_ms_per_image"] = round(elapsed_seconds * 1000.0 / image_count, 4)
    if len(labels) > 1 and probability_values:
        probabilities = np.concatenate(probability_values)
        try:
            if len(labels) == 2:
                result["roc_auc_ovr_macro"] = float(
                    metrics.roc_auc_score(true_values, probabilities[:, 1])
                )
            else:
                y_true_binary = label_binarize(true_values, classes=list(range(len(labels))))
                class_auc = [
                    metrics.roc_auc_score(y_true_binary[:, index], probabilities[:, index])
                    for index in range(len(labels))
                    if np.unique(y_true_binary[:, index]).size == 2
                ]
                result["roc_auc_ovr_macro"] = float(np.mean(class_auc)) if class_auc else None
        except ValueError:
            result["roc_auc_ovr_macro"] = None
    return result


def save_model_comparison(results: Dict[str, Any], path: str | Path | None = None) -> None:
    """Persist side-by-side test metrics; this report never chooses a model."""
    output_path = Path(path) if path is not None else ARTIFACTS_DIR / "model_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "selection_basis": "Models are reported side-by-side; test results are not used for model selection.",
        "models": {
            name: {
                key: value
                for key, value in result.items()
                if key
                in {
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "latency_ms_per_image",
                }
            }
            for name, result in results.items()
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)


def save_metrics(metrics_dict: Dict[str, Any], path: str | Path = METRICS_PATH) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_dict, handle, indent=2)


def save_confusion_matrix_png(
    confusion_matrix: list[list[int]],
    path: str | Path,
) -> None:
    """Write a dependency-light heatmap; matrix row/column order follows saved class labels."""
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("Confusion matrix must be a non-empty square matrix.")
    maximum = float(matrix.max())
    normalized = matrix / maximum if maximum > 0 else matrix
    cell_size = max(12, min(28, 900 // matrix.shape[0]))
    left_margin = 150
    top_margin = 80
    bottom_margin = 130
    width = left_margin + matrix.shape[1] * cell_size + 20
    height = top_margin + matrix.shape[0] * cell_size + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left_margin, 20), "Confusion matrix (class index follows saved label order)", fill="#17231B")
    draw.text((left_margin, 45), "Rows: actual class    Columns: predicted class", fill="#647067")
    for row in range(matrix.shape[0]):
        y0 = top_margin + row * cell_size
        for column in range(matrix.shape[1]):
            x0 = left_margin + column * cell_size
            value = float(normalized[row, column])
            shade = int(250 - value * 190)
            color = (shade, min(255, shade + int(value * 20)), shade)
            draw.rectangle((x0, y0, x0 + cell_size - 1, y0 + cell_size - 1), fill=color)
        if matrix.shape[0] <= 50:
            draw.text((5, y0 + max(0, (cell_size - 10) // 2)), str(row), fill="#17231B")
    for column in range(matrix.shape[1]):
        if column % max(1, matrix.shape[1] // 12) == 0:
            x = left_margin + column * cell_size + cell_size // 2
            draw.text((x, top_margin + matrix.shape[0] * cell_size + 5), str(column), fill="#17231B")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def load_metrics(path: str | Path = METRICS_PATH) -> Dict[str, Any]:
    output_path = Path(path)
    if not output_path.exists():
        return {"status": "not_evaluated"}
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_confidence_summary(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": safe_accuracy_score(y_true, y_pred),
        "error_rate": float(1.0 - safe_accuracy_score(y_true, y_pred)),
    }


def select_confidence_threshold(
    y_true: List[int],
    probabilities: np.ndarray,
    target_accuracy: float = 0.90,
) -> Dict[str, Any]:
    """Select maximum validation coverage while meeting the requested accepted-set accuracy."""
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(y_true) or scores.shape[1] < 2 or not y_true:
        raise ValueError("Threshold selection requires matching non-empty labels and probability rows.")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be greater than 0 and at most 1.")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Threshold selection requires finite probabilities between 0 and 1.")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Threshold selection requires normalized probability rows.")
    true_values = np.asarray(y_true, dtype=int)
    predicted_values = np.argmax(scores, axis=1)
    if np.any(true_values < 0) or np.any(true_values >= scores.shape[1]):
        raise ValueError("True labels must map to model output indices.")

    confidences = np.max(scores, axis=1)
    best_threshold = None
    best_count = 0
    best_accuracy = 0.0
    for threshold in np.unique(confidences):
        accepted = confidences >= threshold
        accepted_count = int(np.count_nonzero(accepted))
        accepted_accuracy = float(np.mean(predicted_values[accepted] == true_values[accepted]))
        if accepted_accuracy >= target_accuracy and accepted_count > best_count:
            best_threshold = float(threshold)
            best_count = accepted_count
            best_accuracy = accepted_accuracy

    return {
        "status": "calibrated" if best_threshold is not None else "target_not_met",
        "threshold": best_threshold,
        "target_accepted_accuracy": target_accuracy,
        "validation_accepted_accuracy": best_accuracy if best_threshold is not None else None,
        "validation_coverage": best_count / len(y_true) if best_threshold is not None else 0.0,
        "validation_count": len(y_true),
        "selection_method": "maximum validation coverage subject to target accepted-set accuracy",
    }
