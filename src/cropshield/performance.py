from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject malformed or unexpected top-level values."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def evaluation_is_valid(value: dict[str, Any]) -> bool:
    """Return whether a saved evaluation contains measured, structurally valid results."""
    models = value.get("models")
    if value.get("status") != "evaluated" or not isinstance(models, dict) or not models:
        return False
    for result in models.values():
        if not isinstance(result, dict):
            continue
        metrics = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        if any(not isinstance(result.get(metric), (int, float)) for metric in metrics):
            continue
        if any(not 0 <= result[metric] <= 1 for metric in metrics):
            continue
        if not isinstance(result.get("image_count"), int) or result["image_count"] < 1:
            continue
        matrix = result.get("confusion_matrix")
        per_class = result.get("per_class")
        if matrix is not None and (
            not isinstance(matrix, list)
            or not matrix
            or any(not isinstance(row, list) or len(row) != len(matrix) for row in matrix)
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for row in matrix
                for value in row
            )
        ):
            continue
        if per_class is not None and not isinstance(per_class, dict):
            continue
        return True
    return False


def get_evaluation_state(
    model_paths: list[str | Path],
    evaluation_paths: list[str | Path],
) -> tuple[str, Path | None, str | None]:
    """Resolve model/evaluation availability without treating broken files as evaluated."""
    if not any(Path(path).is_file() for path in model_paths):
        return "not_trained", None, None
    errors: list[str] = []
    for path_value in evaluation_paths:
        path = Path(path_value)
        if not path.is_file():
            continue
        try:
            saved = load_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        if evaluation_is_valid(saved):
            return "evaluated", path, None
        errors.append(f"Evaluation data in {path} is incomplete or invalid.")
    return "trained_not_evaluated", None, "; ".join(errors) if errors else None
