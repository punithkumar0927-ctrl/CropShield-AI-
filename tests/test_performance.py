import json

import pandas as pd

from scripts.evaluate_model import _load_saved_test_records
from src.cropshield.performance import (
    evaluation_is_valid,
    get_evaluation_state,
    load_json_object,
)


def _valid_evaluation():
    return {
        "status": "evaluated",
        "models": {
            "baseline": {
                "accuracy": 0.75,
                "macro_precision": 0.7,
                "macro_recall": 0.7,
                "macro_f1": 0.7,
                "weighted_f1": 0.75,
                "image_count": 4,
                "confusion_matrix": [[2, 0], [1, 1]],
                "per_class": {"healthy": {"f1-score": 0.8}, "rust": {"f1-score": 0.6}},
            }
        },
    }


def test_load_json_object_reads_valid_object(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(_valid_evaluation()), encoding="utf-8")

    assert load_json_object(path)["status"] == "evaluated"


def test_load_json_object_rejects_corrupted_and_non_object_json(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text("{broken", encoding="utf-8")
    try:
        load_json_object(path)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Malformed evaluation JSON must not be accepted.")

    path.write_text("[]", encoding="utf-8")
    try:
        load_json_object(path)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-object JSON document must not be accepted.")


def test_evaluation_validation_requires_real_shaped_metrics():
    assert evaluation_is_valid(_valid_evaluation())
    assert not evaluation_is_valid({"status": "not_evaluated", "models": {}})
    invalid = _valid_evaluation()
    invalid["models"]["baseline"]["confusion_matrix"] = [[1, 2], [0]]
    assert not evaluation_is_valid(invalid)


def test_page_state_not_trained_when_model_is_missing(tmp_path):
    metrics = tmp_path / "evaluation.json"
    metrics.write_text(json.dumps(_valid_evaluation()), encoding="utf-8")

    state, path, error = get_evaluation_state([tmp_path / "missing.keras"], [metrics])

    assert (state, path, error) == ("not_trained", None, None)


def test_page_state_trained_but_not_evaluated_when_results_are_missing(tmp_path):
    model = tmp_path / "model.keras"
    model.touch()

    state, path, error = get_evaluation_state([model], [tmp_path / "missing.json"])

    assert (state, path, error) == ("trained_not_evaluated", None, None)


def test_page_state_trained_but_not_evaluated_for_corrupt_results(tmp_path):
    model = tmp_path / "model.keras"
    metrics = tmp_path / "evaluation.json"
    model.touch()
    metrics.write_text("{corrupt", encoding="utf-8")

    state, path, error = get_evaluation_state([model], [metrics])

    assert state == "trained_not_evaluated"
    assert path is None
    assert error


def test_page_state_evaluated_only_for_valid_saved_metrics(tmp_path):
    model = tmp_path / "model.keras"
    metrics = tmp_path / "evaluation.json"
    model.touch()
    metrics.write_text(json.dumps(_valid_evaluation()), encoding="utf-8")

    state, path, error = get_evaluation_state([model], [metrics])

    assert (state, path, error) == ("evaluated", metrics, None)


def test_page_state_uses_next_valid_evaluation_location(tmp_path):
    model = tmp_path / "model.keras"
    broken = tmp_path / "reports.json"
    valid = tmp_path / "artifacts.json"
    model.touch()
    broken.write_text("{broken", encoding="utf-8")
    valid.write_text(json.dumps(_valid_evaluation()), encoding="utf-8")

    state, path, error = get_evaluation_state([model], [broken, valid])

    assert (state, path, error) == ("evaluated", valid, None)


def test_missing_confusion_matrix_does_not_hide_overall_evaluation():
    saved = _valid_evaluation()
    del saved["models"]["baseline"]["confusion_matrix"]

    assert evaluation_is_valid(saved)


def test_evaluation_reads_only_the_saved_test_rows(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_image = dataset_root / "healthy" / "train.png"
    test_image = dataset_root / "healthy" / "test.png"
    train_image.parent.mkdir(parents=True)
    train_image.touch()
    test_image.touch()
    manifest_path = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"path": str(train_image), "label": "healthy", "split": "train"},
            {"path": str(test_image), "label": "healthy", "split": "test"},
        ]
    ).to_csv(manifest_path, index=False)

    records = _load_saved_test_records(manifest_path, dataset_root, ["healthy"])

    assert records == [{"path": str(test_image.resolve()), "label": "healthy"}]
