from src.cropshield.evaluate import (
    compute_classification_report,
    evaluate_model,
    generate_confidence_summary,
    select_confidence_threshold,
    save_model_comparison,
    save_confusion_matrix_png,
)
import pytest


class TinyModel:
    def __call__(self, images, training=False):
        import numpy as np

        return np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)


def test_classification_report():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    result = compute_classification_report(y_true, y_pred, ["leaf_healthy", "leaf_rust"])
    assert result["accuracy"] == 0.75
    assert result["macro_precision"] == pytest.approx(5 / 6)
    assert result["macro_recall"] == 0.75
    assert result["macro_f1"] == pytest.approx(0.7333333333333334)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["per_class"]["leaf_healthy"]["support"] == 2
    assert result["per_class"]["leaf_rust"]["f1-score"] == pytest.approx(2 / 3)


def test_confidence_summary():
    summary = generate_confidence_summary([1, 1, 0], [1, 1, 1])
    assert "accuracy" in summary
    assert summary["error_rate"] >= 0.0


def test_held_out_evaluation_and_model_comparison(tmp_path):
    import numpy as np

    class Batch:
        def __iter__(self):
            yield np.zeros((2, 1), dtype=np.float32), np.asarray([0, 1])

    result = evaluate_model(TinyModel(), Batch(), ["healthy", "rust"])
    assert result["accuracy"] == 1.0
    assert result["image_count"] == 2
    assert result["latency_ms_per_image"] >= 0.0
    save_model_comparison({"baseline": result}, tmp_path / "comparison.json")
    contents = (tmp_path / "comparison.json").read_text(encoding="utf-8")
    assert "not used for model selection" in contents


def test_multiclass_roc_auc_uses_per_class_one_vs_rest_scores():
    import numpy as np

    class MulticlassModel:
        def __call__(self, images, training=False):
            return np.asarray(
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                dtype=np.float32,
            )

    class Batch:
        def __iter__(self):
            yield np.zeros((3, 1), dtype=np.float32), np.asarray([0, 1, 2])

    result = evaluate_model(MulticlassModel(), Batch(), ["a", "b", "c"])
    assert result["roc_auc_ovr_macro"] == 1.0


def test_confidence_threshold_uses_validation_accuracy_and_maximum_coverage():
    import numpy as np

    result = select_confidence_threshold(
        [0, 1, 0],
        np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        target_accuracy=0.9,
    )
    assert result["status"] == "calibrated"
    assert result["threshold"] == 0.9
    assert result["validation_accepted_accuracy"] == 1.0
    assert result["validation_coverage"] == 1 / 3


def test_confidence_threshold_is_unavailable_when_target_is_not_met():
    import numpy as np

    result = select_confidence_threshold(
        [1, 1],
        np.asarray([[0.9, 0.1], [0.8, 0.2]]),
        target_accuracy=0.9,
    )
    assert result["status"] == "target_not_met"
    assert result["threshold"] is None
    assert result["validation_coverage"] == 0.0


def test_confusion_matrix_png_is_written_for_saved_reports(tmp_path):
    output_path = tmp_path / "confusion_matrix.png"

    save_confusion_matrix_png([[4, 1], [2, 3]], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
