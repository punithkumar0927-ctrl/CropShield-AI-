from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cropshield.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR
from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.model import load_labels
from src.cropshield.performance import get_evaluation_state, load_json_object

# Clearly labeled illustrative values, shown only when no real evaluation exists yet.
# These are NOT measured results; they sketch the layout of a typical outcome for this model family.
SAMPLE_EVALUATION_RESULTS = {
    "Accuracy": "91.0%",
    "Precision (macro)": "89.0%",
    "Recall (macro)": "88.0%",
    "F1-Score (macro)": "88.5%",
}


def render_sample_evaluation_block() -> None:
    """Show a clearly labeled 'Sample Evaluation Results' preview when real metrics are unavailable."""
    with st.expander("Sample Evaluation Results (illustrative placeholder)", expanded=False):
        st.warning(
            "These are sample placeholder values, NOT real measurements of this model. "
            "Run `python scripts/evaluate_model.py` to replace them with real held-out test results."
        )
        sample_columns = st.columns(4)
        for column, (label, value) in zip(sample_columns, SAMPLE_EVALUATION_RESULTS.items()):
            column.metric(label, value)

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #F7FAF5; }
    h1, h2, h3 { color: #166534; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #d8e5d6;
        padding: 12px; border-radius: 12px;
    }
    p, li, label { color: #17231B; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Model Performance")

if st.button("Refresh evaluation results"):
    st.rerun()

model_candidates = [
    MODELS_DIR / name
    for name in (
        "crop_model.keras",
        "crop_model.h5",
        "cropshield_model.keras",
        "cropshield_model.h5",
        "baseline_best.keras",
        "mobilenet_best.keras",
    )
]
evaluation_candidates = [
    REPORTS_DIR / "evaluation_metrics.json",
    ARTIFACTS_DIR / "evaluation_metrics.json",
    MODELS_DIR / "evaluation_results.json",
]
state, metrics_path, evaluation_error = get_evaluation_state(
    model_candidates, evaluation_candidates
)

if state == "not_trained":
    st.warning("No trained CropShield model is available. Please train the model first.")
    st.code("python scripts/train_model.py", language="powershell")
    render_sample_evaluation_block()
    stop_with_footer()

if state == "trained_not_evaluated":
    st.info("Trained, not evaluated")
    st.warning("Model is trained, but held-out test evaluation has not been completed yet.")
    if evaluation_error:
        st.error("Evaluation file is missing or corrupted. Please rerun scripts/evaluate_model.py.")
        st.caption(evaluation_error)
    st.code("python scripts/evaluate_model.py", language="powershell")
    st.info(
        "Evaluation calculates real test-set performance after training. "
        "It does not run automatically when this page opens."
    )
    render_sample_evaluation_block()
    stop_with_footer()

assert metrics_path is not None
try:
    saved = load_json_object(metrics_path)
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error("Evaluation file is missing or corrupted. Please rerun scripts/evaluate_model.py.")
    st.caption(str(error))
    stop_with_footer()

stored_models = saved.get("models", {})
if not isinstance(stored_models, dict) or not stored_models:
    st.error("Evaluation file is missing or corrupted. Please rerun scripts/evaluate_model.py.")
    stop_with_footer()

trained_model_names = {
    path.stem.replace("_best", "")
    for path in model_candidates
    if path.is_file() and path.stem.endswith("_best")
}
active_model = saved.get("updated_model")
generic_model_available = any(
    (MODELS_DIR / name).is_file()
    for name in ("crop_model.keras", "crop_model.h5", "cropshield_model.keras", "cropshield_model.h5")
)
if active_model and generic_model_available:
    trained_model_names.add(str(active_model))
elif generic_model_available and len(stored_models) == 1:
    trained_model_names.update(str(name) for name in stored_models)
selected_models = {
    name: result
    for name, result in stored_models.items()
    if name in trained_model_names and isinstance(result, dict)
}
if not selected_models:
    st.error(
        "Saved evaluation results do not match an available trained model. "
        "Please rerun scripts/evaluate_model.py."
    )
    stop_with_footer()

st.caption(f"Loaded real saved evaluation results from `{metrics_path.relative_to(metrics_path.parents[1])}`.")
st.info(
    "The test split is reserved for final reporting. Model choices should be made using validation data, "
    "not these test-set results. Accuracy alone can hide weak performance on minority disease classes."
)

model_names = list(selected_models)
selected_model = st.selectbox("Evaluated model", model_names)
result = selected_models[selected_model]
confusion_document: dict = {}
confusion_json_path = REPORTS_DIR / "confusion_matrix.json"
if confusion_json_path.is_file():
    try:
        confusion_document = load_json_object(confusion_json_path)
        if confusion_document.get("model_type") not in (None, selected_model):
            confusion_document = {}
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"Saved confusion-matrix JSON could not be loaded: {error}")
label_paths = [
    MODELS_DIR / f"{selected_model}_class_labels.json",
    MODELS_DIR / "class_labels.json",
    MODELS_DIR / "class_names.json",
]
class_labels: list[str] = []
for label_path in label_paths:
    try:
        class_labels = load_labels(label_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"Could not load class mapping from {label_path.name}: {error}")
        continue
    if class_labels:
        break
if not class_labels:
    class_labels = result.get("class_names", [])
if not class_labels:
    class_labels = confusion_document.get("class_names", [])
class_labels = [str(label) for label in class_labels] if isinstance(class_labels, list) else []

matrix = result.get("confusion_matrix") or confusion_document.get("matrix")
per_class = result.get("per_class")
if not isinstance(per_class, dict):
    classification_path = REPORTS_DIR / "classification_report.json"
    if classification_path.is_file():
        try:
            classification_document = load_json_object(classification_path)
            if classification_document.get("model_type") in (None, selected_model):
                per_class = classification_document.get("report", {})
        except (OSError, json.JSONDecodeError, ValueError) as error:
            st.warning(f"Saved classification report could not be loaded: {error}")
matrix_valid = (
    isinstance(matrix, list)
    and bool(matrix)
    and all(isinstance(row, list) and len(row) == len(matrix) for row in matrix)
    and all(
        isinstance(value, (int, float)) and value >= 0
        for row in matrix
        for value in row
    )
)
class_mapping_error = None
if not class_labels:
    class_mapping_error = "The model's class mapping is missing or unreadable."
reported_labels = result.get("class_names")
if class_labels and isinstance(reported_labels, list) and reported_labels != class_labels:
    class_mapping_error = "Evaluation class names or order do not match the saved model mapping."
elif matrix_valid and len(class_labels) != len(matrix):
    class_mapping_error = "The evaluation class count does not match the saved model mapping."
if class_mapping_error:
    st.error(class_mapping_error)
    st.error("Evaluation file is missing or corrupted. Please rerun scripts/evaluate_model.py.")
    stop_with_footer()
st.success("Evaluated")

metadata = result.get("metadata", {})
evaluated_at = metadata.get("evaluated_at") or result.get("evaluated_at") or saved.get("updated_at", "Not recorded")
overall_columns = st.columns(4)
overall_values = [
    ("Accuracy", result.get("accuracy"), "{:.3%}"),
    ("Macro precision", result.get("macro_precision"), "{:.3%}"),
    ("Macro recall", result.get("macro_recall"), "{:.3%}"),
    ("Macro F1-score", result.get("macro_f1"), "{:.3%}"),
    ("Weighted F1-score", result.get("weighted_f1"), "{:.3%}"),
    ("Test images", result.get("image_count"), "{}"),
    ("Classes", len(class_labels) if class_labels else len(matrix) if matrix_valid else "Unknown", "{}"),
    ("Evaluated", evaluated_at, "{}"),
]
for index, (label, value, format_string) in enumerate(overall_values):
    with overall_columns[index % len(overall_columns)]:
        display = format_string.format(value) if isinstance(value, (int, float)) else str(value)
        st.metric(label, display)

with st.expander("Evaluation protocol and metadata"):
    st.write(
        {
            key: metadata.get(key)
            for key in (
                "model_file",
                "test_image_count",
                "number_of_classes",
                "image_size",
                "batch_size",
                "random_seed",
                "dataset_path",
                "test_split_file",
                "test_split_method",
            )
            if metadata.get(key) is not None
        }
        or {"test_split": result.get("test_split", saved.get("test_split", "Held-out test split"))}
    )

st.subheader("Per-class performance")
class_rows = []
if class_labels and isinstance(per_class, dict) and matrix_valid and len(class_labels) == len(matrix):
    for index, class_name in enumerate(class_labels):
        values = per_class.get(class_name)
        if not isinstance(values, dict):
            values = per_class.get(str(index), {})
        if not isinstance(values, dict):
            continue
        support = int(values.get("support", sum(matrix[index])))
        correct = int(matrix[index][index])
        class_rows.append(
            {
                "Class name": class_name,
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1-score": values.get("f1-score", values.get("f1_score")),
                "Support": support,
                "Correct predictions": correct,
                "Total test images": support,
            }
        )
if class_rows:
    class_frame = pd.DataFrame(class_rows)
    scored_classes = class_frame.dropna(subset=["F1-score"])
    best_index = scored_classes["F1-score"].idxmax() if not scored_classes.empty else None
    weakest_index = scored_classes["F1-score"].idxmin() if not scored_classes.empty else None

    def highlight_class(row):
        if row.name == best_index == weakest_index:
            color = "background-color: #dcefdc"
        elif row.name == best_index:
            color = "background-color: #dcefdc"
        elif row.name == weakest_index:
            color = "background-color: #f8e4dc"
        else:
            color = ""
        return [color] * len(row)

    # --- Classification report: per-class rows plus real macro/weighted average rows from the saved report. ---
    report_frame = class_frame[["Class name", "Precision", "Recall", "F1-score", "Support"]].copy()
    report_frame = report_frame.rename(columns={"Class name": "Class"})
    for average_name in ("macro avg", "weighted avg"):
        average_values = per_class.get(average_name) if isinstance(per_class, dict) else None
        if isinstance(average_values, dict):
            report_frame.loc[len(report_frame)] = {
                "Class": average_name,
                "Precision": average_values.get("precision"),
                "Recall": average_values.get("recall"),
                "F1-score": average_values.get("f1-score", average_values.get("f1_score")),
                "Support": average_values.get("support", result.get("image_count")),
            }
    st.markdown("**Classification report**")
    st.dataframe(
        report_frame.style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Per-class results (best and weakest class highlighted)**")

    st.dataframe(
        class_frame.style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
        ).apply(highlight_class, axis=1),
        width="stretch",
        hide_index=True,
    )
    valid_f1 = class_frame.dropna(subset=["F1-score"])
    if not valid_f1.empty:
        best = valid_f1.loc[valid_f1["F1-score"].idxmax()]
        weakest = valid_f1.loc[valid_f1["F1-score"].idxmin()]
        st.caption(
            f"Best class by F1: {best['Class name']} ({best['F1-score']:.3f}); "
            f"weakest class by F1: {weakest['Class name']} ({weakest['F1-score']:.3f})."
        )
elif class_labels:
    st.info("Per-class results are not included in this evaluation file.")

st.subheader("Confusion matrix")
if matrix_valid and class_labels and len(class_labels) == len(matrix):
    normalized = st.checkbox("Normalize each true-class row", value=False)
    numeric_matrix = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    if normalized:
        row_totals = numeric_matrix.sum(axis=1).replace(0, 1)
        numeric_matrix = numeric_matrix.div(row_totals, axis=0)
    figure = px.imshow(
        numeric_matrix,
        x=class_labels,
        y=class_labels,
        labels={"x": "Predicted class", "y": "True class", "color": "Proportion" if normalized else "Images"},
        aspect="auto",
        title=f"{selected_model.replace('_', ' ').title()} held-out test confusion matrix",
        color_continuous_scale="Greens",
    )
    figure.update_layout(xaxis_tickangle=-75, height=650)
    st.plotly_chart(figure, width="stretch")
    confusion_image = REPORTS_DIR / "confusion_matrix.png"
    if confusion_image.is_file() and not normalized:
        st.caption("Also saved by the evaluation script:")
        st.image(str(confusion_image), use_container_width=True)
else:
    st.info("Confusion-matrix data is missing or does not match the class mapping.")

st.subheader("Training history")
history_path = REPORTS_DIR / "training_history.json"
csv_history_path = ARTIFACTS_DIR / f"{selected_model}_training_history.csv"
history_epochs: dict[str, list] = {}
history_metadata: dict = {}
if csv_history_path.is_file():
    history_frame = pd.read_csv(csv_history_path)
    history_epochs = {column: history_frame[column].dropna().tolist() for column in history_frame}
if history_path.is_file():
    try:
        history_document = load_json_object(history_path)
        if history_document.get("model_type") in (None, selected_model):
            history_metadata = history_document
            if not history_epochs:
                epochs = history_document.get("epochs", {})
                if isinstance(epochs, dict):
                    history_epochs = epochs
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"Training history could not be loaded: {error}")
if history_epochs:
    history_frame = pd.DataFrame(history_epochs)
    history_frame.index = history_frame.index + 1
    history_frame.index.name = "Epoch"
    if {"loss", "val_loss"}.issubset(history_frame.columns):
        st.plotly_chart(
            px.line(history_frame, y=["loss", "val_loss"], markers=True, title="Training and validation loss"),
            width="stretch",
        )
    if {"accuracy", "val_accuracy"}.issubset(history_frame.columns):
        st.plotly_chart(
            px.line(
                history_frame,
                y=["accuracy", "val_accuracy"],
                markers=True,
                title="Training and validation accuracy",
            ),
            width="stretch",
        )
    st.caption(f"Epochs completed: {len(history_frame)}.")
    validation_losses = history_frame.get("val_loss")
    if validation_losses is not None and not validation_losses.empty:
        best_epoch = history_metadata.get("best_epoch")
        if not isinstance(best_epoch, int):
            best_epoch = int(validation_losses.idxmin())
        st.caption(f"Best epoch by validation loss: {best_epoch}.")
    early_stopping = history_metadata.get("early_stopping")
    if isinstance(early_stopping, dict):
        stop_result = (
            f"Stopped early at epoch {early_stopping['stopped_epoch']}."
            if early_stopping.get("stopped_early") and early_stopping.get("stopped_epoch")
            else "Reached the configured epoch limit."
        )
        st.caption(
            f"Early stopping: {stop_result} "
            f"(monitor: {early_stopping.get('monitor', 'not recorded')}, "
            f"patience: {early_stopping.get('patience', 'not recorded')})."
        )
    elif history_path.is_file():
        st.caption("Early-stopping outcome was not recorded in this training-history artifact.")
else:
    st.info("Training history is not available for this model.")

st.subheader("Calibration and trust")
calibration_path = REPORTS_DIR / "calibration_metrics.json"
if calibration_path.is_file():
    try:
        calibration = load_json_object(calibration_path)
        ece = calibration.get("expected_calibration_error", calibration.get("ece"))
        if isinstance(ece, (int, float)):
            st.metric("Expected Calibration Error (ECE)", f"{ece:.4f}")
        calibration_image = REPORTS_DIR / "reliability_diagram.png"
        if calibration_image.is_file():
            st.image(str(calibration_image), caption="Saved reliability diagram", use_container_width=True)
        for key, title in (
            ("confidence_distribution", "Confidence distribution"),
            ("accuracy_by_confidence_bin", "Accuracy by confidence bin"),
        ):
            bins = calibration.get(key)
            if isinstance(bins, list) and bins:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(bins), width="stretch", hide_index=True)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"Calibration file could not be loaded: {error}")
else:
    st.info("Calibration was not evaluated in this run.")

for title, filename, unavailable_message, metric_keys in (
    (
        "Robustness results",
        "robustness_metrics.json",
        "Robustness evaluation has not been completed yet.",
        ("clean_accuracy", "corrupted_accuracy", "performance_drop"),
    ),
    (
        "Out-of-distribution results",
        "ood_metrics.json",
        "Out-of-distribution evaluation has not been completed yet.",
        ("auroc", "detection_rate", "false_positive_rate"),
    ),
):
    st.subheader(title)
    result_path = REPORTS_DIR / filename
    if not result_path.is_file():
        st.info(unavailable_message)
        continue
    try:
        extra_results = load_json_object(result_path)
        values = {key: extra_results[key] for key in metric_keys if key in extra_results}
        if values:
            st.json(values)
        detail_rows = (
            extra_results.get("by_corruption")
            or extra_results.get("results_by_corruption")
            or extra_results.get("coverage_vs_accuracy")
        )
        if isinstance(detail_rows, list) and detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
        rejected_examples = extra_results.get("rejected_examples")
        if isinstance(rejected_examples, list) and rejected_examples:
            st.markdown("**Examples of rejected uncertain predictions**")
            st.dataframe(pd.DataFrame(rejected_examples), width="stretch", hide_index=True)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"{filename} could not be loaded: {error}")

st.subheader("Known limitations")
st.warning(
    "High accuracy on PlantVillage does not guarantee the same performance on real farm photographs. "
    "Real-world images may have cluttered backgrounds, varying lighting, partial leaves, shadows, "
    "camera noise, and different crop varieties."
)
if class_labels:
    crops = sorted({label.split("___", maxsplit=1)[0] for label in class_labels})
    st.markdown(f"**Supported crop labels in this evaluated model:** {', '.join(crops)}.")
    with st.expander("Supported disease classes"):
        st.write(class_labels)
    st.markdown("**Unsupported crops and diseases:** Not enumerated by the current evaluation dataset.")
else:
    st.markdown("**Supported and unsupported classes:** Cannot verify without a valid class mapping.")
st.markdown(
    "- **Known failure cases:** Not measured by the current held-out evaluation; field-photo failure analysis is needed.\n"
    "- **Dataset limitations:** PlantVillage-style images may not represent real-world field conditions; related leaves may cross image-level splits.\n"
    "- **Ethical considerations:** Treat predictions as informational, seek expert review for consequential decisions, and do not use model output alone to choose pesticide or dosage."
)

json_payload = json.dumps(saved, indent=2)
st.download_button(
    "Download evaluation JSON",
    data=json_payload,
    file_name="evaluation_metrics.json",
    mime="application/json",
)
if class_rows:
    st.download_button(
        "Download per-class CSV",
        data=pd.DataFrame(class_rows).to_csv(index=False),
        file_name=f"{selected_model}_per_class_metrics.csv",
        mime="text/csv",
    )

render_about_footer()
