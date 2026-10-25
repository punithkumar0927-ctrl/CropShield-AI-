from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.cropshield.config import METRICS_PATH, MODEL_PATH, MODELS_DIR, ensure_directories
from src.cropshield.dataset import summarize_dataset
from src.cropshield.explainability import gradcam_for_prediction, overlay_heatmap
from src.cropshield.footer import render_about_footer
from src.cropshield.knowledge_base import is_healthy_class
from src.cropshield.model import load_labels, load_model
from src.cropshield.predict import predict_image_file

ensure_directories()

st.set_page_config(page_title="CropShield AI 2.0", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #F7FAF5; }
    [data-testid="stSidebar"] { background: #edf4eb; }
    h1, h2, h3 { color: #166534; }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #d8e5d6;
        padding: 16px;
        border-radius: 12px;
    }
    p, li, label { color: #17231B; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_status(model_path: str):
    model = load_model(model_path)
    return model is not None


@st.cache_resource
def get_demo_model():
    """Load the trained model and class mapping for the home-page live demo."""
    candidates = (
        MODEL_PATH,
        MODELS_DIR / "cropshield_model.keras",
        MODELS_DIR / "cropshield_model.h5",
    )
    model_path = next((path for path in candidates if path.is_file()), MODEL_PATH)
    model = load_model(model_path)
    labels = []
    try:
        labels = load_labels(MODELS_DIR / "class_labels.json")
    except (OSError, ValueError):
        labels = []
    threshold = None
    config_path = MODELS_DIR / "model_config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                policy = json.load(config_file).get("confidence_policy")
            if isinstance(policy, dict) and isinstance(policy.get("threshold"), (int, float)):
                threshold = float(policy["threshold"])
        except (OSError, ValueError):
            threshold = None
    return model, labels, threshold


@st.cache_data(ttl=1800)
def get_demo_prediction() -> dict:
    """Run one real inference + Grad-CAM on a fixed dataset sample for the live demo."""
    model, labels, threshold = get_demo_model()
    if model is None or not labels:
        return {"status": "model_unavailable"}
    dataset_root = Path(summary_dataset_root_for_demo()) if summary_dataset_root_for_demo() else None
    if dataset_root is None:
        return {"status": "dataset_missing"}
    class_dirs = sorted(
        directory
        for directory in dataset_root.iterdir()
        if directory.is_dir() and any(
            file.suffix.lower() in {".jpg", ".jpeg", ".png"} for file in directory.iterdir() if file.is_file()
        )
    )
    if not class_dirs:
        return {"status": "dataset_missing"}
    # Deterministic sample: first valid image (by name) of the first Tomato class, else first class.
    tomato_dirs = [directory for directory in class_dirs if directory.name.startswith("Tomato___")]
    preferred = tomato_dirs or class_dirs
    sample_path = None
    for directory in preferred:
        for file in sorted(directory.iterdir()):
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                sample_path = file
                break
        if sample_path is not None:
            break
    if sample_path is None:
        return {"status": "dataset_missing"}
    result = predict_image_file(sample_path, model=model, confidence_threshold=threshold)
    if result.get("status") != "ok":
        return result
    try:
        with Image.open(sample_path) as image:
            image_array = np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        image_array = None
    overlay = None
    if image_array is not None:
        explanation = gradcam_for_prediction(model, image_array)
        if explanation.get("status") == "ok":
            overlay = overlay_heatmap(image_array, explanation["heatmap"])
    return {
        "status": "ok",
        "sample_path": str(sample_path),
        "overlay": overlay,
        **result,
    }


def summary_dataset_root_for_demo() -> str | None:
    data = get_dataset_summary()
    root = data.get("dataset_root")
    return str(root) if root else None


@st.cache_data(ttl=300)
def get_dataset_summary():
    return summarize_dataset()


st.title("CropShield AI 2.0")
st.caption("A preliminary plant-health field assistant, built on real data and measurable evaluation.")
st.page_link(
    "pages/1_Disease_Detection.py",
    label="Upload Plant Photo & Detect Disease",
    icon="📷",
)
st.markdown("### Practical crop-leaf image analysis, built on real data and measurable evaluation.")

st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/1_Disease_Detection.py", label="Disease Detection")
st.sidebar.page_link("pages/2_Dataset_Explorer.py", label="Dataset Explorer")
st.sidebar.page_link("pages/3_Model_Performance.py", label="Model Performance")
st.sidebar.page_link("pages/4_Explainable_AI.py", label="Explainable AI")
st.sidebar.page_link("pages/5_Prediction_History.py", label="Prediction History")
st.sidebar.page_link("pages/8_Plant_Timeline.py", label="Plant Follow-up Timeline")
st.sidebar.page_link("pages/6_About.py", label="About & Documentation")
st.sidebar.page_link("pages/7_Treatment_Guide.py", label="Treatment & Prevention Guide")
if st.sidebar.button("Refresh model and dataset"):
    get_model_status.clear()
    get_dataset_summary.clear()
    st.rerun()

summary = get_dataset_summary()
model_available = get_model_status(str(MODEL_PATH))
evaluation = {"status": "not_evaluated"}
if METRICS_PATH.exists():
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        evaluation = json.load(metrics_file)
models_evaluated = evaluation.get("models", {}) if evaluation.get("status") == "evaluated" else {}
crop_count = (
    len({name.split("___", maxsplit=1)[0] for name in summary["class_counts"]})
    if summary["status"] == "available"
    else None
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Supported crops", crop_count if crop_count is not None else "Not known yet")
with col2:
    st.metric("Disease classes", summary["class_count"] if summary["status"] == "available" else "Not known yet")
with col3:
    model_status = (
        "Available"
        if model_available
        else "Artifact unreadable"
        if MODEL_PATH.exists()
        else "Not trained"
    )
    st.metric("Model status", model_status)
with col4:
    st.metric("Evaluation", "Evaluated" if models_evaluated else "Not evaluated yet")
st.markdown("### How it works")
st.write(
    "CropShield AI analyzes crop leaf images, preprocesses them with the same pipeline used during training, "
    "and predicts the most likely crop-disease class. Results are informational and should not replace a professional agronomist diagnosis."
)

if summary["status"] == "available":
    st.success(
        f"Dataset detected with {summary['class_count']} supported classes at: {summary['dataset_root']}"
    )
else:
    st.warning(
        "Dataset not prepared locally. Run `python scripts/prepare_dataset.py --dataset-archive PATH` "
        "to extract the color images from your PlantVillage ZIP archive."
    )

st.markdown("### Try a real example")
st.info(
    "When the dataset and trained model are present, Disease Detection can run inference on a real dataset sample. "
    "No sample prediction or model score is simulated."
)

# --- Live Demo Preview: one REAL inference run locally on a fixed dataset sample. ---
# Deliberately not a mock result: predictions, confidence, and the heatmap all come
# from the trained model so the home page never shows fabricated numbers.
st.markdown("### \U0001F3AC Live Demo Preview")
demo = get_demo_prediction()
if demo.get("status") == "ok":
    predicted_class = demo["predicted_class"]
    crop_name, _, disease_name = predicted_class.partition("___")
    display_disease = disease_name.replace("_", " ") if disease_name else predicted_class
    # Indicative severity derived from the prediction itself (healthy vs diseased, model uncertainty).
    if is_healthy_class(predicted_class):
        severity, severity_color = "Low", "#2e7d32"
    elif demo.get("is_uncertain"):
        severity, severity_color = "Medium", "#b26a00"
    else:
        severity, severity_color = "High", "#c62828"
    demo_left, demo_right = st.columns([1, 1])
    with demo_left:
        st.image(demo["sample_path"], caption="Sample leaf image (from the local PlantVillage dataset)", width="stretch")
        if demo.get("overlay") is not None:
            st.image(demo["overlay"], caption="Grad-CAM attention overlay on the sample", width="stretch")
    with demo_right:
        st.metric("Crop", crop_name)
        st.metric("Predicted Disease", display_disease)
        st.metric("Confidence", f"{demo['confidence']:.1%}")
        st.markdown(
            f"<div style='display:inline-block;padding:6px 16px;border-radius:16px;"
            f"background:{severity_color};color:white;font-weight:600;'>Severity: {severity} (indicative)</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, float(demo["confidence"])))
        st.write("Top-3 classes:")
        for prediction in demo["top_predictions"]:
            st.write(f"- `{prediction['class_name']}` — {prediction['probability']:.1%}")
    st.caption(
        "This is a real, locally computed sample result, not a simulation. "
        "Upload an image in Disease Detection to get an actual prediction for your own photo."
    )
else:
    reason_messages = {
        "model_unavailable": "The live demo needs a trained model. Run `python scripts/train_model.py` to create one.",
        "dataset_missing": "The live demo needs the local PlantVillage dataset. Prepare it with `python scripts/prepare_dataset.py`.",
    }
    st.info(reason_messages.get(demo.get("status"), "Live demo is unavailable: " + str(demo.get("message", "unknown"))))

# --- Tech Stack: badges for the libraries this project actually uses. ---
TECH_STACK = (
    ("Python", "Core language for the full pipeline"),
    ("Streamlit", "Web app UI and pages"),
    ("TensorFlow / Keras", "MobileNetV2 transfer-learning model"),
    ("CNN", "Convolutional backbone behind Grad-CAM"),
    ("OpenCV", "Heatmap overlays and image analysis"),
    ("NumPy", "Array preprocessing and metrics"),
    ("Pandas", "DataFrames for counts and results"),
    ("Matplotlib", "Confusion-matrix figure export"),
    ("Scikit-learn", "Precision, recall, F1, confusion matrix"),
    ("Pillow", "Image decoding and validation"),
)
badges_html = "".join(
    f"<span style='display:inline-block;margin:4px 6px;padding:6px 14px;border-radius:16px;"
    f"background:#FFFFFF;border:1px solid #d8e5d6;color:#166534;font-weight:600;font-size:0.9rem;' "
    f"title='{description}'>{name}</span>"
    for name, description in TECH_STACK
)
st.markdown("### \U0001F6E0 Tech Stack")
st.markdown(f"<div style='text-align:left;'>{badges_html}</div>", unsafe_allow_html=True)
st.caption("Hover a badge to see what it does in this project. Versions are listed in requirements.txt.")

st.markdown("### Disclaimer")
st.warning("Predictions are for informational use only and are not definitive agricultural diagnoses. Use expert agronomic advice for critical decisions.")
st.caption(
    "Local mode: when the trained model and guidance files are available on this device, "
    "prediction and guidance do not require an online service. External source links require internet access."
)

render_about_footer()
