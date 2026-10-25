from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.cropshield.config import DETECTION_THRESHOLDS_PATH, LABELS_PATH, MODEL_PATH, MODELS_DIR, TFLITE_MODEL_PATH
from src.cropshield.dataset import list_dataset_samples
from src.cropshield.explainability import gradcam_for_prediction, overlay_heatmap
from src.cropshield.field_risk import assess_field_risk
from src.cropshield.footer import render_about_footer
from src.cropshield.history import (
    add_plant_observation,
    append_prediction,
    fetch_possible_plot_spread,
    fetch_tracked_plants,
)
from src.cropshield.image_quality import (
    assess_image_quality,
    crop_largest_leaf_like_region,
)
from src.cropshield.knowledge_base import (
    get_verified_guidance,
    get_verified_weather_rule,
    is_healthy_class,
)
from src.cropshield.model import TFLiteModel, load_labels, load_model, load_tflite_model
from src.cropshield.predict import predict_image_array, predict_image_file, predict_uploaded_file
from src.cropshield.preprocessing import validate_upload
from src.cropshield.report import build_prediction_report
from src.cropshield.treatment_guide import SAFE_PRACTICE_WARNINGS, build_action_plan
from src.cropshield.trust import assess_trust, load_trust_thresholds

st.set_page_config(page_title="Disease Detection", page_icon="🌱", layout="wide")
st.title("Disease Detection")
st.write("Upload a crop leaf photo or select a genuine image from a locally available dataset.")
st.caption(
    "Inference uses a locally available Keras or TensorFlow Lite model. TFLite conversion and prediction "
    "are supported, but no trained CropShield model is currently available to validate offline field operation; "
    "opening external source links requires internet access."
)


@st.cache_resource
def get_model():
    selected_model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), MODEL_PATH)
    model = load_model(selected_model_path)
    if model is None and TFLITE_MODEL_PATH.is_file():
        model = load_tflite_model(TFLITE_MODEL_PATH)
        selected_model_path = TFLITE_MODEL_PATH
    return model, selected_model_path.name


MODEL_CANDIDATES = (
    MODEL_PATH,
    MODELS_DIR / "cropshield_model.keras",
    MODELS_DIR / "cropshield_model.h5",
)
try:
    model, model_file_name = get_model()
except (OSError, ValueError, RuntimeError) as error:
    model = None
    model_file_name = MODEL_PATH.name
    st.error(f"The local model could not be loaded: {error}")
try:
    label_candidates = (LABELS_PATH, MODELS_DIR / "class_names.json")
    selected_labels_path = next((path for path in label_candidates if path.is_file()), LABELS_PATH)
    labels = load_labels(selected_labels_path)
except (OSError, ValueError) as error:
    labels = []
    st.error(f"The model class mapping could not be loaded: {error}")
if st.sidebar.button("Refresh model and data"):
    get_model.clear()
    st.rerun()
if model is None:
    if any(path.is_file() for path in (*MODEL_CANDIDATES, TFLITE_MODEL_PATH)):
        st.error(
            "A model file exists but could not be loaded. Check that it is a valid CropShield model "
            "with compatible labels, or retrain it using scripts/train_model.py."
        )
    else:
        st.warning(
            "No trained CropShield model is available. Please run scripts/train_model.py first. "
            "This page will not generate sample predictions without a real model. "
            "The training pipeline creates the model, matching class mapping, and preprocessing configuration."
        )
        with st.expander("How to train the model"):
            st.write(
                "The local PlantVillage dataset contains 10 Tomato classes. The recommended focused "
                "MobileNetV2 model trains on those classes rather than all 38."
            )
            st.code(
                "py -3.13 scripts\\train_model.py "
                "--dataset-root \"data\\PlantVillage-Dataset\\color\" "
                "--model-type mobilenet --crop Tomato --batch-size 16 --image-size 224 --epochs 8",
                language="powershell",
            )
            st.write(
                "After training finishes, refresh this page. The training script writes the model, "
                "class mapping, and preprocessing configuration needed for inference."
            )
elif not labels:
    st.error(
        "The trained model was found, but its class mapping is missing or invalid. "
        "Restore models/class_labels.json from the same training run or retrain the model."
    )

config_path = MODELS_DIR / "model_config.json"
model_config = {}
try:
    trust_thresholds = load_trust_thresholds(DETECTION_THRESHOLDS_PATH)
except (OSError, ValueError) as error:
    trust_thresholds = None
    st.error(f"Trust-level thresholds could not be loaded: {error}")
confidence_threshold = None
confidence_policy = None
if config_path.exists():
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        if not isinstance(model_config, dict):
            raise ValueError("Configuration root must be an object.")
        confidence_policy = model_config.get("confidence_policy")
        if confidence_policy is not None:
            if not isinstance(confidence_policy, dict):
                raise ValueError("Confidence policy must be an object.")
            confidence_threshold = confidence_policy.get("threshold")
            if confidence_threshold is not None and (
                isinstance(confidence_threshold, bool)
                or not isinstance(confidence_threshold, (int, float))
                or not 0.0 <= confidence_threshold <= 1.0
            ):
                raise ValueError("Confidence threshold must be between 0 and 1.")
    except (OSError, ValueError) as error:
        confidence_threshold = None
        model_config = {}
        st.error(f"The preprocessing/model configuration could not be read: {error}")
preprocessing_config_path = MODELS_DIR / "preprocessing_config.json"
if model is None:
    preprocessing_config = None
elif preprocessing_config_path.is_file():
    try:
        with preprocessing_config_path.open("r", encoding="utf-8") as config_file:
            preprocessing_config = json.load(config_file)
        if not isinstance(preprocessing_config, dict):
            raise ValueError("Configuration root must be an object.")
        if preprocessing_config.get("color_mode") != "RGB":
            raise ValueError("Only RGB preprocessing configurations are supported.")
        if preprocessing_config.get("resize_method") != "bilinear":
            raise ValueError("The saved model requires an unsupported image resize method.")
        saved_input_size = preprocessing_config.get("input_size")
        if (
            not isinstance(saved_input_size, list)
            or len(saved_input_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in saved_input_size)
        ):
            raise ValueError("Input size must be a pair of positive integer dimensions.")
        input_range = preprocessing_config.get("input_pixel_range")
        if input_range != [0, 255]:
            raise ValueError("Only raw 0..255 RGB input is supported by this workflow.")
        if model is not None:
            model_shape = getattr(model, "input_shape", None)
            if model_shape is not None and all(value is not None for value in model_shape[1:3]):
                expected_size = [int(model_shape[1]), int(model_shape[2])]
                if saved_input_size != expected_size:
                    raise ValueError(
                        f"Saved input size {saved_input_size} does not match model input {expected_size}."
                    )
    except (OSError, ValueError) as error:
        preprocessing_config = None
        st.error(f"Preprocessing configuration is invalid: {error}")
else:
    preprocessing_config = None
    st.warning(
        "The trained model is present, but preprocessing_config.json is missing. Prediction is disabled "
        "because input normalization and resize settings cannot be verified. Retrain with scripts/train_model.py."
    )

image_source = st.radio(
    "Image source",
    ["Upload image", "Take a photo", "Use a PlantVillage sample"],
    horizontal=True,
)
uploaded_file = None
sample_path = None
quality_assessment = None
quality_acceptable = False
if image_source == "Upload image":
    uploaded_file = st.file_uploader(
        "Choose a crop leaf image (JPG, JPEG, PNG, or WEBP; up to 10 MB)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Images above 40 megapixels are rejected to protect memory.",
    )
    image_available = uploaded_file is not None
elif image_source == "Take a photo":
    uploaded_file = st.camera_input("Take a clear photo of the affected leaf")
    image_available = uploaded_file is not None
else:
    samples = list_dataset_samples(limit=40)
    if not samples:
        st.info("No PlantVillage sample images are available. Set up the dataset to enable this option.")
        image_available = False
    else:
        sample_path = st.selectbox(
            "Select a real dataset image",
            options=samples,
            format_func=lambda path: f"{Path(path).parent.name} — {Path(path).name}",
        )
        image_available = bool(sample_path)

if image_available:
    if uploaded_file is not None:
        valid, message, width, height = validate_upload(uploaded_file)
        if not valid:
            st.error(message)
            image_available = False
        else:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded image — original dimensions: {width} × {height}", width="stretch")
    else:
        image = Image.open(sample_path).convert("RGB")
        st.image(
            image,
            caption=f"{Path(sample_path).parent.name} — original dimensions: {image.width} × {image.height}",
            width="stretch",
        )
    if image_available:
        quality_assessment = assess_image_quality(image)
        quality_acceptable = quality_assessment.acceptable
        st.subheader("Smart Photo Coach")
        with st.expander("Image-quality details", expanded=True):
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            quality_col1.metric("Brightness", f"{quality_assessment.brightness:.0f}/255")
            quality_col2.metric("Focus score", f"{quality_assessment.blur_score:.0f}")
            quality_col3.metric("Bright glare", f"{quality_assessment.glare_fraction:.1%}")
            if quality_assessment.feedback:
                for feedback in quality_assessment.feedback:
                    st.warning(feedback)
            else:
                st.success("Good basic photo quality. You can proceed with detection.")
            st.metric("Green-color coverage estimate", f"{quality_assessment.green_pixel_fraction:.1%}")
            st.metric("Leaf-like segmented coverage estimate", f"{quality_assessment.leaf_coverage_fraction:.1%}")
            st.caption(
                "Green-pixel coverage and OpenCV contour segmentation are color-based estimates, not a "
                "guaranteed leaf detector. Yellowing, brown leaves, lighting, and background colors can cause errors."
            )
        experimental_crop = crop_largest_leaf_like_region(image)
        compare_cropped_prediction = st.checkbox(
            "Experimental: compare prediction on the largest leaf-like crop",
            value=False,
            disabled=experimental_crop is None,
            help="OpenCV color segmentation is experimental and may crop background or omit discolored leaf areas.",
        )
        if experimental_crop is not None and compare_cropped_prediction:
            st.image(
                experimental_crop,
                caption="Experimental leaf-like crop preview. Original image remains unchanged.",
                width="stretch",
            )
        if not quality_acceptable:
            st.error(
                "Please retake the photo before analysis; the image may be too small, blank, too dark/bright, "
                "or too blurry."
            )

save_history = st.checkbox(
    "Save prediction details to local history",
    value=False,
    disabled=model is None or not labels,
    help="Stores prediction metadata, image-quality measurements and optional notes locally. Image bytes are never saved here.",
)
history_notes = st.text_input(
    "Optional note to save with history",
    max_chars=2000,
    disabled=not save_history,
    placeholder="No personal or identifying information is needed.",
)
show_ai_attention = st.checkbox(
    "Show AI attention (Grad-CAM, when supported)",
    value=True,
    help="The heatmap highlights model activations; it is not proof that the model used valid disease features.",
)
tracked_plants = fetch_tracked_plants()
timeline_options = {None: "Do not add this scan to a plant timeline"}
timeline_options.update({int(plant["id"]): plant["name"] for plant in tracked_plants})


def format_timeline_option(plant_id: int | None) -> str:
    if plant_id is None:
        return "Do not add this scan to a plant timeline"
    return timeline_options[plant_id]


timeline_plant_id = st.selectbox(
    "Plant timeline (optional)",
    options=list(timeline_options),
    format_func=format_timeline_option,
    help="The scan will be added only after successful inference and only when you enable saving below.",
)
save_scan_to_timeline = st.checkbox(
    "Save successful scan to the selected plant timeline",
    value=False,
    disabled=timeline_plant_id is None or model is None or not labels,
)
timeline_notes = st.text_input(
    "Optional field note for this scan",
    max_chars=500,
    placeholder="For example, lower leaves look more yellow today.",
    disabled=not save_scan_to_timeline,
)
retain_scan_photo = st.checkbox(
    "I explicitly agree to retain this scan photo in the plant timeline",
    value=False,
    disabled=not save_scan_to_timeline or not image_available or model is None,
    help="Photo bytes are saved only if this is checked and inference succeeds.",
)
with st.expander("Optional: preliminary field-risk guidance"):
    st.caption("This transparent checklist is not a disease diagnosis and does not replace local expert advice.")
    inspected_leaves = st.number_input("Leaves inspected", min_value=1, max_value=10000, value=20)
    affected_leaves = st.number_input(
        "Leaves with similar symptoms",
        min_value=0,
        max_value=int(inspected_leaves),
        value=0,
    )
    nearby_affected = st.selectbox("Similar symptoms on nearby plants?", ["Not sure", "No", "Yes"])
    prolonged_wetness = st.selectbox("Prolonged leaf wetness or humid weather?", ["Not sure", "No", "Yes"])
    overhead_irrigation = st.selectbox("Overhead irrigation that wets leaves?", ["Not sure", "No", "Yes"])
    rapid_symptom_spread = st.selectbox("Have you noticed rapid symptom spread?", ["Not sure", "No", "Yes"])

if image_available and st.button(
    "Detect Disease",
    type="primary",
    disabled=model is None or not labels or not quality_acceptable or preprocessing_config is None,
):
    started_at = time.perf_counter()
    with st.spinner("Analyzing image with the trained model..."):
        result = (
            predict_uploaded_file(
                uploaded_file,
                model=model,
                confidence_threshold=confidence_threshold,
            )
            if uploaded_file is not None
            else predict_image_file(
                sample_path,
                model=model,
                confidence_threshold=confidence_threshold,
            )
        )
    processing_seconds = time.perf_counter() - started_at
    if result["status"] == "ok":
        image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        explanation = None
        heatmap_leaf_focus = None
        if show_ai_attention:
            if isinstance(model, TFLiteModel):
                st.info("Explainability heatmap is not available for the TensorFlow Lite model.")
            else:
                explanation = gradcam_for_prediction(
                    model,
                    image_array,
                    class_index=max(
                        range(len(labels)),
                        key=lambda index: result["all_probabilities"][index]["probability"],
                    ),
                )
                if explanation["status"] == "ok":
                    mask = quality_assessment.leaf_mask
                    heatmap = explanation["heatmap"]
                    heatmap_total = float(np.sum(heatmap))
                    if heatmap_total > 0 and np.any(mask):
                        heatmap_leaf_focus = float(np.sum(heatmap[mask]) / heatmap_total)
                else:
                    st.info(
                        "Explainability heatmap is not available for this model architecture. "
                        f"Details: {explanation.get('message', 'No compatible activation map.')}"
                    )
        trust_result = (
            assess_trust(
                float(result["confidence"]),
                float(result["top_prediction_margin"]),
                quality_assessment,
                confidence_threshold,
                heatmap_leaf_focus=heatmap_leaf_focus,
                thresholds=trust_thresholds,
            )
            if trust_thresholds is not None
            else None
        )
        risk_assessment = assess_field_risk(
            int(affected_leaves),
            int(inspected_leaves),
            {"Not sure": None, "No": False, "Yes": True}[nearby_affected],
            {"Not sure": None, "No": False, "Yes": True}[prolonged_wetness],
            {"Not sure": None, "No": False, "Yes": True}[overhead_irrigation],
            {"Not sure": None, "No": False, "Yes": True}[rapid_symptom_spread],
        )
        st.subheader("Preliminary field-risk guidance")
        st.metric("Reported spread-risk indicator", risk_assessment.level)
        st.write(f"Reported affected leaves: {affected_leaves} of {inspected_leaves}")
        for factor in risk_assessment.factors:
            st.markdown(f"- {factor}")
        st.caption(
            "Rule-based screening from your observations only. It does not identify a disease, "
            "predict an outbreak, or recommend treatment."
        )
        predicted_crop, separator, predicted_disease = result["predicted_class"].partition("___")
        if trust_result is not None:
            trust_badge = {
                "High": "🟢 High",
                "Medium": "🟡 Medium",
                "Low": "🟠 Low",
                "Cannot Assess": "⚪ Cannot Assess",
            }[trust_result["level"]]
            st.metric("Trust level (heuristic, not diagnosis)", trust_badge)
            st.write(trust_result["explanation"])
        low_trust = trust_result is not None and trust_result["level"] in {"Low", "Cannot Assess"}
        if result["is_uncertain"] or low_trust:
            st.warning(
                f"Preliminary model output: {result['predicted_class']} ({result['confidence']:.1%}). "
                "Photo quality, model ambiguity, or attention-map signals make this result unreliable; "
                "do not treat it as a confirmed diagnosis."
            )
            st.warning(
                "This condition may not be covered by the current model. Please consult a local agricultural expert."
            )
        else:
            st.success(
                f"Likely class: {result['predicted_class']} · Model confidence: {result['confidence']:.1%}"
            )
        if confidence_threshold is None:
            st.warning(
                "This model has no usable confidence threshold derived from validation data. "
                "Treat the output as uncalibrated and seek expert confirmation."
            )
        st.write(f"Crop: {predicted_crop if separator else 'Not separately encoded in label'}")
        st.write(f"Disease/condition: {predicted_disease if separator else result['predicted_class']}")
        st.write(f"Prediction processing time: {processing_seconds:.3f} seconds")
        st.write(f"Prediction time: {processing_seconds * 1000:.1f} ms")
        st.write(f"Model file: {model_file_name}")
        with st.expander("Model preprocessing and output details"):
            preprocessing_details = (
                f"RGB color; {preprocessing_config.get('resize_method')} resize; "
                f"input pixel range {preprocessing_config.get('input_pixel_range')}."
                if preprocessing_config is not None
                else "RGB float32 input resized to model shape using bilinear interpolation."
            )
            st.write(preprocessing_details)
            st.write("No train-time augmentation is applied during inference.")
            st.write(f"Model input shape: {getattr(model, 'input_shape', 'not exposed')}")
            st.write(f"Top-1 / top-2 probability margin: {result['top_prediction_margin']:.1%}")
            st.write(
                "Confidence calibration: validation-derived threshold available."
                if confidence_threshold is not None
                else "Confidence calibration: no validation-derived threshold is available."
            )
        st.caption(
            "Confidence is the model's probability output, not a measure of diagnostic certainty. "
            "PlantVillage images are controlled; performance on field photographs and unfamiliar conditions may be lower. "
            "This result is informational and is not a definitive agricultural diagnosis."
        )
        prediction_rows = [
            {"Class": prediction["class_name"], "Probability": prediction["probability"]}
            for prediction in result["top_predictions"]
        ]
        st.dataframe(
            [
                {
                    "Class": prediction["class_name"],
                    "Probability": f"{prediction['probability']:.1%}",
                }
                for prediction in result["top_predictions"]
            ],
            hide_index=True,
            width="stretch",
        )
        chart = px.bar(
            pd.DataFrame(prediction_rows),
            x="Class",
            y="Probability",
            color="Probability",
            color_continuous_scale=["#d5e4cf", "#1d6849"],
            title="Top three model outputs",
        )
        chart.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(chart, width="stretch")
        if show_ai_attention and explanation is not None and explanation["status"] == "ok":
            with st.expander("Grad-CAM attention explanation", expanded=True):
                st.image(
                    overlay_heatmap(image_array, explanation["heatmap"]),
                    caption=(
                        "AI attention overlay for the predicted class. Highlighted areas indicate "
                        "model activation, not verified disease evidence."
                    ),
                    width="stretch",
                )
                st.write(
                    "The highlighted areas show where the model had stronger class-specific activation. "
                    "This is not proof that those pixels represent disease."
                )
                if heatmap_leaf_focus is not None and heatmap_leaf_focus < trust_thresholds[
                    "minimum_heatmap_leaf_focus"
                ]:
                    st.warning(
                        "Warning: the model may be focusing on the background rather than the color-segmented "
                        "leaf-like region. This prediction may be unreliable."
                    )
                elif heatmap_leaf_focus is None:
                    st.caption(
                        "Leaf-focus comparison is unavailable because no color-segmented leaf-like region was found."
                    )
        if compare_cropped_prediction and experimental_crop is not None:
            cropped_started_at = time.perf_counter()
            cropped_result = predict_image_array(
                np.asarray(experimental_crop, dtype=np.float32),
                labels,
                model,
                confidence_threshold=confidence_threshold,
            )
            cropped_elapsed_ms = (time.perf_counter() - cropped_started_at) * 1000
            st.warning(
                "Experimental crop comparison: color-based segmentation may include background or remove "
                "yellow/brown leaf regions. The original-image prediction remains the primary result."
            )
            if cropped_result["status"] == "ok":
                crop_col1, crop_col2 = st.columns(2)
                crop_col1.metric("Original image", result["predicted_class"], f"{result['confidence']:.1%}")
                crop_col2.metric(
                    "Experimental cropped image",
                    cropped_result["predicted_class"],
                    f"{cropped_result['confidence']:.1%}",
                )
                st.caption(f"Cropped-image model time: {cropped_elapsed_ms:.1f} ms")
            else:
                st.info(cropped_result["message"])
        model_version = "trained-model"
        if config_path.exists():
            model_version = model_config.get("model_version", model_version)
        if save_history:
            try:
                append_prediction(
                    result["predicted_class"],
                    model_version,
                    float(result["confidence"]),
                    top_predictions=result["top_predictions"],
                    trust_level=trust_result["level"] if trust_result is not None else None,
                    inference_ms=processing_seconds * 1000,
                    image_quality={
                        "width": quality_assessment.width,
                        "height": quality_assessment.height,
                        "brightness": quality_assessment.brightness,
                        "blur_score": quality_assessment.blur_score,
                        "glare_fraction": quality_assessment.glare_fraction,
                        "green_pixel_fraction": quality_assessment.green_pixel_fraction,
                        "leaf_coverage_fraction": quality_assessment.leaf_coverage_fraction,
                        "leaf_contour_count": quality_assessment.leaf_contour_count,
                    },
                    user_notes=history_notes,
                    image_name=(
                        Path(uploaded_file.name).name
                        if uploaded_file is not None
                        else Path(sample_path).name if sample_path is not None else None
                    ),
                )
            except (sqlite3.Error, ValueError, OSError) as error:
                st.error(f"Prediction succeeded, but local history could not be saved: {error}")
            else:
                st.info("Prediction metadata and optional notes were saved locally. Image bytes were not saved.")
        if save_scan_to_timeline and timeline_plant_id is not None:
            observation_note = timeline_notes.strip() or "No field note added for this model scan."
            try:
                add_plant_observation(
                    int(timeline_plant_id),
                    date.today(),
                    observation_note,
                    predicted_class=result["predicted_class"],
                    confidence=float(result["confidence"]),
                    retained_image=image if retain_scan_photo else None,
                    record_type="ai_prediction",
                    trust_level=trust_result["level"] if trust_result is not None else None,
                )
            except (sqlite3.Error, ValueError, OSError) as error:
                st.error(f"Prediction succeeded, but the plant timeline could not be updated: {error}")
            else:
                st.success(
                    "Successful scan added to the selected plant timeline"
                    + (" with its explicitly retained photo." if retain_scan_photo else ".")
                )
                selected_timeline_plant = next(
                    plant for plant in tracked_plants if int(plant["id"]) == int(timeline_plant_id)
                )
                plot_spread = fetch_possible_plot_spread(str(selected_timeline_plant["plot_name"]))
                for spread in plot_spread:
                    if spread["predicted_class"] == result["predicted_class"]:
                        st.warning(
                            f"Possible spread risk: {spread['scan_count']} AI scans across "
                            f"{spread['plant_count']} plant(s) in **{selected_timeline_plant['plot_name']}** "
                            f"recorded **{spread['predicted_class']}**. These are unconfirmed model predictions, "
                            "not confirmation that disease has spread."
                        )
        guidance = (
            None
            if trust_result is not None and trust_result["level"] == "Cannot Assess"
            else get_verified_guidance(result["predicted_class"])
        )
        healthy_class = is_healthy_class(result["predicted_class"])
        st.subheader("How to Treat or Manage This Plant Disease")
        if healthy_class:
            st.info(
                "The predicted label is a healthy category. This is not proof that the plant is disease-free. "
                "If you see symptoms, upload a clear photo and consult a local agricultural expert."
            )
        elif guidance is None:
            st.info(
                "No verified treatment guidance is available for this disease yet. Expert confirmation is required; "
                "no treatment advice is generated from this model label. Ask a local agricultural officer or "
                "qualified plant pathologist to confirm the condition."
            )
        else:
            st.write(f"**Category:** {guidance['disease_category']}")
            st.write(f"**Symptoms:** {guidance['symptoms']}")
            st.write(f"**Distinguishing signs:** {guidance['distinguishing_signs']}")
            st.write(f"**Conditions that favor it:** {guidance['favorable_conditions']}")
            st.write(f"**Possible look-alikes:** {guidance['look_alikes']}")
            st.caption(
                f"Source: {guidance['source_name']} · Last reviewed: {guidance['last_reviewed']}"
            )
            st.link_button("Open the source", guidance["source_url"])
        action_plan = build_action_plan(guidance)
        st.subheader("Prioritized action plan")
        for section, items in action_plan.items():
            st.markdown(f"**{section}**")
            for item in items:
                st.markdown(f"- {item}")
        st.subheader("What Not to Do")
        for item in SAFE_PRACTICE_WARNINGS:
            st.markdown(f"- {item}")
        if guidance:
            for item in guidance.get("what_not_to_do", []):
                st.markdown(f"- {item}")
        weather_rule = get_verified_weather_rule(result["predicted_class"])
        st.subheader("Weather-aware disease risk")
        if weather_rule:
            st.info(
                "No weather-based risk assessment can be made because weather data is not configured. "
                f"The matching verified rule is from {weather_rule['source_name']} "
                f"(reviewed {weather_rule['last_reviewed']})."
            )
            st.link_button("Open the weather-rule source", weather_rule["source_url"])
        else:
            st.info(
                "No weather-based risk assessment can be made: weather data is unavailable and no matching "
                "verified disease-weather rule is available."
            )
        if result["is_uncertain"]:
            st.info(
                "This result is marked uncertain. Use the Plant Follow-up Timeline to prepare an "
                "expert-verification report; no information is sent automatically."
            )
        st.subheader("How to Prevent This Disease")
        if healthy_class:
            st.info("No disease-specific prevention guide is attached to a healthy class.")
        elif guidance is None:
            st.info("Disease-specific prevention advice is unavailable until a trusted source is reviewed.")
        else:
            for item in guidance["prevention"]:
                st.markdown(f"- {item}")
            if guidance["safety_precautions"]:
                st.markdown("**Safety precautions**")
                for item in guidance["safety_precautions"]:
                    st.markdown(f"- {item}")
        report_result = {
            **result,
            "model_file_name": model_file_name,
            "trust_level": trust_result["level"] if trust_result is not None else None,
            "trust_explanation": trust_result["explanation"] if trust_result is not None else "",
        }
        report = build_prediction_report(report_result, guidance, processing_seconds, image=image)
        st.download_button(
            "Download or print this report",
            data=report,
            file_name="cropshield-prediction-report.html",
            mime="text/html",
        )
    else:
        st.error(result["message"])

render_about_footer()
