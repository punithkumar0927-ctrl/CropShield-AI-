from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.cropshield.config import LABELS_PATH, MODEL_PATH
from src.cropshield.explainability import generate_explanation
from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.model import load_labels, load_model
from src.cropshield.predict import predict_uploaded_file
from src.cropshield.preprocessing import validate_upload

st.set_page_config(page_title="Explainable AI", page_icon="🔎", layout="wide")
st.title("Explainable AI")
st.write("Grad-CAM highlights image regions that influence the model's selected class.")


@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


model = get_model(str(MODEL_PATH))
labels = load_labels(LABELS_PATH)
if model is None or not labels:
    st.warning("A trained model and matching class labels are required for a real Grad-CAM explanation.")
    stop_with_footer()

uploaded = st.file_uploader(
    "Upload a leaf image (JPG, JPEG, or PNG; up to 8 MB)",
    type=["jpg", "jpeg", "png"],
)
if uploaded is None:
    st.info("Upload an image to generate a Grad-CAM explanation.")
    stop_with_footer()

valid, message, width, height = validate_upload(uploaded)
if not valid:
    st.error(message)
    stop_with_footer()
uploaded.seek(0)
try:
    with Image.open(uploaded) as image:
        original = image.convert("RGB")
        image_array = np.asarray(original)
except (OSError, UnidentifiedImageError, ValueError):
    st.error("The uploaded image could not be decoded.")
    stop_with_footer()

st.image(original, caption=f"Original image — {width} × {height}", width="stretch")
prediction = predict_uploaded_file(uploaded, model=model)
if prediction["status"] != "ok":
    st.error(prediction["message"])
    stop_with_footer()

class_index = next(
    index
    for index, item in enumerate(prediction["all_probabilities"])
    if item["class_name"] == prediction["predicted_class"]
)
st.write(
    f"Predicted class: **{prediction['predicted_class']}** "
    f"(model confidence: {prediction['confidence']:.1%})"
)
explanation = generate_explanation(
    image_array,
    str(MODEL_PATH),
    class_index=class_index,
    model=model,
)
if explanation["status"] != "ok":
    st.warning(explanation.get("message", "A reliable explanation could not be generated for this model."))
else:
    # --- Grad-CAM view: original, raw heatmap, and overlay side by side. ---
    first, second, third = st.columns(3)
    with first:
        st.image(original, caption="Original image", width="stretch")
    with second:
        heatmap_uint8 = np.uint8(255 * explanation["heatmap"])
        st.image(heatmap_uint8, caption="Raw Grad-CAM activation map", width="stretch")
    with third:
        st.image(
            Image.fromarray(explanation["overlay"]),
            caption="Grad-CAM overlay",
            width="stretch",
        )
    st.caption("Brighter regions indicate areas that influenced the model's prediction.")
    st.caption(
        "The heatmap indicates regions influencing this model prediction; it is not proof of disease "
        "and is not a definitive agricultural diagnosis."
    )

render_about_footer()
