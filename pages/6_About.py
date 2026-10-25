from __future__ import annotations

import streamlit as st

from src.cropshield.footer import render_about_footer
from src.cropshield.knowledge_base import load_local_expert_contacts

st.set_page_config(page_title="About & Documentation", layout="wide")
st.title("About & Documentation")

st.markdown(
    """
    # CropShield AI 2.0

    CropShield AI is a practical crop disease identification project built for machine learning and computer vision portfolio work.

    ## Intended use
    The platform uses a trained image model to help identify disease patterns in leaf images. It is intended for research, learning, and demonstration, not for real-world field diagnosis without expert review.

    ## Dataset
    The project is designed around the PlantVillage dataset and includes setup instructions for preparing a compatible folder-per-class dataset.

    ## Limitations
    - models can fail on lighting variation or different crop varieties
    - performance on public datasets does not guarantee field performance
    - explainability maps are helpful indicators, not proof of disease
    - the photo coach uses basic image statistics and does not verify that an image contains a plant
    - spread-risk guidance is a transparent checklist, not a disease-specific forecast
    """
)

st.subheader("Contact a local agricultural expert")
contacts = load_local_expert_contacts()
if not contacts:
    st.info(
        "No local contacts are configured. Administrators can add verified extension-service or university "
        "contacts to `data/local_expert_contacts.json`; no phone numbers or government contacts are invented."
    )
else:
    for contact in contacts:
        st.markdown(f"**{contact['name']}** · {contact['organization']}")
        if "phone" in contact:
            st.write(f"Phone: {contact['phone']}")
        st.link_button("Open contact website", contact["url"])

with st.expander("Project structure"):
    st.code(
        """CropShield-AI/
app.py
pages/
  1_Disease_Detection.py
  2_Dataset_Explorer.py
  3_Model_Performance.py
  4_Explainable_AI.py
  5_Prediction_History.py
  6_About.py
src/cropshield/
  config.py
  dataset.py
  preprocessing.py
  model.py
  train.py
  evaluate.py
  predict.py
  explainability.py
  history.py
tests/
  test_dataset.py
  test_preprocessing.py
  test_predict.py
  test_evaluation.py
  test_explainability.py
README.md
requirements.txt
"""
    )

render_about_footer()
