from __future__ import annotations

import json

import streamlit as st

from src.cropshield.dataset import detect_dataset_root, iter_class_dirs
from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.history import (
    create_action_checklist,
    fetch_action_checklists,
    fetch_tracked_plants,
    set_action_checklist_item,
)
from src.cropshield.knowledge_base import (
    get_verified_guidance,
    get_verified_weather_rule,
    is_healthy_class,
    load_disease_knowledge,
)
from src.cropshield.model import load_labels
from src.cropshield.treatment_guide import SAFE_PRACTICE_WARNINGS, build_verified_checklist

st.set_page_config(page_title="Treatment & Prevention Guide", page_icon="🌿", layout="wide")
st.title("Disease Treatment & Prevention Guide")
st.caption(
    "Guidance is shown only when a disease-specific record has been reviewed against an authoritative source."
)

records = load_disease_knowledge()
labels = load_labels()
if not labels:
    dataset_root = detect_dataset_root()
    labels = [folder.name for folder in iter_class_dirs(dataset_root)] if dataset_root else []

if not labels:
    st.info("No supported disease classes are available yet. Prepare the dataset or train a model first.")
    stop_with_footer()

class_name = st.selectbox("Choose a supported crop condition", sorted(labels))
guidance = get_verified_guidance(class_name, records)

st.subheader("Safe practice warnings — applies to every model label")
for warning in SAFE_PRACTICE_WARNINGS:
    st.warning(warning)

st.subheader("Weather-aware disease risk")
weather_rule = get_verified_weather_rule(class_name, records)
if weather_rule:
    st.info(
        "No weather-based risk assessment can be made because weather data is not configured. "
        f"A verified rule is on file from {weather_rule['source_name']} "
        f"(last reviewed {weather_rule['last_reviewed']}); no risk is inferred without weather data."
    )
    st.link_button("Open the weather-rule source", weather_rule["source_url"])
else:
    st.info(
        "No weather-based risk assessment can be made: weather data is unavailable and no matching "
        "verified weather-disease rule is available."
    )

if guidance is None:
    if is_healthy_class(class_name):
        st.info(
            "This label represents a healthy category, not a guarantee that the plant is disease-free. "
            "Ask a local agricultural officer or qualified plant pathologist to examine visible symptoms."
        )
    else:
        st.warning(
            f"No verified treatment guidance is available for **{class_name}** yet. "
            "Expert confirmation required. This guidance has not been fully verified, so no disease-specific "
            "treatment is shown. Ask a local agricultural officer or qualified plant pathologist to confirm the condition."
        )
    st.caption(
        "Chemical products, doses, and application schedules are intentionally omitted unless they can be "
        "checked against current local approval and product-label instructions."
    )
    st.caption(
        "Offline field inference and Kannada, Hindi, or Telugu treatment translations are not available or "
        "validated in this build. Voice input is not configured."
    )
    stop_with_footer()

st.subheader("How to Treat or Manage This Plant Disease")
st.markdown(f"**Crop:** {guidance['crop']}  \n**Condition:** {guidance['disease']}")
st.markdown(f"**Category:** {guidance['disease_category']}")
st.markdown(f"**Common symptoms:** {guidance['symptoms']}")
st.markdown(f"**Distinguishing signs:** {guidance['distinguishing_signs']}")
st.markdown(f"**Conditions that favor it:** {guidance['favorable_conditions']}")
st.markdown(f"**Possible look-alikes:** {guidance['look_alikes']}")

for section_title, field in (
    ("What to do first", "immediate_steps"),
    ("Management options", "management"),
    ("What to monitor", "monitoring"),
):
    st.markdown(f"### {section_title}")
    for item in guidance[field]:
        st.markdown(f"- {item}")

for section_title, field in (
    ("Cultural practices", "cultural_practices"),
    ("Biological options", "biological_options"),
    ("Verified chemical options", "chemical_options"),
    ("Follow-up inspection", "follow_up_inspection"),
):
    items = guidance.get(field, [])
    if items:
        st.markdown(f"### {section_title}")
        for item in items:
            st.markdown(f"- {item}")

st.subheader("How to Prevent This Disease")
for item in guidance["prevention"]:
    st.markdown(f"- {item}")

if guidance["safety_precautions"]:
    st.subheader("Safety precautions")
    for item in guidance["safety_precautions"]:
        st.markdown(f"- {item}")

st.subheader("What Not to Do")
for item in guidance.get("what_not_to_do", []):
    st.markdown(f"- {item}")
for item in SAFE_PRACTICE_WARNINGS:
    st.markdown(f"- {item}")

st.subheader("When to contact an expert")
st.write(
    guidance.get(
        "when_to_contact_expert",
        "Contact a qualified local agricultural expert if symptoms are severe, spreading, or remain uncertain.",
    )
)
reinspection_interval = guidance.get("reinspection_interval")
st.info(
    f"Verified re-inspection guidance: {reinspection_interval}"
    if reinspection_interval
    else "Monitor regularly and re-inspect after a few days."
)

st.caption(f"Source: {guidance['source_name']} · Last reviewed: {guidance['last_reviewed']}")
st.link_button("Open the source", guidance["source_url"])

st.subheader("Recovery checklist")
checklist_actions = build_verified_checklist(guidance)
plants = fetch_tracked_plants()
if not checklist_actions:
    st.info("This verified source does not yet provide checklist actions to save.")
elif not plants:
    st.info("Create a plant timeline before saving a checklist locally.")
else:
    plant_options = {int(plant["id"]): plant["name"] for plant in plants}
    selected_plant_id = st.selectbox(
        "Save checklist to a plant timeline",
        list(plant_options),
        format_func=lambda plant_id: plant_options[plant_id],
    )
    if st.button("Save verified checklist locally"):
        checklist_id = create_action_checklist(
            selected_plant_id,
            class_name,
            guidance["source_name"],
            guidance["source_url"],
            checklist_actions,
        )
        st.success(f"Checklist saved locally (ID {checklist_id}).")
    saved_items = fetch_action_checklists(selected_plant_id, class_name)
    if saved_items:
        st.markdown("**Saved checklists for this plant and condition**")
        checklist_groups: dict[int, list[dict]] = {}
        for item in saved_items:
            checklist_groups.setdefault(int(item["checklist_id"]), []).append(item)
        for checklist_id, checklist_rows in checklist_groups.items():
            st.markdown(f"**Saved {checklist_rows[0]['created_at']}**")
            for item in checklist_rows:
                completed = st.checkbox(
                    f"{item['category']}: {item['action']}",
                    value=bool(item["completed"]),
                    key=f"checklist_item_{item['item_id']}",
                )
                if completed != bool(item["completed"]):
                    set_action_checklist_item(int(item["item_id"]), completed)
                    st.rerun()
            st.caption(
                f"Source: {checklist_rows[0]['source_name']} · "
                f"{checklist_rows[0]['source_url']}"
            )

speech_text = " ".join(
    [
        f"{guidance['disease']}.",
        *guidance["immediate_steps"],
        *guidance["management"],
        *guidance["monitoring"],
        *guidance.get("follow_up_inspection", []),
    ]
)
speech_payload = json.dumps(speech_text, ensure_ascii=True).replace("<", "\\u003c")
st.subheader("Voice accessibility")
st.caption(
    "Read aloud uses the browser's installed English speech voice; speech stays in your browser. "
    "Translations are not available in this build, and voice input is not configured."
)
st.components.v1.html(
    f"""<button type="button" onclick="if (!('speechSynthesis' in window)) {{
      document.getElementById('speech-status').textContent = 'Speech synthesis is not supported by this browser.';
    }} else {{
      const utterance = new SpeechSynthesisUtterance({speech_payload});
      utterance.lang = 'en';
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
      document.getElementById('speech-status').textContent = 'Reading verified English guidance aloud.';
    }}">Read English guidance aloud</button>
    <span id="speech-status" aria-live="polite"></span>""",
    height=55,
)
st.caption(
    "Offline TFLite inference, severity scoring, and machine-generated translations are not enabled or "
    "validated. The app does not claim offline inference capability."
)

render_about_footer()
