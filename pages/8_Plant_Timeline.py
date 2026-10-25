from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.cropshield.config import LABELS_PATH
from src.cropshield.expert_request import build_expert_request
from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.history import (
    add_plant_observation,
    create_tracked_plant,
    delete_plant_observation,
    delete_tracked_plant,
    fetch_plot_map,
    fetch_possible_plot_spread,
    fetch_plant_observations,
    fetch_tracked_plants,
    get_plant_timeline_image_path,
)
from src.cropshield.model import load_labels
from src.cropshield.preprocessing import validate_upload

st.set_page_config(page_title="Plant Timeline", page_icon="🌱", layout="wide")
st.title("Plant Follow-up Timeline")
st.caption(
    "Keep dated notes for the same plant or plot. Repeated model scores are observations, "
    "not proof of disease progression or treatment effectiveness."
)
st.info(
    "Timeline records are stored locally in SQLite. Photos are never retained unless you explicitly "
    "enable photo retention for an observation. Do not add names or personal details you do not need."
)

with st.expander("Create a plant or plot"):
    with st.form("create_tracked_plant", clear_on_submit=True):
        new_name = st.text_input("Plant or group name", max_chars=80, placeholder="For example, Plant 1")
        new_plot = st.text_input("Plot or field name", max_chars=80, placeholder="For example, Tomato bed A")
        assign_grid = st.checkbox("Place this plant on the simple field map")
        grid_row = st.number_input("Map row", min_value=1, max_value=100, value=1, disabled=not assign_grid)
        grid_column = st.number_input(
            "Map column", min_value=1, max_value=100, value=1, disabled=not assign_grid
        )
        create_submitted = st.form_submit_button("Create timeline")
    if create_submitted:
        try:
            create_tracked_plant(
                new_name,
                plot_name=new_plot or None,
                grid_row=int(grid_row) if assign_grid else None,
                grid_column=int(grid_column) if assign_grid else None,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("Timeline created.")
            st.rerun()

plants = fetch_tracked_plants()
if not plants:
    st.info("Create a plant or plot timeline to add dated follow-up observations.")
    stop_with_footer()

plant_by_id = {int(plant["id"]): plant for plant in plants}
selected_id = st.selectbox(
    "Choose a plant or plot",
    options=list(plant_by_id),
    format_func=lambda plant_id: (
        f"{plant_by_id[plant_id]['name']} · "
        f"{plant_by_id[plant_id]['observation_count']} observation(s)"
    ),
)
selected_plant = plant_by_id[selected_id]

labels = load_labels(LABELS_PATH)
with st.expander("Add a follow-up observation", expanded=not bool(selected_plant["observation_count"])):
    record_type_label = st.selectbox(
        "Record provenance",
        ["User observation", "Expert confirmation (reported by user)"],
        help="An expert confirmation is entered by the user and is not independently authenticated by CropShield.",
    )
    record_type = (
        "expert_confirmed"
        if record_type_label == "Expert confirmation (reported by user)"
        else "user_observation"
    )
    if record_type == "expert_confirmed":
        st.info("Choose this only after a human expert has confirmed the condition.")
    with st.form("add_plant_observation", clear_on_submit=True):
        observed_on = st.date_input("Observation date", value=date.today())
        notes = st.text_area(
            "What did you observe?",
            max_chars=2000,
            placeholder="Describe visible symptoms or changes in your own words.",
        )
        known_class = st.selectbox(
            (
                "Optional class label reported by the expert"
                if record_type == "expert_confirmed"
                else "Optional model class from a separate scan"
            ),
            options=["Not recorded"] + labels,
            help=(
                "This entry is user-reported and is not independently authenticated."
                if record_type == "expert_confirmed"
                else "A class is included only if you select it. It is not a confirmed diagnosis."
            ),
        )
        if record_type == "user_observation":
            record_confidence = st.checkbox(
                "Record the model confidence shown for that scan",
                value=False,
                help="This is a model probability, not diagnostic certainty.",
            )
        else:
            record_confidence = False
            st.caption("No model-confidence value is recorded for an expert-confirmation entry.")
        confidence_percent = (
            st.slider("Model confidence (%)", min_value=0, max_value=100, value=50)
            if record_confidence
            else None
        )
        retain_photo = st.checkbox(
            "I explicitly agree to retain this follow-up photo locally for this timeline",
            value=False,
        )
        followup_photo = (
            st.file_uploader(
                "Optional follow-up photo (JPG, JPEG, or PNG; up to 8 MB)",
                type=["jpg", "jpeg", "png"],
                key="timeline_photo",
            )
            if retain_photo
            else None
        )
        if record_type == "user_observation":
            affected_status = st.selectbox(
                "Your observation: does this plant appear affected?",
                ["Not recorded", "Yes — user observed affected", "No — user observed no symptoms"],
                help="This is your observation, separate from any AI prediction.",
            )
        else:
            affected_status = "Not recorded"
        if followup_photo is not None:
            st.caption("The photo is retained only after you submit this form.")
        submitted = st.form_submit_button("Save observation")

    if submitted:
        if not notes.strip():
            st.error("Add a short observation before saving.")
        else:
            photo = None
            if followup_photo is not None:
                valid, message, _width, _height = validate_upload(followup_photo)
                if not valid:
                    st.error(message)
                else:
                    try:
                        followup_photo.seek(0)
                        with Image.open(followup_photo) as opened_image:
                            photo = opened_image.convert("RGB")
                    except (OSError, UnidentifiedImageError, ValueError):
                        st.error("The follow-up photo could not be decoded.")
                        photo = None
                        valid = False
            else:
                valid = True
            if valid:
                try:
                    add_plant_observation(
                        selected_id,
                        observed_on,
                        notes,
                        predicted_class=None if known_class == "Not recorded" else known_class,
                        confidence=confidence_percent / 100 if confidence_percent is not None else None,
                        retained_image=photo,
                        record_type=record_type,
                        user_marked_affected=(
                            None
                            if affected_status == "Not recorded" or record_type != "user_observation"
                            else affected_status.startswith("Yes")
                        ),
                    )
                except (ValueError, FileNotFoundError, OSError) as error:
                    st.error(str(error))
                else:
                    saved_message = (
                        "User-reported expert confirmation saved locally"
                        if record_type == "expert_confirmed"
                        else "Observation saved locally"
                    )
                    st.success(
                        saved_message
                        + (" with its explicitly retained photo." if photo is not None else ".")
                    )
                    st.rerun()

observations = fetch_plant_observations(selected_id)
plot_names = sorted({str(plant["plot_name"]) for plant in plants if plant["plot_name"]})
if plot_names:
    st.subheader("Field map and spread signals")
    selected_plot = st.selectbox("Choose a field or plot", plot_names)
    map_rows = fetch_plot_map(selected_plot)
    spread_rows = fetch_possible_plot_spread(selected_plot)
    if spread_rows:
        for spread in spread_rows:
            st.warning(
                f"Possible spread risk: {spread['scan_count']} AI scans across "
                f"{spread['plant_count']} plant(s) in **{selected_plot}** recorded "
                f"**{spread['predicted_class']}**. This is a pattern in unconfirmed model predictions, "
                "not evidence of confirmed disease spread."
            )
    else:
        st.info(
            "No repeated same-class disease predictions in this plot are recorded yet. "
            "This is not a disease-risk assessment."
        )
    placed_plants = [
        {
            **row,
            "Map status": " · ".join(
                [
                    *(
                        ["User observation: affected"]
                        if row["user_marked_affected"] == 1
                        else ["User observation: no symptoms recorded"]
                        if row["user_marked_affected"] == 0
                        else []
                    ),
                    *(
                        [
                            "Expert confirmation reported by user "
                            "(not independently verified): "
                            f"{row['latest_expert_confirmation']}"
                        ]
                        if row["latest_expert_confirmation"]
                        else []
                    ),
                    *(
                        [f"AI prediction (not confirmed): {row['latest_ai_prediction']}"]
                        if row["latest_ai_prediction"]
                        else []
                    ),
                ]
            )
            or "No recorded affected status",
            "Latest AI prediction": row["latest_ai_prediction"] or "Not recorded",
            "User-reported expert confirmation": row["latest_expert_confirmation"] or "Not recorded",
        }
        for row in map_rows
        if row["grid_row"] is not None and row["grid_column"] is not None
    ]
    if placed_plants:
        field_map = px.scatter(
            pd.DataFrame(placed_plants),
            x="grid_column",
            y="grid_row",
            color="Map status",
            hover_name="name",
            hover_data=["Latest AI prediction", "User-reported expert confirmation"],
            title=f"Plant positions in {selected_plot} (user observations and AI predictions are distinct)",
        )
        field_map.update_yaxes(autorange="reversed", dtick=1, title="Map row")
        field_map.update_xaxes(dtick=1, title="Map column")
        field_map.update_traces(marker={"size": 18})
        st.plotly_chart(field_map, width="stretch")
        st.caption("Map positions are user-entered grid coordinates, not GPS or a surveyed field boundary.")
    unplaced_count = len(map_rows) - len(placed_plants)
    if unplaced_count:
        st.caption(f"{unplaced_count} plant(s) in this plot do not have map coordinates.")

if observations:
    st.subheader(f"Timeline: {selected_plant['name']}")
    confidence_rows = [
        {
            "Date": pd.to_datetime(observation["observed_on"]),
            "Model confidence": float(observation["confidence"]),
            "Predicted class": observation["predicted_class"] or "Not recorded",
            "Observation": observation["notes"],
        }
        for observation in observations
        if observation["confidence"] is not None
    ]
    if confidence_rows:
        chart = px.line(
            pd.DataFrame(confidence_rows),
            x="Date",
            y="Model confidence",
            markers=True,
            hover_data=["Predicted class", "Observation"],
            title="Recorded model confidence by visit",
        )
        chart.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(chart, width="stretch")
        st.caption(
            "Confidence can vary with the photo and model. Changes do not establish that symptoms improved, "
            "worsened, or responded to treatment."
        )

    for observation in observations:
        title = observation["observed_on"]
        if observation["predicted_class"]:
            title += f" · {observation['predicted_class']}"
        with st.expander(title, expanded=True):
            st.caption(
                {
                    "ai_prediction": "AI prediction — not confirmed",
                    "expert_confirmed": "Expert confirmation reported by user — not independently verified",
                    "user_observation": "User observation",
                }.get(observation["record_type"], "Observation")
            )
            st.write(observation["notes"])
            if observation["user_marked_affected"] is not None:
                st.caption(
                    "User marked this plant as affected."
                    if observation["user_marked_affected"]
                    else "User marked no symptoms observed."
                )
            if observation["confidence"] is not None:
                st.metric("Recorded model confidence", f"{observation['confidence']:.1%}")
            if observation["trust_level"]:
                st.metric("Recorded heuristic trust level", observation["trust_level"])
            if observation["image_filename"]:
                try:
                    image_path = get_plant_timeline_image_path(observation["image_filename"])
                    st.image(str(image_path), caption="Photo explicitly retained for this timeline", width="stretch")
                except FileNotFoundError as error:
                    st.warning(str(error))
    retained_observations = [
        observation for observation in observations if observation["image_filename"]
    ]
    if len(retained_observations) >= 2:
        with st.expander("Compare retained timeline photos"):
            photo_options = [int(item["id"]) for item in retained_observations]

            def format_photo_option(item_id: int) -> str:
                row = next(item for item in retained_observations if int(item["id"]) == item_id)
                return f"{row['observed_on']} · {row['predicted_class'] or 'Field note'}"

            photo_col1, photo_col2 = st.columns(2)
            first_photo_id = photo_col1.selectbox(
                "Earlier or first photo",
                photo_options,
                format_func=format_photo_option,
                key=f"compare_first_{selected_id}",
            )
            second_photo_options = [item for item in photo_options if item != first_photo_id]
            second_photo_id = photo_col2.selectbox(
                "Later or second photo",
                second_photo_options,
                format_func=format_photo_option,
                key=f"compare_second_{selected_id}",
            )
            compare_columns = st.columns(2)
            for column, photo_id in zip(compare_columns, (first_photo_id, second_photo_id), strict=True):
                photo_row = next(item for item in retained_observations if int(item["id"]) == photo_id)
                try:
                    photo_path = get_plant_timeline_image_path(photo_row["image_filename"])
                    with Image.open(photo_path) as saved_photo:
                        column.image(
                            saved_photo,
                            caption=f"{photo_row['observed_on']} · {photo_row['predicted_class'] or 'Field note'}",
                            width="stretch",
                        )
                except (FileNotFoundError, OSError) as error:
                    column.warning(str(error))
else:
    st.info("No observations yet. Add a dated note to start this plant's timeline.")

with st.expander("Prepare an expert-verification report"):
    st.caption(
        "The report is generated locally and is not submitted or sent anywhere. "
        "No heatmap is available. Include a location or retained photo only if you choose."
    )
    if not observations:
        st.info("Add an observation before preparing an expert report.")
    else:
        selected_observation_id = st.selectbox(
            "Prediction or observation to highlight",
            options=[int(item["id"]) for item in observations],
            format_func=lambda item_id: next(
                f"{next(row for row in observations if int(row['id']) == item_id)['observed_on']} · "
                f"{next(row for row in observations if int(row['id']) == item_id)['record_type'].replace('_', ' ')} · "
                f"{next(row for row in observations if int(row['id']) == item_id)['predicted_class'] or 'no model label'}"
                for row in observations
                if int(row["id"]) == item_id
            ),
        )
        symptoms = st.text_area(
            "Observed symptoms to share",
            max_chars=2000,
            key=f"expert_symptoms_{selected_id}",
        )
        review_priority = st.selectbox(
            "Review priority",
            ["Low — expert review requested", "Preliminary — not verified"],
            key=f"expert_priority_{selected_id}",
        )
        location = st.text_input(
            "Optional location",
            max_chars=200,
            key=f"expert_location_{selected_id}",
            help="Location is optional and appears in the downloaded report only.",
        )
        selected_observation = next(
            row for row in observations if int(row["id"]) == selected_observation_id
        )
        include_photo = st.checkbox(
            "Include the already-retained photo from this selected record",
            value=False,
            disabled=not bool(selected_observation["image_filename"]),
            key=f"expert_photo_{selected_id}",
        )
        if st.button("Prepare report locally", key=f"prepare_expert_report_{selected_id}"):
            try:
                selected_image = None
                if include_photo:
                    image_path = get_plant_timeline_image_path(selected_observation["image_filename"])
                    with Image.open(image_path) as opened_image:
                        selected_image = opened_image.convert("RGB")
                st.session_state[f"expert_report_{selected_id}"] = build_expert_request(
                    plant_name=str(selected_plant["name"]),
                    plot_name=str(selected_plant["plot_name"]),
                    observations=observations,
                    selected_observation_id=selected_observation_id,
                    observed_symptoms=symptoms,
                    trust_level=review_priority,
                    location=location or None,
                    include_selected_photo=include_photo,
                    image=selected_image,
                )
            except (OSError, ValueError, FileNotFoundError) as error:
                st.error(str(error))
        expert_report = st.session_state.get(f"expert_report_{selected_id}")
        if expert_report:
            st.download_button(
                "Download expert-verification report",
                data=expert_report,
                file_name="cropshield-expert-verification.html",
                mime="text/html",
                key=f"download_expert_report_{selected_id}",
            )

st.divider()
st.subheader("Remove an observation")
deletable_observations = [
    observation
    for observation in observations
]
if deletable_observations:
    observation_to_delete = st.selectbox(
        "Choose an observation to remove",
        options=[int(observation["id"]) for observation in deletable_observations],
        format_func=lambda observation_id: next(
            f"{observation['observed_on']} · "
            f"{observation['predicted_class'] or 'Field note'} · "
            f"{observation['notes'][:60]}"
            for observation in deletable_observations
            if int(observation["id"]) == observation_id
        ),
    )
    confirm_observation_delete = st.checkbox(
        "I understand this permanently removes the selected observation and its retained photo."
    )
    if st.button("Delete selected observation", disabled=not confirm_observation_delete):
        delete_plant_observation(selected_id, observation_to_delete)
        st.success("Observation and any retained photo deleted.")
        st.rerun()
else:
    st.info("There are no observations to remove.")

st.divider()
st.subheader("Delete this timeline")
st.caption("This permanently deletes this plant's observation records and any photos retained for it.")
if st.button("Delete selected plant timeline", type="secondary"):
    delete_tracked_plant(selected_id)
    st.success("Plant timeline and retained photos deleted.")
    st.rerun()

render_about_footer()
