from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cropshield.footer import render_about_footer, stop_with_footer
from src.cropshield.history import clear_history, fetch_history

st.set_page_config(page_title="Prediction History", layout="wide")
st.title("Prediction History")
st.caption(
    "History is opt-in and stores prediction metadata, image name, quality measurements, and optional notes "
    "in a local SQLite file; uploaded image bytes are not stored. "
    "Cloud-hosted history may not persist across redeployments. For separately consented follow-up records, "
    "use the Plant Follow-up Timeline page."
)

confirm_clear = st.checkbox("I understand clearing history permanently deletes all prediction records.")
if st.button("Clear history", disabled=not confirm_clear):
    clear_history()
    st.success("Prediction history cleared.")

rows = fetch_history(limit=25)
if not rows:
    st.info("No local prediction history yet. Predictions will appear here once you run inference in the app.")
    stop_with_footer()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
st.dataframe(df, width="stretch")

render_about_footer()
