"""Shared "About Me" page footer rendered at the bottom of every page."""

import streamlit as st

_FOOTER_HTML = """
<div style="margin-top:2.5rem;padding-top:1.5rem;border-top:2px solid #d8e5d6;text-align:center;">
    <p style="font-size:1.05rem;font-weight:600;color:#166534;margin-bottom:0.25rem;">Punith Kumar AB</p>
    <p style="color:#17231B;margin-bottom:0.25rem;">
        Bachelor of Engineering in Artificial Intelligence and Machine Learning<br>
        G M Institute of Technology, Davanagere
    </p>
    <p style="margin-bottom:0.25rem;">
        <a href="mailto:punithkumar0927@gmail.com" style="color:#166534;text-decoration:none;font-weight:500;">&#9993; punithkumar0927@gmail.com</a>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <a href="https://github.com/punithkumar0927-ctrl" target="_blank" style="color:#166534;text-decoration:none;font-weight:500;">&#128736; GitHub</a>
    </p>
    <p style="color:#647067;font-size:0.8rem;">CropShield AI 2.0 &mdash; honest, explainable plant-health assistance for the field.</p>
</div>
"""


def render_about_footer() -> None:
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)


def stop_with_footer() -> None:
    """Render the footer, then stop the page (for early-exit branches)."""
    render_about_footer()
    st.stop()
