import streamlit as st
from src.app_state import init_state
from src.ui import load_css, render_navigation

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

init_state()
load_css()

from pages import (
    home, upload, analysis, preprocessing, visualization,
    feature_engineering, machine_learning, experiments,
    prediction, reports
)

PAGES = {
    "Home": home,
    "Upload Data": upload,
    "Analysis": analysis,
    "Preprocessing": preprocessing,
    "Visualization": visualization,
    "Feature Engineering": feature_engineering,
    "Machine Learning": machine_learning,
    "Experiments": experiments,
    "Prediction": prediction,
    "Reports": reports,
}

content = render_navigation()
with content:
    PAGES[st.session_state.selected_page].render()
