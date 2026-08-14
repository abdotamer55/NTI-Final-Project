import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset

def render():
    page_header("🧪 Experiment History & Model Benchmark", "Compare model benchmark runs, inspect metrics, and switch active pipelines.")

    df = get_dataset()
    if df is not None:
        render_dataset_toolbar()

    experiments = st.session_state.get("experiments", [])

    if not experiments:
        st.info("No experiments recorded yet. Train models in the **Machine Learning** tab to build your experiment log.")
        return

    col_btn1, col_btn2 = st.columns([1.5, 4])
    with col_btn1:
        if st.button("🗑️ Clear Experiment History", key="btn_clear_exp"):
            st.session_state.experiments.clear()
            st.rerun()

    for index, exp in enumerate(experiments, start=1):
        st.subheader(f"Experiment {index} — Target: {exp['target']} ({exp['problem_type']})")
        results_df = exp["results"]
        st.dataframe(results_df, use_container_width=True)

        metric_col = "Accuracy" if "Accuracy" in results_df.columns else "R2 Score"
        if metric_col in results_df.columns:
            fig = px.bar(results_df, x="Model", y=metric_col, color="Model", title=f"Model Performance Benchmark ({metric_col})", text_auto=".4f")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
