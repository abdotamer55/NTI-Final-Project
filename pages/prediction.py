import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset
from utils.prediction_utils import (
    validate_prediction_columns,
    predict_batch,
    make_single_row
)

def render():
    page_header("🎯 Prediction Studio", "Run single instance predictions or upload batch CSV files using your trained pipeline.")

    df = get_dataset()
    if df is not None:
        render_dataset_toolbar()

    bundle = st.session_state.get("model_bundle")

    if not bundle:
        st.warning("⚠️ No model pipeline found. Please go to **Machine Learning** and train a model first.")
        return

    st.success(
        f"Active Model Pipeline: **{st.session_state.best_model_name}** | "
        f"Target Column: **{bundle['target']}** | "
        f"Problem Type: **{bundle['problem_type']}**"
    )

    single_tab, batch_tab = st.tabs([
        "👤 Single Instance Prediction",
        "📦 Batch CSV Prediction"
    ])

    # Tab 1: Single Prediction
    with single_tab:
        st.subheader("Single Record Prediction")
        st.write("Enter feature values below:")

        values = {}
        features = bundle["features"]
        cols = st.columns(min(3, max(1, len(features))))

        raw_df = st.session_state.get("raw_dataset")
        source_df = raw_df if raw_df is not None else df

        for idx, feature in enumerate(features):
            with cols[idx % len(cols)]:
                if source_df is not None and feature in source_df.columns:
                    col_data = source_df[feature]
                    if pd.api.types.is_numeric_dtype(col_data):
                        col_min = col_data.min()
                        col_max = col_data.max()
                        col_med = col_data.median()
                        min_v = float(col_min) if pd.notna(col_min) else 0.0
                        max_v = float(col_max) if pd.notna(col_max) else 100.0
                        def_v = float(col_med) if pd.notna(col_med) else 0.0
                        # Guard against degenerate ranges (e.g. constant column)
                        if max_v <= min_v:
                            max_v = min_v + 1.0
                        values[feature] = st.number_input(
                            f"{feature}",
                            min_value=min_v,
                            max_value=max_v,
                            value=def_v,
                            key=f"pred_num_{feature}"
                        )
                    else:
                        unique_vals = [str(u) for u in col_data.dropna().unique()[:50]]
                        if not unique_vals:
                            unique_vals = ["Unknown"]
                        values[feature] = st.selectbox(f"{feature}", unique_vals, key=f"pred_cat_{feature}")
                else:
                    values[feature] = st.text_input(f"{feature}", value="0", key=f"pred_txt_{feature}")

        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True, key="btn_single_pred"):
            row = make_single_row(values, features, source_df=source_df)
            try:
                prediction = bundle["pipeline"].predict(row)[0]
                st.markdown(
                    f'''
                    <div style="padding:1.5rem; background:linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(109,93,252,0.15) 100%); border:1px solid rgba(16,185,129,0.4); border-radius:16px; text-align:center; margin-top:1rem;">
                        <span style="font-size:0.9rem; color:var(--text-muted);">Predicted Result ({bundle["target"]}):</span>
                        <h1 style="color:#ffffff; font-size:2.8rem; margin:0.3rem 0 0 0;">{prediction}</h1>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            except Exception as error:
                st.error(f"Error executing prediction pipeline: {error}")

    # Tab 2: Batch Prediction
    with batch_tab:
        st.subheader("Batch CSV Prediction")
        uploaded = st.file_uploader("Upload New CSV File for Batch Inference", type=["csv"], key="batch_prediction_file")

        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            missing = validate_prediction_columns(new_df, bundle["features"])

            if missing:
                st.error("Missing required input features in uploaded file: " + ", ".join(missing))
            else:
                st.write("Preview of uploaded batch dataset:")
                st.dataframe(new_df.head(5), use_container_width=True)

                if st.button("⚡ Run Batch Prediction", type="primary", use_container_width=True, key="btn_batch_pred"):
                    try:
                        output = new_df.copy()
                        output["Prediction"] = predict_batch(bundle["pipeline"], new_df, bundle["features"])
                    except Exception as error:
                        st.error(f"Error executing batch prediction pipeline: {error}")
                    else:
                        b1, b2 = st.columns(2)
                        b1.metric("Batch Records Processed", len(output))
                        b2.metric("Features Used", len(bundle["features"]))

                        st.subheader("Batch Predictions Sample")
                        st.dataframe(output.head(20), use_container_width=True)

                        if pd.api.types.is_numeric_dtype(output["Prediction"]):
                            fig_dist = px.histogram(output, x="Prediction", title="Batch Prediction Distribution", color_discrete_sequence=["#22d3ee"])
                        else:
                            value_counts_df = output["Prediction"].value_counts().reset_index()
                            value_counts_df.columns = ["Prediction", "count"]
                            fig_dist = px.bar(value_counts_df, x="Prediction", y="count", title="Batch Prediction Counts", color_discrete_sequence=["#6d5dfc"])

                        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                        st.plotly_chart(fig_dist, use_container_width=True)

                        st.download_button(
                            "⬇ Download Predictions CSV",
                            output.to_csv(index=False).encode(),
                            "predictions_result.csv",
                            "text/csv",
                            type="primary"
                        )
