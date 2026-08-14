import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, set_dataset
from utils.dataset_analyzer import analyze_dataset
from utils.recommender import build_recommendations
from utils.preprocessing_utils import (
    impute_column,
    drop_columns,
    remove_duplicates,
    cap_outliers_iqr,
    apply_log1p
)

def render():
    page_header("🔍 Dataset Analysis & Smart Recommendations", "Inspect data quality, health metrics, and execute one-click smart recommendations.")

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    # Perform fresh analysis
    analysis = analyze_dataset(df)
    recommendations = build_recommendations(df, analysis)

    st.session_state.analysis = analysis
    st.session_state.recommendations = recommendations

    # Metric summary grid
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows", f"{analysis['rows']:,}")
    m2.metric("Columns", analysis["columns"])
    m3.metric("Memory", f"{analysis['memory_mb']:.2f} MB")
    m4.metric("Duplicates", analysis["duplicates"])
    m5.metric("Missing Cells", f"{sum(analysis['missing_counts'].values()):,}")
    m6.metric("Outlier Cols", len(analysis["outliers"]))

    # Visual data breakdown
    st.subheader("📊 Dataset Structure Overview")
    v1, v2 = st.columns(2)

    with v1:
        type_df = pd.DataFrame({
            "Type": ["Numeric Features", "Categorical Features"],
            "Count": [len(analysis["numeric_columns"]), len(analysis["categorical_columns"])]
        })
        fig_pie = px.pie(
            type_df, names="Type", values="Count", title="Feature Type Distribution",
            color_discrete_sequence=["#6d5dfc", "#22d3ee"], hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig_pie, use_container_width=True)

    with v2:
        missing_items = {k: v for k, v in analysis["missing_percent"].items() if v > 0}
        if missing_items:
            miss_df = pd.DataFrame({"Column": list(missing_items.keys()), "Missing %": list(missing_items.values())}).head(10)
            fig_bar = px.bar(
                miss_df, x="Missing %", y="Column", orientation="h", title="Top Missing Columns (%)",
                color="Missing %", color_continuous_scale="Reds"
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("🎉 Zero missing values detected across all columns!")

    # Smart Recommendations Section
    st.subheader("🤖 Smart Actionable Recommendations")
    st.write("Review auto-detected issues and click any button below to apply the recommendation directly to your dataset.")

    if not recommendations:
        st.success("✨ No major dataset quality issues detected! Your dataset is clean and ready for feature engineering & model training.")
        return

    # Render actionable recommendation cards
    for idx, item in enumerate(recommendations):
        severity_class = f"{item.get('severity', 'info')}-severity"
        badge_class = item.get("severity", "info")

        st.markdown(
            f'''
            <div class="recommendation {severity_class}">
                <div class="rec-title">
                    <span>{item["issue"]}</span>
                    <span class="rec-badge {badge_class}">{badge_class.upper()}</span>
                    <span style="color:var(--text-muted); font-size:0.85rem; font-weight:normal;">Target: <b>{item["column"]}</b></span>
                </div>
                <div class="rec-body">{item["recommendation"]}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        btn_col, _ = st.columns([1.5, 3])
        with btn_col:
            button_key = f"btn_rec_{item['id']}_{idx}"
            if st.button(item["action_label"], type="primary" if item["severity"] == "high" else "secondary", key=button_key):
                act = item["action_type"]
                col = item.get("target_col", item["column"])

                if act == "remove_duplicates":
                    updated = remove_duplicates(df)
                    desc = "Applied Recommendation: Removed Duplicate Rows"
                elif act == "drop_column" or act == "drop_column_col2":
                    updated = drop_columns(df, [col])
                    desc = f"Applied Recommendation: Dropped column '{col}'"
                elif act == "impute_median":
                    updated = impute_column(df, col, strategy="median")
                    desc = f"Applied Recommendation: Median imputation on '{col}'"
                elif act == "impute_mean":
                    updated = impute_column(df, col, strategy="mean")
                    desc = f"Applied Recommendation: Mean imputation on '{col}'"
                elif act == "impute_mode":
                    updated = impute_column(df, col, strategy="mode")
                    desc = f"Applied Recommendation: Mode imputation on '{col}'"
                elif act == "cap_outliers_iqr":
                    updated = cap_outliers_iqr(df, col, factor=1.5)
                    desc = f"Applied Recommendation: Capped IQR outliers in '{col}'"
                elif act == "apply_log1p":
                    updated = apply_log1p(df, col)
                    desc = f"Applied Recommendation: Log1p transformation on '{col}'"
                else:
                    updated = df.copy()
                    desc = "No action executed."

                set_dataset(updated, action_description=desc)
                st.success(f"Action executed: {desc}")
                st.rerun()

    st.subheader("📋 Dataset Sample Preview")
    st.dataframe(df.head(20), use_container_width=True)
