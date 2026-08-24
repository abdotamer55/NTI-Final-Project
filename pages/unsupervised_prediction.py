"""
Unsupervised Prediction Studio
================================
Lets users apply a fitted unsupervised model (K-Means, Agglomerative,
DBSCAN, Isolation Forest) to:
  - A single manually-entered record
  - An uploaded batch CSV
  - Explore the training clusters interactively
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset
from utils.model_recommender import predict_unsupervised


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _result_badge(label: str, score: float | None = None, algorithm: str = "") -> str:
    """Return a styled HTML card for the prediction result."""
    is_anomaly = algorithm == "Isolation Forest (Anomaly Detection)"
    if is_anomaly:
        colour = "#ef4444" if label == "Anomaly" else "#10b981"
        icon   = "🚨" if label == "Anomaly" else "✅"
        body   = f"{icon} {label}"
        sub    = f"Anomaly Score: <b>{score:.4f}</b>" if score is not None else ""
    else:
        colour = "#6d5dfc"
        icon   = "🧩"
        body   = f"{icon} Cluster {label}"
        sub    = ""

    return f"""
    <div style="
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg,
            {colour}22 0%,
            {colour}11 100%);
        border: 1.5px solid {colour}88;
        border-radius: 18px;
        text-align: center;
        margin-top: 1.2rem;
    ">
        <div style="font-size:0.88rem; color:var(--text-muted); margin-bottom:0.3rem;">
            Prediction Result
        </div>
        <div style="font-size:2.6rem; font-weight:800; color:#ffffff; margin-bottom:0.2rem;">
            {body}
        </div>
        <div style="font-size:0.9rem; color:{colour}; margin-top:0.3rem;">
            {sub}
        </div>
    </div>
    """


def _single_input_form(bundle: dict, source_df: pd.DataFrame | None) -> pd.DataFrame:
    """Render one number-input / selectbox per feature and return a 1-row DataFrame."""
    features = bundle["features"]
    values   = {}
    cols     = st.columns(min(4, max(1, len(features))))

    for idx, feat in enumerate(features):
        with cols[idx % len(cols)]:
            if source_df is not None and feat in source_df.columns:
                col_data = source_df[feat].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    mn  = float(col_data.min())
                    mx  = float(col_data.max())
                    med = float(col_data.median())
                    if mx <= mn:
                        mx = mn + 1.0
                    values[feat] = st.number_input(
                        feat, min_value=mn, max_value=mx, value=med,
                        key=f"us_pred_num_{feat}"
                    )
                else:
                    opts = [str(v) for v in col_data.unique()[:50]]
                    values[feat] = st.selectbox(feat, opts, key=f"us_pred_cat_{feat}")
            else:
                values[feat] = st.number_input(feat, value=0.0, key=f"us_pred_txt_{feat}")

    return pd.DataFrame([values])


# ─────────────────────────────────────────────────────────────────────────────
# Page render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    page_header(
        "🧩 Unsupervised Prediction Studio",
        "Score individual records or upload a batch CSV using your trained clustering / anomaly-detection model."
    )

    df = get_dataset()
    if df is not None:
        render_dataset_toolbar()

    bundle = st.session_state.get("unsupervised_bundle")

    if bundle is None:
        st.warning(
            "⚠️ No unsupervised model found.  \n"
            "Go to **🤖 Machine Learning → Unsupervised Clustering & Anomaly Detection** "
            "and train a model first."
        )
        return

    algorithm  = bundle["algorithm"]
    features   = bundle["features"]
    cluster_df = bundle["cluster_df"]
    is_anomaly = algorithm == "Isolation Forest (Anomaly Detection)"

    # Status banner
    label_col = "Anomaly_Label" if is_anomaly else "Cluster"
    unique_labels = cluster_df["Cluster"].unique() if not is_anomaly else ["Anomaly", "Normal"]
    n_label = len(unique_labels)

    st.markdown(
        f"""
        <div style="background:rgba(109,93,252,0.10); border:1px solid rgba(109,93,252,0.3);
                    border-radius:12px; padding:0.75rem 1.2rem; margin-bottom:1rem;">
            <b>Active Model:</b> <code style="color:var(--secondary);">{algorithm}</code> &nbsp;|&nbsp;
            <b>Features:</b> {len(features)} &nbsp;|&nbsp;
            <b>{"Classes" if is_anomaly else "Clusters"}:</b> {n_label}
            {"&nbsp;|&nbsp; <b>Silhouette Score:</b> " + f"{bundle['silhouette_score']:.4f}"
             if bundle.get("silhouette_score") is not None else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_single, tab_batch, tab_explorer = st.tabs([
        "👤 Single Record Prediction",
        "📦 Batch CSV Prediction",
        "📊 Cluster Explorer"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Single record
    # ─────────────────────────────────────────────────────────────────────────
    with tab_single:
        st.subheader("Single Record Scoring")
        st.caption("Enter feature values below; the model will assign a cluster label or anomaly score.")

        raw = st.session_state.get("raw_dataset")
        source_df = raw if raw is not None else df
        input_row = _single_input_form(bundle, source_df)

        if st.button("🎯 Predict", type="primary", use_container_width=True, key="btn_us_single"):
            try:
                result = predict_unsupervised(bundle, input_row)

                if is_anomaly:
                    pred_label = result["Anomaly_Label"].iloc[0]
                    pred_score = float(result["Anomaly_Score"].iloc[0])
                else:
                    pred_label = result["Cluster"].iloc[0]
                    pred_score = None

                st.markdown(
                    _result_badge(pred_label, pred_score, algorithm),
                    unsafe_allow_html=True
                )

                # Show nearest cluster neighbours from training set
                if not is_anomaly and "PCA1" in cluster_df.columns:
                    st.markdown("##### 🔍 Closest Training Points in this Cluster")
                    same_cluster = cluster_df[cluster_df["Cluster"] == str(pred_label)]
                    st.dataframe(
                        same_cluster[features[:min(6, len(features))]].head(10),
                        use_container_width=True
                    )

            except Exception as err:
                st.error(f"Prediction error: {err}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Batch CSV
    # ─────────────────────────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch CSV Inference")
        st.caption(
            f"Upload a CSV that contains the required feature columns: "
            f"`{', '.join(features[:8])}{'…' if len(features)>8 else ''}`"
        )

        uploaded = st.file_uploader(
            "Upload CSV for Batch Scoring", type=["csv"], key="us_batch_file"
        )

        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                return

            missing = [f for f in features if f not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: `{', '.join(missing)}`")
            else:
                st.write(f"**{len(new_df):,} rows** detected. Preview:")
                st.dataframe(new_df[features].head(5), use_container_width=True)

                if st.button(
                    "⚡ Run Batch Prediction", type="primary",
                    use_container_width=True, key="btn_us_batch"
                ):
                    try:
                        with st.spinner("Scoring records…"):
                            result_df = predict_unsupervised(bundle, new_df)

                        output_col = "Anomaly_Label" if is_anomaly else "Cluster"
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Records Processed", f"{len(result_df):,}")
                        b2.metric("Features Used", len(features))

                        if is_anomaly:
                            anomaly_n = int((result_df["Anomaly_Label"] == "Anomaly").sum())
                            b3.metric("Anomalies Found", f"{anomaly_n:,}")
                        else:
                            b3.metric("Distinct Clusters", result_df["Cluster"].nunique())

                        # Distribution chart
                        if is_anomaly:
                            vc = result_df["Anomaly_Label"].value_counts().reset_index()
                            vc.columns = ["Label", "Count"]
                            fig_b = px.bar(
                                vc, x="Label", y="Count", color="Label",
                                title="Anomaly vs Normal Distribution",
                                color_discrete_map={
                                    "Anomaly": "#ef4444", "Normal": "#10b981"
                                }
                            )
                        else:
                            vc = result_df["Cluster"].value_counts().sort_index().reset_index()
                            vc.columns = ["Cluster", "Count"]
                            fig_b = px.bar(
                                vc, x="Cluster", y="Count", color="Cluster",
                                title="Cluster Assignment Distribution"
                            )
                        fig_b.update_layout(
                            height=350, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
                        )
                        st.plotly_chart(fig_b, use_container_width=True)

                        st.subheader("Sample Results (first 20 rows)")
                        display_cols = features[:min(5, len(features))] + (
                            ["Anomaly_Label", "Anomaly_Score"] if is_anomaly else ["Cluster"]
                        )
                        st.dataframe(result_df[display_cols].head(20), use_container_width=True)

                        # Download
                        st.download_button(
                            "⬇ Download Full Results CSV",
                            result_df.to_csv(index=False).encode(),
                            file_name=f"unsupervised_predictions_{algorithm[:10].lower().replace(' ','_')}.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True,
                            key="btn_us_dl"
                        )

                    except Exception as err:
                        st.error(f"Batch prediction error: {err}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Cluster Explorer
    # ─────────────────────────────────────────────────────────────────────────
    with tab_explorer:
        st.subheader("📊 Training-Set Cluster Explorer")

        if cluster_df is None or cluster_df.empty:
            st.info("No cluster data available. Train a model first.")
            return

        # PCA scatter
        st.markdown("#### PCA 2-D Projection of Training Data")
        if is_anomaly and "Anomaly_Score" in cluster_df.columns:
            fig_exp = px.scatter(
                cluster_df, x="PCA1", y="PCA2",
                color="Anomaly_Score",
                color_continuous_scale="Reds",
                hover_data={c: True for c in features if c in cluster_df},
                title=f"{algorithm} — Training Set Anomaly Map",
                opacity=0.8
            )
        else:
            fig_exp = px.scatter(
                cluster_df, x="PCA1", y="PCA2",
                color="Cluster",
                hover_data={c: True for c in features if c in cluster_df},
                title=f"{algorithm} — Training Set Cluster Map",
                opacity=0.85
            )
        fig_exp.update_layout(
            height=500, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        # Centroid table
        if bundle.get("cluster_centers") is not None:
            st.markdown("#### Cluster Centroid Feature Averages")
            centers = bundle["cluster_centers"]
            num_cols = [c for c in centers.columns if c != "Cluster"]
            st.dataframe(
                centers.style.format({c: "{:.3f}" for c in num_cols}),
                use_container_width=True
            )

        # Per-cluster feature distributions
        st.markdown("#### Per-Cluster Feature Distributions")
        numeric_feats = [f for f in features if pd.api.types.is_numeric_dtype(cluster_df.get(f, pd.Series(dtype=float)))]

        if numeric_feats:
            chosen_feat = st.selectbox(
                "Select Feature to Explore", numeric_feats, key="explorer_feat"
            )
            group_col = "Cluster"

            col_fig1, col_fig2 = st.columns(2)
            with col_fig1:
                fig_box = px.box(
                    cluster_df, x=group_col, y=chosen_feat,
                    color=group_col, points="outliers",
                    title=f"Box Plot: <b>{chosen_feat}</b> by {group_col}"
                )
                fig_box.update_layout(
                    height=400, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col_fig2:
                fig_hist = px.histogram(
                    cluster_df, x=chosen_feat, color=group_col,
                    barmode="overlay", opacity=0.7,
                    title=f"Histogram: <b>{chosen_feat}</b> by {group_col}"
                )
                fig_hist.update_layout(
                    height=400, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Anomaly score distribution (Isolation Forest only)
            if is_anomaly and "Anomaly_Score" in cluster_df.columns:
                st.markdown("#### Anomaly Score Distribution")
                fig_score = px.histogram(
                    cluster_df, x="Anomaly_Score", color="Cluster",
                    barmode="overlay", nbins=50, opacity=0.8,
                    color_discrete_map={"Anomaly": "#ef4444", "Normal": "#10b981"},
                    title="Anomaly Score Distribution (higher = more anomalous)"
                )
                fig_score.update_layout(
                    height=380, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
                )
                st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No numeric features found in the clustering results for distribution plots.")
