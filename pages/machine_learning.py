import io
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, get_active_features
from utils.pipeline_manager import detect_problem_type
from utils.model_recommender import (
    get_available_models,
    train_recommended_models,
    train_kmeans_clustering,
    train_unsupervised_model,
    UNSUPERVISED_ALGORITHMS,
    get_class_distribution,
    IMBLEARN_AVAILABLE,
)

BALANCE_OPTIONS = {
    "None (keep as-is)": "none",
    "Random Oversampling": "oversample",
    "Random Undersampling": "undersample",
    "SMOTE (synthetic oversampling)": "smote",
}


def render():
    page_header("🤖 Machine Learning & Clustering Studio", "Train, evaluate, and diagnose machine learning models and unsupervised clustering pipelines.")

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    tabs = st.tabs([
        "🚀 Supervised Model Training",
        "📈 Diagnostics & Confusion Matrix",
        "🧩 Unsupervised K-Means Clustering",
        "💾 Export Model Bundle"
    ])

    # Tab 1: Supervised Training
    with tabs[0]:
        st.subheader("Train & Benchmark Machine Learning Pipelines")
        all_cols = df.columns.tolist()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target = st.selectbox("Select Target Column", all_cols, index=len(all_cols)-1, key="ml_target")
        with c2:
            detected_type = detect_problem_type(df[target].dropna()) if target in df else "Classification"
            st.info(f"Problem Type: **{detected_type}**")
        with c3:
            test_size = st.slider("Test Size Split", 0.10, 0.40, 0.20, 0.05, key="ml_test_size")
        with c4:
            cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, key="ml_cv_folds")

        # --- Class distribution & balancing (classification only) --- #
        balance_key = "none"
        if detected_type == "Classification" and target in df:
            dist_df, ratio = get_class_distribution(df[target].dropna())

            with st.expander(f"📊 Class Distribution (imbalance ratio: {ratio:.1f}x)", expanded=ratio >= 3):
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
                if ratio >= 3:
                    st.warning(
                        f"The largest class is **{ratio:.1f}x** bigger than the smallest. "
                        "This is common in real-world data, but a model trained as-is will likely "
                        "favor the majority class. Consider balancing the training data below."
                    )

                if not IMBLEARN_AVAILABLE:
                    st.caption("⚠️ Install `imbalanced-learn` (`pip install imbalanced-learn`) to enable balancing options below.")

                balance_label = st.selectbox(
                    "Balance training data before fitting",
                    list(BALANCE_OPTIONS.keys()),
                    index=0,
                    key="ml_balance_strategy",
                    disabled=not IMBLEARN_AVAILABLE,
                    help=(
                        "Applied ONLY to the training split, inside the pipeline — the test set "
                        "always stays in its original, real-world distribution."
                    ),
                )
                balance_key = BALANCE_OPTIONS[balance_label]

        available_dict = get_available_models(detected_type)
        selected_model_names = st.multiselect(
            "Select Algorithms to Compare",
            list(available_dict.keys()),
            default=list(available_dict.keys())[:min(4, len(available_dict))],
            key="ml_algos"
        )

        if st.button("🚀 Run Model Training & Evaluation", type="primary", use_container_width=True, key="btn_train_ml"):
            if not selected_model_names:
                st.warning("Select at least one algorithm to train.")
            else:
                try:
                    with st.spinner("Building pipelines and fitting models..."):
                        base_df = df
                        # Extract feature columns excluding target
                        candidate_cols = [c for c in base_df.columns if c != target]
                        clean_features = []
                        for c in candidate_cols:
                            c_lower = c.lower()
                            if c_lower.endswith("_id") or c_lower == "id":
                                continue
                            if pd.api.types.is_string_dtype(base_df[c]) and base_df[c].nunique() > 0.8 * len(base_df):
                                continue
                            clean_features.append(c)

                        sub_df = base_df[clean_features + [target]].copy()

                        prob_type, results, bundles, features, eval_data = train_recommended_models(
                            sub_df, target, selected_models=selected_model_names,
                            test_size=test_size, cv_folds=cv_folds,
                            balance_strategy=balance_key
                        )

                    results_df = pd.DataFrame(results)
                    sort_metric = "Accuracy" if prob_type == "Classification" else "R2 Score"
                    results_df = results_df.sort_values(sort_metric, ascending=False).reset_index(drop=True)

                    st.subheader("🏆 Model Leaderboard")
                    st.dataframe(results_df, use_container_width=True)

                    balance_info = eval_data.get("balance_info")
                    if balance_info:
                        if balance_info["applied"]:
                            st.success(f"Training data was balanced using: **{balance_info['strategy']}** (test set left untouched).")
                        if balance_info["warning"]:
                            st.warning(balance_info["warning"])

                    best_model_name = results_df.iloc[0]["Model"]
                    best_pipeline = bundles[best_model_name]

                    # Save into session state
                    st.session_state.problem_type = prob_type
                    st.session_state.target_column = target
                    st.session_state.best_model_name = best_model_name
                    st.session_state.model_bundle = {
                        "pipeline": best_pipeline,
                        "features": features,
                        "target": target,
                        "problem_type": prob_type,
                        "eval_data": eval_data,
                        "all_bundles": bundles,
                        "leaderboard": results_df
                    }

                    st.session_state.experiments.append({
                        "target": target,
                        "problem_type": prob_type,
                        "results": results_df,
                        "best_model": best_model_name
                    })

                    st.success(f"🏆 Best Model Selected: **{best_model_name}** ({sort_metric}: {results_df.iloc[0][sort_metric]:.4f}). Ready for Prediction & Reports!")
                except ValueError as e:
                    st.error(str(e))

    # Tab 2: Diagnostics
    with tabs[1]:
        st.subheader("Model Diagnostic Charts")
        bundle = st.session_state.get("model_bundle")
        if not bundle:
            st.info("Please train models in Tab 1 first.")
        else:
            best_model_name = st.session_state.best_model_name
            pipeline = bundle["pipeline"]
            eval_data = bundle["eval_data"]
            prob_type = bundle["problem_type"]
            X_test, y_test = eval_data["X_test"], eval_data["y_test"]

            st.write(f"Diagnostic Analysis for **{best_model_name}** ({prob_type})")
            preds = pipeline.predict(X_test)

            if prob_type == "Classification":
                d1, d2 = st.columns(2)
                with d1:
                    labels = np.unique(y_test)
                    cm = confusion_matrix(y_test, preds)
                    fig_cm = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels], text_auto=True, title="Confusion Matrix Heatmap", color_continuous_scale="Blues")
                    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_cm, use_container_width=True)

                with d2:
                    if len(labels) == 2 and hasattr(pipeline, "predict_proba"):
                        try:
                            proba = pipeline.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, proba, pos_label=labels[1])
                            roc_auc = auc(fpr, tpr)
                            fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.3f})", labels=dict(x="False Positive Rate", y="True Positive Rate"))
                            fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                            fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                            st.plotly_chart(fig_roc, use_container_width=True)
                        except Exception:
                            st.info("ROC Curve not available for multiclass target.")
                    else:
                        st.info("Multiclass classification target.")

            else:
                d1, d2 = st.columns(2)
                with d1:
                    residuals = y_test - preds
                    fig_res = px.scatter(x=preds, y=residuals, labels={"x": "Predicted Values", "y": "Residuals (Actual - Pred)"}, title="Residual Analysis Plot")
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_res, use_container_width=True)

                with d2:
                    fig_act = px.scatter(x=y_test, y=preds, labels={"x": "Actual Values", "y": "Predicted Values"}, title="Actual vs Predicted Values")
                    fig_act.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_act, use_container_width=True)

    # Tab 3: Unsupervised Clustering & Anomaly Detection
    with tabs[2]:
        st.subheader("🧩 Unsupervised Clustering & Anomaly Detection Studio")
        st.write(
            "Train a clustering or anomaly-detection model on your dataset. "
            "After fitting, navigate to **🧩 Unsupervised Prediction** to score new records."
        )

        all_numeric = df.select_dtypes(include=np.number).columns.tolist()
        all_cols_us = df.columns.tolist()

        if not all_cols_us:
            st.info("No features available.")
        else:
            ua1, ua2 = st.columns([2, 2])
            with ua1:
                us_algo = st.selectbox(
                    "Algorithm",
                    UNSUPERVISED_ALGORITHMS,
                    key="us_algo",
                    help="K-Means / Agglomerative: cluster-label assignment. DBSCAN: density clusters. Isolation Forest: anomaly scoring."
                )
            with ua2:
                us_features = st.multiselect(
                    "Features to Use",
                    all_cols_us,
                    default=all_numeric[:min(5, len(all_numeric))],
                    key="us_feats"
                )

            # Per-algorithm hyperparameters
            us_params = {}
            hp1, hp2, hp3 = st.columns(3)
            if us_algo in ("K-Means Clustering", "Agglomerative Clustering"):
                us_params["n_clusters"] = hp1.slider("Number of Clusters (K)", 2, 15, 3, key="us_k")
                if us_algo == "Agglomerative Clustering":
                    us_params["linkage"] = hp2.selectbox(
                        "Linkage Method", ["ward", "complete", "average", "single"], key="us_linkage"
                    )

            elif us_algo == "DBSCAN":
                us_params["eps"]         = hp1.slider("Epsilon (ε) — Neighbourhood Radius", 0.1, 5.0, 0.5, 0.05, key="us_eps")
                us_params["min_samples"] = hp2.slider("Min Samples per Core Point", 2, 20, 5, key="us_min_s")

            elif us_algo == "Isolation Forest (Anomaly Detection)":
                us_params["contamination"] = hp1.slider(
                    "Contamination (expected anomaly fraction)", 0.01, 0.40, 0.05, 0.01, key="us_contam"
                )

            if st.button("🚀 Train Unsupervised Model", type="primary", use_container_width=True, key="btn_us_train"):
                if not us_features:
                    st.warning("Select at least one feature.")
                else:
                    try:
                        with st.spinner(f"Fitting {us_algo}…"):
                            bundle = train_unsupervised_model(df, us_features, algorithm=us_algo, **us_params)
                        st.session_state.unsupervised_bundle = bundle

                        # ── Summary metrics ────────────────────────────────
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Algorithm", us_algo.split("(")[0].strip())
                        m2.metric("Features Used", len(us_features))
                        if bundle["silhouette_score"] is not None:
                            m3.metric("Silhouette Score", f"{bundle['silhouette_score']:.4f}")
                        elif us_algo == "Isolation Forest (Anomaly Detection)":
                            anomaly_count = int((bundle["labels"] == "Anomaly").sum())
                            m3.metric("Anomalies Detected", f"{anomaly_count:,}")
                        elif us_algo == "DBSCAN":
                            noise_count = int((bundle["labels"] == -1).sum())
                            m3.metric("Noise Points", f"{noise_count:,}")

                        # ── PCA Scatter ────────────────────────────────────
                        cluster_df = bundle["cluster_df"]
                        color_col  = "Cluster" if "Cluster" in cluster_df else "Anomaly_Score"
                        title_str  = f"{us_algo} — PCA 2-D Projection"
                        if us_algo == "Isolation Forest (Anomaly Detection)" and "Anomaly_Score" in cluster_df:
                            fig_us = px.scatter(
                                cluster_df, x="PCA1", y="PCA2",
                                color="Anomaly_Score",
                                color_continuous_scale="Reds",
                                hover_data={c: True for c in us_features if c in cluster_df},
                                title=title_str, opacity=0.8
                            )
                        else:
                            fig_us = px.scatter(
                                cluster_df, x="PCA1", y="PCA2",
                                color="Cluster",
                                hover_data={c: True for c in us_features if c in cluster_df},
                                title=title_str, opacity=0.85
                            )
                        fig_us.update_layout(
                            height=480, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc"
                        )
                        st.plotly_chart(fig_us, use_container_width=True)

                        # ── Centroid table ─────────────────────────────────
                        if bundle["cluster_centers"] is not None:
                            st.markdown("#### Cluster Centroid Feature Averages")
                            st.dataframe(bundle["cluster_centers"].style.format({
                                c: "{:.3f}" for c in bundle["cluster_centers"].columns if c != "Cluster"
                            }), use_container_width=True)

                        st.success(
                            f"✅ **{us_algo}** trained successfully. "
                            "Navigate to **🧩 Unsupervised Prediction** in the sidebar to score new records."
                        )

                    except Exception as err:
                        st.error(f"Training failed: {err}")

    # Tab 4: Export Model Bundle
    with tabs[3]:
        st.subheader("Save & Download Model Pipeline")
        bundle = st.session_state.get("model_bundle")
        if not bundle:
            st.info("Train a model first to generate a download package.")
        else:
            best_model_name = st.session_state.best_model_name
            st.write(f"Active Model: **{best_model_name}**")
            st.write(f"Target Feature: **{bundle['target']}**")
            st.write(f"Required Input Features: `{', '.join(bundle['features'])}`")

            # Buffer joblib output
            buf = io.BytesIO()
            joblib.dump(bundle["pipeline"], buf)
            buf.seek(0)

            st.download_button(
                "⬇ Download Trained Pipeline (.pkl)",
                data=buf,
                file_name=f"smartprepml_{best_model_name.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                type="primary",
                use_container_width=True
            )
