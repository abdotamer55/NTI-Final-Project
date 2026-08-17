import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, set_dataset
from utils.pipeline_manager import detect_problem_type
from utils.feature_engineering import (
    build_feature_engineering_recommendations,
    apply_feature_recommendations,
    MAX_TOP_FEATURES,
    CV_FOLDS,
    MIN_IMPROVEMENT,
)


def render():
    page_header(
        "⚙️ Feature Engineering & Selection Studio",
        "Automated cross-validation guided feature engineering recommendation system and manual feature creation suite."
    )

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    main_tabs = st.tabs([
        "🤖 Automated Feature Engineering Engine",
        "📐 Math & Transforms",
        "✖️ Interaction Features",
        "📊 Binning & Discretization",
        "🗓️ Datetime Extraction",
        "🎯 Manual Feature Selection Suite",
    ])

    # =========================================================================
    # TAB 1: AUTOMATED FEATURE ENGINEERING RECOMMENDATION ENGINE
    # =========================================================================
    with main_tabs[0]:
        st.subheader("🚀 Performance-Guided Feature Engineering Engine")
        st.write(
            "This engine trains a baseline model, analyzes feature importances, generates candidate features "
            "(Log, Polynomial, Interactions, Ratios, Binning, Datetime), evaluates each candidate using Cross-Validation, "
            "and recommends only features that empirically improve model accuracy/R2."
        )

        all_cols = df.columns.tolist()

        c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 1.2])
        with c1:
            default_idx = len(all_cols) - 1 if all_cols else 0
            target_col = st.selectbox("Select Target Column", all_cols, index=default_idx, key="fe_engine_target")
        with c2:
            top_n = st.number_input("Top Features (N)", min_value=2, max_value=30, value=MAX_TOP_FEATURES, key="fe_top_n")
        with c3:
            min_imp = st.number_input("Min Improvement", min_value=0.001, max_value=0.1, value=MIN_IMPROVEMENT, step=0.001, format="%.3f", key="fe_min_imp")
        with c4:
            cv_f = st.slider("CV Folds", min_value=2, max_value=10, value=CV_FOLDS, key="fe_cv_folds")

        if st.button("⚡ Run Feature Engineering Analysis & Impact Testing", type="primary", use_container_width=True, key="btn_run_fe_engine"):
            with st.spinner("Training baseline model, generating candidate features, and testing cross-validation impact..."):
                try:
                    fe_results = build_feature_engineering_recommendations(
                        df,
                        target_col=target_col,
                        top_n_features=top_n,
                        cv_folds=cv_f,
                        min_improvement=min_imp,
                        random_state=42
                    )
                    st.session_state.fe_engine_results = fe_results
                    st.session_state.fe_target_col = target_col
                    st.success("Feature Engineering Analysis completed successfully!")
                except Exception as e:
                    st.error(f"Feature Engineering Analysis failed: {e}")

        # Render results if available
        results = st.session_state.get("fe_engine_results")
        current_fe_target = st.session_state.get("fe_target_col")

        if results and current_fe_target == target_col:
            st.markdown("---")

            baseline = results["baseline"]
            prob_type = baseline["problem_type"]

            # 1. Summary Statistics Cards
            st.markdown("### 📊 Engine Performance & Summary Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Original Features", results["total_original_features"])
            m2.metric("Generated Candidates", results["total_candidates"])
            m3.metric("Tested Candidates", results["total_tested"])
            m4.metric("Recommended Features", results["recommended_count"])

            final_feat_cnt = len(df.columns) + results["recommended_count"]
            m5.metric("Projected Final Features", final_feat_cnt)

            st.markdown("#### 🏁 Baseline Model Performance")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if prob_type == "Classification":
                    st.metric("Baseline Accuracy", f"{baseline['accuracy'] * 100:.2f}%")
                else:
                    st.metric("Baseline R² Score", f"{baseline['r2']:.4f}")
            with b_col2:
                if prob_type == "Classification":
                    st.metric("Baseline F1 Score", f"{baseline['f1']:.4f}")
                else:
                    st.metric("Baseline RMSE", f"{baseline['rmse']:.4f}")

            # 2. Feature Importance Chart/Table
            st.markdown("---")
            st.markdown("### 🌲 Feature Importance Analysis (Top Features for Candidates)")
            imp_df = results["importance_df"]
            if not imp_df.empty:
                f_ic1, f_ic2 = st.columns([2, 1])
                with f_ic1:
                    fig_imp = px.bar(
                        imp_df.head(top_n),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Top {top_n} Features by Tree Importance ({prob_type})",
                        color="Importance",
                        color_continuous_scale="Viridis"
                    )
                    fig_imp.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#f8fafc",
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                with f_ic2:
                    st.dataframe(imp_df.head(top_n), use_container_width=True, hide_index=True)

            # 3. Tested Feature Impact Results Table
            st.markdown("---")
            st.markdown("### 🧪 Tested Feature Impact Results")
            tested_list = results["tested_results"]
            if tested_list:
                table_data = []
                for item in tested_list:
                    imp_val = item["improvement"]
                    imp_str = f"+{imp_val:.4f}" if imp_val >= 0 else f"{imp_val:.4f}"
                    status = "✅ Recommended" if item["is_recommended"] else "❌ Rejected"
                    table_data.append({
                        "Feature Name": item["feature_name"],
                        "Type": item["feature_type"],
                        "Source Column(s)": ", ".join(item["source_columns"]),
                        "Baseline Score": f"{item['baseline_score']:.4f}",
                        "New Score": f"{item['new_score']:.4f}",
                        "Improvement": imp_str,
                        "Status": status
                    })

                t_df = pd.DataFrame(table_data)

                # Filter option for table
                show_all = st.checkbox("Show all tested candidates (including rejected)", value=False, key="chk_show_all_tested")
                if not show_all:
                    display_t_df = t_df[t_df["Status"] == "✅ Recommended"]
                else:
                    display_t_df = t_df

                st.dataframe(display_t_df, use_container_width=True, hide_index=True)

            # 4. Feature Engineering Recommendations & Application
            st.markdown("---")
            st.markdown("### 💡 Recommended Feature Engineering Actions")

            recs = results["recommendations"]
            if not recs:
                st.info("No candidate feature met the minimum performance improvement threshold over baseline.")
            else:
                st.success(f"Found **{len(recs)}** recommended features that improve model performance!")

                rec_cols = st.columns([1, 1])
                with rec_cols[0]:
                    if st.button("✨ Apply All Recommended Features", type="primary", use_container_width=True, key="btn_apply_all_fe_recs"):
                        updated_df = apply_feature_recommendations(df, recs, target_col=target_col)
                        applied_names = [r["feature_name"] for r in recs]
                        set_dataset(updated_df, action_description=f"Applied {len(recs)} recommended features: {', '.join(applied_names)}")
                        st.session_state.fe_engine_results = None
                        st.success(f"Applied all {len(recs)} recommended features!")
                        st.rerun()

                # Multi-select list for selective application
                selected_rec_names = st.multiselect(
                    "Select Features to Apply",
                    [r["feature_name"] for r in recs],
                    default=[r["feature_name"] for r in recs],
                    key="multisel_fe_recs"
                )

                if st.button("📌 Apply Selected Features", use_container_width=True, key="btn_apply_selected_fe_recs"):
                    selected_rec_objects = [r for r in recs if r["feature_name"] in selected_rec_names]
                    if selected_rec_objects:
                        updated_df = apply_feature_recommendations(df, selected_rec_objects, target_col=target_col)
                        set_dataset(updated_df, action_description=f"Applied {len(selected_rec_objects)} selected features: {', '.join(selected_rec_names)}")
                        st.session_state.fe_engine_results = None
                        st.success(f"Applied {len(selected_rec_objects)} features successfully!")
                        st.rerun()

                st.markdown("#### Individual Recommendations Details")
                for rec in recs:
                    sev_badge = "🔴 High Impact" if rec["severity"] == "high" else ("🟡 Medium Impact" if rec["severity"] == "medium" else "🟢 Low Impact")
                    with st.expander(f"{sev_badge} | {rec['feature_name']} ({rec['feature_type']}) - Improvement: +{rec['improvement']:.4f}", expanded=True):
                        st.write(rec["recommendation"])
                        r1, r2, r3 = st.columns(3)
                        r1.write(f"**Baseline Score:** {rec['baseline_score']:.4f}")
                        r2.write(f"**New CV Score:** {rec['new_score']:.4f}")
                        r3.write(f"**Improvement:** +{rec['improvement']:.4f}")

                        if st.button(f"Apply {rec['feature_name']}", key=f"btn_apply_single_{rec['id']}"):
                            updated_df = apply_feature_recommendations(df, [rec], target_col=target_col)
                            set_dataset(updated_df, action_description=f"Applied feature '{rec['feature_name']}'")
                            st.session_state.fe_engine_results = None
                            st.success(f"Applied feature '{rec['feature_name']}' successfully!")
                            st.rerun()

            # 5. Dataset Preview (Before vs After)
            st.markdown("---")
            st.markdown("### 📋 Dataset Preview (Before vs Projected After)")
            pv1, pv2 = st.columns(2)
            with pv1:
                st.markdown("#### Current Dataset")
                st.dataframe(df.head(5), use_container_width=True)
            with pv2:
                st.markdown("#### Projected Dataset Preview (If All Recommended Applied)")
                if recs:
                    preview_df = apply_feature_recommendations(df, recs, target_col=target_col)
                    st.dataframe(preview_df.head(5), use_container_width=True)
                else:
                    st.dataframe(df.head(5), use_container_width=True)

    # =========================================================================
    # TAB 2: MANUAL MATH & TRANSFORMS
    # =========================================================================
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    with main_tabs[1]:
        st.subheader("Mathematical Feature Transformations")
        if numeric_columns:
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                target_trans_col = st.selectbox("Select Feature", numeric_columns, key="trans_col")
            with tc2:
                trans_type = st.selectbox(
                    "Transformation Type",
                    ["Log1p (log(1+x))", "Square Root (sqrt(x))", "Square (x^2)", "Reciprocal (1/(x+eps))", "Standard Z-Score"],
                    key="trans_type"
                )
            with tc3:
                new_col_name = st.text_input("New Column Name", value=f"{target_trans_col}_transformed", key="trans_new_name")

            if st.button("Apply Transformation", type="primary", key="btn_trans"):
                out = df.copy()
                col_data = out[target_trans_col].clip(lower=0)

                if "Log1p" in trans_type:
                    out[new_col_name] = np.log1p(col_data)
                elif "Square Root" in trans_type:
                    out[new_col_name] = np.sqrt(col_data)
                elif "Square" in trans_type:
                    out[new_col_name] = out[target_trans_col] ** 2
                elif "Reciprocal" in trans_type:
                    out[new_col_name] = 1.0 / (out[target_trans_col] + 1e-5)
                elif "Standard" in trans_type:
                    std = out[target_trans_col].std()
                    out[new_col_name] = (out[target_trans_col] - out[target_trans_col].mean()) / (std if std != 0 else 1.0)

                set_dataset(out, action_description=f"Engineered feature '{new_col_name}' from '{target_trans_col}'")
                st.success(f"New feature '{new_col_name}' created successfully.")
                st.rerun()
        else:
            st.info("No numeric columns available for transformation.")

    # =========================================================================
    # TAB 3: INTERACTION FEATURES
    # =========================================================================
    with main_tabs[2]:
        st.subheader("Cross-Feature Interaction Terms")
        if len(numeric_columns) >= 2:
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                feat_a = st.selectbox("Feature A", numeric_columns, index=0, key="inter_a")
            with ic2:
                op_type = st.selectbox("Operation", ["Multiply (A * B)", "Divide (A / (B + eps))", "Add (A + B)", "Subtract (A - B)"], key="inter_op")
            with ic3:
                feat_b = st.selectbox("Feature B", numeric_columns, index=min(1, len(numeric_columns)-1), key="inter_b")

            inter_col_name = st.text_input("New Feature Name", value=f"{feat_a}_{op_type[0].lower()}_{feat_b}", key="inter_new_name")

            if st.button("Generate Interaction Feature", type="primary", key="btn_inter"):
                out = df.copy()
                if "Multiply" in op_type:
                    out[inter_col_name] = out[feat_a] * out[feat_b]
                elif "Divide" in op_type:
                    out[inter_col_name] = out[feat_a] / (out[feat_b] + 1e-5)
                elif "Add" in op_type:
                    out[inter_col_name] = out[feat_a] + out[feat_b]
                elif "Subtract" in op_type:
                    out[inter_col_name] = out[feat_a] - out[feat_b]

                set_dataset(out, action_description=f"Generated interaction feature '{inter_col_name}'")
                st.success(f"Interaction feature '{inter_col_name}' generated.")
                st.rerun()
        else:
            st.info("Requires at least two numeric features.")

    # =========================================================================
    # TAB 4: BINNING & DISCRETIZATION
    # =========================================================================
    with main_tabs[3]:
        st.subheader("Feature Binning & Quantile Discretization")
        if numeric_columns:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bin_col = st.selectbox("Numeric Column", numeric_columns, key="bin_col_sel")
            with bc2:
                num_bins = st.slider("Number of Bins", 2, 10, 4, key="bin_cnt")
            with bc3:
                bin_strategy = st.selectbox("Binning Strategy", ["Equal Width", "Quantile / Equal Frequency"], key="bin_strat")

            if st.button("Apply Binning", type="primary", key="btn_bin"):
                out = df.copy()
                bin_col_name = f"{bin_col}_binned"
                try:
                    if "Equal Width" in bin_strategy:
                        out[bin_col_name] = pd.cut(out[bin_col], bins=num_bins, labels=False)
                    else:
                        out[bin_col_name] = pd.qcut(out[bin_col], q=num_bins, labels=False, duplicates="drop")

                    set_dataset(out, action_description=f"Binned feature '{bin_col}' into {num_bins} bins")
                    st.success(f"Created binned feature '{bin_col_name}'.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Unable to bin column: {err}")
        else:
            st.info("No numeric columns available for binning.")

    # =========================================================================
    # TAB 5: DATETIME EXTRACTION
    # =========================================================================
    with main_tabs[4]:
        st.subheader("Date & Time Feature Extraction")
        date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or df[c].dtype == "datetime64[ns]"]
        if not date_candidates:
            date_candidates = df.columns.tolist()

        dt_col = st.selectbox("Select Date Column", date_candidates, key="dt_col_sel")
        parts = st.multiselect("Extract Components", ["Year", "Month", "Day", "Day of Week", "Is Weekend", "Quarter"], default=["Year", "Month", "Day"], key="dt_parts")

        if st.button("Extract Datetime Features", type="primary", key="btn_dt"):
            out = df.copy()
            dt_series = pd.to_datetime(out[dt_col], errors="coerce")

            for part in parts:
                if part == "Year":
                    out[f"{dt_col}_year"] = dt_series.dt.year
                elif part == "Month":
                    out[f"{dt_col}_month"] = dt_series.dt.month
                elif part == "Day":
                    out[f"{dt_col}_day"] = dt_series.dt.day
                elif part == "Day of Week":
                    out[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek
                elif part == "Is Weekend":
                    out[f"{dt_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                elif part == "Quarter":
                    out[f"{dt_col}_quarter"] = dt_series.dt.quarter

            set_dataset(out, action_description=f"Extracted datetime components from '{dt_col}'")
            st.success("Datetime features extracted.")
            st.rerun()

    # =========================================================================
    # TAB 6: FEATURE SELECTION SUITE
    # =========================================================================
    with main_tabs[5]:
        st.subheader("🎯 Manual Feature Selection Suite")
        st.write("Score and select the most relevant features to optimize machine learning performance.")

        sel_tab1, sel_tab2 = st.tabs(["📉 Collinearity & Variance Filter", "🌲 Random Forest Importance"])

        # Collinearity
        with sel_tab1:
            st.markdown("#### Drop High Collinearity Features")
            col_thresh = st.slider("Correlation Threshold (Drop pairs higher than)", 0.70, 0.99, 0.85, 0.01, key="col_slider")

            if len(numeric_columns) >= 2:
                corr_matrix = df[numeric_columns].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > col_thresh)]

                if to_drop:
                    st.warning(f"Features exceeding {col_thresh} correlation: **{', '.join(to_drop)}**")
                    if st.button(f"🗑️ Drop {len(to_drop)} Collinear Features", key="btn_drop_coll"):
                        out = df.drop(columns=to_drop).copy()
                        set_dataset(out, action_description=f"Dropped collinear features: {to_drop}")
                        st.success(f"Dropped {len(to_drop)} features.")
                        st.rerun()
                else:
                    st.success("No collinear feature pairs exceed the selected threshold.")
            else:
                st.info("Requires at least two numeric features for correlation filtering.")

        # Tree Importance
        with sel_tab2:
            st.markdown("#### Random Forest Feature Importances")
            target_tree = st.selectbox("Select Target Variable", df.columns.tolist(), index=len(df.columns)-1, key="rf_target")
            X_tree_cols = [c for c in numeric_columns if c != target_tree]

            if X_tree_cols and target_tree:
                clean_tree = df.dropna(subset=[target_tree] + X_tree_cols)
                if len(clean_tree) > 10:
                    prob_tree = detect_problem_type(clean_tree[target_tree])
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) if prob_tree == "Classification" else RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(clean_tree[X_tree_cols], clean_tree[target_tree])

                    imp_df = pd.DataFrame({"Feature": X_tree_cols, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=True)
                    fig_tree = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title=f"Random Forest Feature Importances ({prob_tree})")
                    fig_tree.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Current Features Preview")
    st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    render()
