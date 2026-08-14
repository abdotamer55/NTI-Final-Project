import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, set_dataset
from utils.pipeline_manager import detect_problem_type

def render():
    page_header("⚙️ Feature Engineering & Selection Studio", "Transform features, engineer interaction terms, and apply automated feature selection techniques.")

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    tabs = st.tabs([
        "📐 Math & Transforms",
        "✖️ Interaction Features",
        "📊 Binning & Discretization",
        "🗓️ Datetime Extraction",
        "🎯 Feature Selection Suite"
    ])

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # Tab 1: Mathematical Transforms
    with tabs[0]:
        st.subheader("Mathematical Feature Transformations")
        if numeric_columns:
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                target_col = st.selectbox("Select Feature", numeric_columns, key="trans_col")
            with tc2:
                trans_type = st.selectbox("Transformation Type", ["Log1p (log(1+x))", "Square Root (sqrt(x))", "Square (x^2)", "Reciprocal (1/(x+eps))", "Standard Z-Score"], key="trans_type")
            with tc3:
                new_col_name = st.text_input("New Column Name", value=f"{target_col}_transformed", key="trans_new_name")

            if st.button("Apply Transformation", type="primary", key="btn_trans"):
                out = df.copy()
                col_data = out[target_col].clip(lower=0)

                if "Log1p" in trans_type:
                    out[new_col_name] = np.log1p(col_data)
                elif "Square Root" in trans_type:
                    out[new_col_name] = np.sqrt(col_data)
                elif "Square" in trans_type:
                    out[new_col_name] = out[target_col] ** 2
                elif "Reciprocal" in trans_type:
                    out[new_col_name] = 1.0 / (out[target_col] + 1e-5)
                elif "Standard" in trans_type:
                    std = out[target_col].std()
                    out[new_col_name] = (out[target_col] - out[target_col].mean()) / (std if std != 0 else 1.0)

                set_dataset(out, action_description=f"Engineered feature '{new_col_name}' from '{target_col}'")
                st.success(f"New feature '{new_col_name}' created successfully.")
                st.rerun()
        else:
            st.info("No numeric columns available for transformation.")

    # Tab 2: Interaction Features
    with tabs[1]:
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

    # Tab 3: Binning & Discretization
    with tabs[2]:
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

    # Tab 4: Datetime Extraction
    with tabs[3]:
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

    # Tab 5: Feature Selection Suite
    with tabs[4]:
        st.subheader("🎯 Automated Feature Selection Suite")
        st.write("Score and select the most relevant features to optimize machine learning performance.")

        sel_tab1, sel_tab2, sel_tab3 = st.tabs(["📉 Collinearity & Variance Filter", "🏆 Statistical Ranking (SelectKBest)", "🌲 Tree Feature Importance"])

        # Collinearity & Variance
        with sel_tab1:
            st.markdown("#### Drop Low Variance & High Collinearity Features")
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

        # SelectKBest
        with sel_tab2:
            st.markdown("#### SelectKBest Scoring (ANOVA / Mutual Information)")
            all_cols = df.columns.tolist()
            target_feat = st.selectbox("Select Target Variable for Feature Selection", all_cols, index=len(all_cols)-1, key="kb_target")
            
            X_cols = [c for c in numeric_columns if c != target_feat]
            if X_cols and target_feat:
                clean_data = df.dropna(subset=[target_feat] + X_cols)
                if len(clean_data) > 10:
                    y_vec = clean_data[target_feat]
                    prob_type = detect_problem_type(y_vec)
                    
                    metric_func = mutual_info_classif if prob_type == "Classification" else mutual_info_regression
                    scores = metric_func(clean_data[X_cols], y_vec)
                    score_df = pd.DataFrame({"Feature": X_cols, "Score": scores}).sort_values("Score", ascending=True)

                    fig_score = px.bar(score_df, x="Score", y="Feature", orientation="h", title=f"Mutual Information Feature Scores ({prob_type})")
                    fig_score.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig_score, use_container_width=True)

                    k_val = st.slider("Select Top K Features to Keep", 1, len(X_cols), min(5, len(X_cols)), key="kb_slider")
                    top_k_feats = score_df.tail(k_val)["Feature"].tolist()

                    if st.button(f"Keep Only Top {k_val} Features (+ Target)", key="btn_keep_top_k"):
                        cols_to_keep = top_k_feats + [target_feat]
                        out = df[cols_to_keep].copy()
                        st.session_state.selected_features = cols_to_keep
                        set_dataset(out, action_description=f"Filtered dataset to top {k_val} features using SelectKBest")
                        st.success(f"Filtered dataset to top {k_val} features.")
                        st.rerun()

        # Tree Importance
        with sel_tab3:
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

    st.markdown("### 📋 Current Features Preview")
    st.dataframe(df.head(10), use_container_width=True)
