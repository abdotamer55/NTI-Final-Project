import streamlit as st
import numpy as np
import pandas as pd
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, set_dataset
from utils.preprocessing_utils import (
    fill_missing_recommended,
    impute_column,
    remove_duplicates,
    drop_constant_columns,
    drop_columns,
    cap_outliers_iqr,
    cap_outliers_zscore,
    encode_categorical,
    scale_features,
    cast_column_type,
    recommend_features,
    create_datetime_features,
    create_interaction_features,
    create_binned_features,
    create_log_transform,
    create_group_aggregate_features,
    create_text_stats_features,
    create_frequency_encoding,
    auto_generate_recommended_features,
)

# Any column (numeric or text) with at most this many unique values is
# treated as categorical / encodable. Keeps high-cardinality ID-like text
# columns out of the encoding tab by default.
CATEGORICAL_CARDINALITY_LIMIT = 50


def render():
    page_header("🧹 Preprocessing Studio", "Clean, impute, transform, scale and encode features with full undo/redo history.")

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    tabs = st.tabs([
        "⚡ Quick Auto-Clean",
        "🩹 Missing Values",
        "✂️ Outliers Handling",
        "🏷️ Encoding",
        "📏 Feature Scaling",
        "🔧 Column Operations",
        "🪄 Feature Engineering",
    ])

    # ------------------------------------------------------------------ #
    # Tab 1: Auto Clean
    # ------------------------------------------------------------------ #
    with tabs[0]:
        st.subheader("Automated Data Cleaning")
        st.write("Apply recommended pipeline steps automatically to your dataset.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✨ Apply All Recommendations", type="primary", use_container_width=True):
                try:
                    cleaned = fill_missing_recommended(df)
                    cleaned = remove_duplicates(cleaned)
                    cleaned, dropped = drop_constant_columns(cleaned)
                    set_dataset(cleaned, action_description="Applied Auto-Cleaning pipeline")
                    st.success("Automated cleaning applied successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Auto-cleaning failed: {e}")

        with c2:
            if st.button("🧹 Remove Duplicate Rows", use_container_width=True):
                cnt_before = len(df)
                cleaned = remove_duplicates(df)
                diff = cnt_before - len(cleaned)
                set_dataset(cleaned, action_description=f"Removed {diff} duplicate rows")
                st.success(f"Removed {diff} duplicate rows.")
                st.rerun()

        with c3:
            if st.button("❌ Drop Zero-Variance Columns", use_container_width=True):
                cleaned, dropped = drop_constant_columns(df)
                if dropped:
                    set_dataset(cleaned, action_description=f"Dropped constant columns: {', '.join(dropped)}")
                    st.success(f"Dropped columns: {', '.join(dropped)}")
                    st.rerun()
                else:
                    st.info("No constant columns detected.")

    # ------------------------------------------------------------------ #
    # Tab 2: Missing Values
    # ------------------------------------------------------------------ #
    with tabs[1]:
        st.subheader("Missing Value Imputation")
        missing_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().any()}

        if missing_cols:
            st.write("Columns with missing values:")
            m_df = pd.DataFrame([
                {"Column": col, "Missing Count": cnt, "Missing %": f"{(cnt / len(df)) * 100:.1f}%", "Dtype": str(df[col].dtype)}
                for col, cnt in missing_cols.items()
            ])
            st.dataframe(m_df, use_container_width=True)

            col1, col2, col3 = st.columns([1.5, 1.5, 1])
            with col1:
                target_col = st.selectbox("Select Target Column", list(missing_cols.keys()), key="imp_col")
            with col2:
                is_num = pd.api.types.is_numeric_dtype(df[target_col])
                options = ["Median", "Mean", "Mode", "Constant Fill", "Drop Rows with NA", "Drop Column"] if is_num else ["Mode", "Constant Fill", "Drop Rows with NA", "Drop Column"]
                strategy_sel = st.selectbox("Imputation Strategy", options, key="imp_strat")
            with col3:
                const_val = ""
                if strategy_sel == "Constant Fill":
                    const_val = st.text_input("Fill Value", value="Unknown", key="imp_const")

            if st.button("Apply Imputation", type="primary", key="btn_apply_imp"):
                strat_map = {
                    "Median": "median", "Mean": "mean", "Mode": "mode",
                    "Constant Fill": "constant", "Drop Rows with NA": "drop_rows",
                }
                try:
                    if strategy_sel == "Drop Column":
                        out = drop_columns(df, [target_col])
                        desc = f"Dropped column '{target_col}'"
                    else:
                        rows_before = len(df)
                        out = impute_column(df, target_col, strategy=strat_map[strategy_sel], fill_value=const_val)
                        if strategy_sel == "Drop Rows with NA":
                            desc = f"Dropped {rows_before - len(out)} rows with missing '{target_col}'"
                        else:
                            desc = f"Imputed '{target_col}' using {strategy_sel}"

                    set_dataset(out, action_description=desc)
                    st.success(desc + ".")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            st.success("🎉 No missing values detected in the current dataset.")

    # ------------------------------------------------------------------ #
    # Tab 3: Outliers Handling
    # ------------------------------------------------------------------ #
    with tabs[2]:
        st.subheader("Outlier Capping & Truncation")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            oc1, oc2 = st.columns([2, 1.5])
            with oc1:
                outlier_cols = st.multiselect("Select Numeric Column(s)", numeric_cols, key="out_cols")
            with oc2:
                out_method = st.selectbox("Outlier Method", ["IQR Capping (1.5x IQR)", "Z-Score Truncation (3.0 std)"], key="out_method")

            if st.button("Apply Outlier Handling", type="primary", key="btn_apply_out"):
                if not outlier_cols:
                    st.warning("Select at least one numeric column.")
                else:
                    out = df
                    applied, skipped = [], []
                    for col in outlier_cols:
                        before = out[col].copy()
                        if "IQR" in out_method:
                            out = cap_outliers_iqr(out, col, factor=1.5)
                        else:
                            out = cap_outliers_zscore(out, col, threshold=3.0)
                        # A column is "skipped" internally (too few values / zero
                        # variance) if nothing changed.
                        if out[col].equals(before):
                            skipped.append(col)
                        else:
                            applied.append(col)

                    method_label = "1.5x IQR" if "IQR" in out_method else "Z-score (3.0 std)"
                    if applied:
                        set_dataset(out, action_description=f"Capped outliers in {applied} using {method_label}")
                        st.success(f"Outlier treatment applied to: {', '.join(applied)}.")
                    if skipped:
                        st.warning(f"Skipped (not enough data or zero variance): {', '.join(skipped)}.")
                    if applied:
                        st.rerun()
        else:
            st.info("No numeric columns available for outlier treatment.")

    # ------------------------------------------------------------------ #
    # Tab 4: Categorical Encoding
    # ------------------------------------------------------------------ #
    with tabs[3]:
        st.subheader("Categorical Encoding")
        cat_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= CATEGORICAL_CARDINALITY_LIMIT]

        if cat_cols:
            ec1, ec2 = st.columns(2)
            with ec1:
                selected_enc_cols = st.multiselect("Select Categorical Columns to Encode", cat_cols, default=cat_cols[:min(3, len(cat_cols))], key="enc_multisel")
            with ec2:
                enc_method = st.selectbox("Encoding Method", ["One-Hot Encoding", "Ordinal / Label Encoding", "Frequency Encoding"], key="enc_method_sel")

            if st.button("Apply Encoding", type="primary", key="btn_apply_enc"):
                if not selected_enc_cols:
                    st.warning("Please select at least one column to encode.")
                else:
                    method_key = "onehot" if "One-Hot" in enc_method else ("label" if "Label" in enc_method else "frequency")

                    # Warn upfront about columns one-hot will skip due to high cardinality.
                    if method_key == "onehot":
                        oversized = [c for c in selected_enc_cols if df[c].nunique() > 15]
                        if oversized:
                            st.warning(
                                f"Skipping {oversized} for One-Hot: more than 15 unique values "
                                "would explode column count. Try Label or Frequency encoding for these."
                            )

                    # Warn upfront about NaN becoming -1 with label encoding.
                    if method_key == "label":
                        na_cols = [c for c in selected_enc_cols if df[c].isna().any()]
                        if na_cols:
                            st.warning(
                                f"Columns {na_cols} contain missing values, which Label Encoding "
                                "will convert to -1. Handle missing values first if that's not intended."
                            )

                    try:
                        out = encode_categorical(df, selected_enc_cols, method=method_key)
                        set_dataset(out, action_description=f"Encoded {selected_enc_cols} using {enc_method}")
                        st.success("Categorical encoding completed successfully.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        else:
            st.info("No categorical columns available for encoding.")

    # ------------------------------------------------------------------ #
    # Tab 5: Scaling & Normalization
    # ------------------------------------------------------------------ #
    with tabs[4]:
        st.subheader("Feature Scaling & Normalization")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            sc1, sc2 = st.columns(2)
            with sc1:
                scale_cols = st.multiselect("Select Numeric Features to Scale", num_cols, default=num_cols, key="scale_cols_sel")
            with sc2:
                scale_method = st.selectbox("Scaler Type", ["StandardScaler (Mean=0, Std=1)", "MinMaxScaler (Range 0-1)", "RobustScaler (IQR based)"], key="scale_method_sel")

            st.caption("Columns with missing values must be handled first (Missing Values tab) — scaling will refuse to run otherwise.")

            if st.button("Apply Feature Scaling", type="primary", key="btn_apply_scale"):
                if not scale_cols:
                    st.warning("Select at least one feature to scale.")
                else:
                    m_key = "standard" if "Standard" in scale_method else ("minmax" if "MinMax" in scale_method else "robust")
                    try:
                        out = scale_features(df, scale_cols, method=m_key)
                        set_dataset(out, action_description=f"Scaled features using {scale_method}")
                        st.success("Feature scaling applied.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        else:
            st.info("No numeric features available for scaling.")

    # ------------------------------------------------------------------ #
    # Tab 6: Column Operations
    # ------------------------------------------------------------------ #
    with tabs[5]:
        st.subheader("Column Management & Type Conversion")
        col_op1, col_op2 = st.columns(2)

        with col_op1:
            st.markdown("#### Drop Columns")
            drop_targets = st.multiselect("Select Columns to Drop", df.columns.tolist(), key="drop_cols_sel")
            if st.button("🗑️ Drop Selected Columns", key="btn_drop_cols"):
                if drop_targets:
                    out = drop_columns(df, drop_targets)
                    set_dataset(out, action_description=f"Dropped columns: {', '.join(drop_targets)}")
                    st.success(f"Dropped {len(drop_targets)} columns.")
                    st.rerun()
                else:
                    st.warning("Select at least one column to drop.")

        with col_op2:
            st.markdown("#### Convert Column Data Type")
            target_cast_col = st.selectbox("Column to Cast", df.columns.tolist(), key="cast_col_sel")
            new_type_sel = st.selectbox("New Type", ["numeric", "datetime", "category", "string"], key="cast_type_sel")
            if st.button("Cast Column Type", key="btn_cast_col"):
                try:
                    na_before = df[target_cast_col].isna().sum()
                    out = cast_column_type(df, target_cast_col, new_type_sel)
                    na_after = out[target_cast_col].isna().sum()
                    new_na = int(na_after - na_before)

                    set_dataset(out, action_description=f"Cast '{target_cast_col}' to {new_type_sel}")
                    st.success(f"Cast '{target_cast_col}' to {new_type_sel}.")
                    if new_na > 0:
                        st.warning(f"{new_na} value(s) could not be converted and became missing (NaN).")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # ------------------------------------------------------------------ #
    # Tab 7: Feature Engineering & Recommendations
    # ------------------------------------------------------------------ #
    with tabs[6]:
        st.subheader("Automated & Custom Feature Engineering")
        st.write("Intelligent dataset analysis recommends new high-value features for any uploaded dataset.")

        recs = recommend_features(df)
        if recs:
            st.markdown("### 💡 Recommended Feature Engineering Steps")

            c_auto1, c_auto2 = st.columns([2, 1])
            with c_auto1:
                st.info(f"Found {len(recs)} potential feature engineering recommendations for your dataset.")
            with c_auto2:
                if st.button("✨ Apply All Recommendations", type="primary", key="btn_apply_all_fe"):
                    try:
                        out, applied = auto_generate_recommended_features(df, max_features=10)
                        if applied:
                            set_dataset(out, action_description=f"Auto-generated {len(applied)} features")
                            st.success(f"Generated {len(applied)} features successfully!")
                            st.rerun()
                        else:
                            st.warning("No recommendations could be applied.")
                    except Exception as e:
                        st.error(f"Failed to generate features: {e}")

            for i, rec in enumerate(recs):
                badge = "🔴 High" if rec['priority'] == 'High' else ("🟡 Medium" if rec['priority'] == 'Medium' else "🟢 Low")
                with st.expander(f"{badge} | {rec['title']}", expanded=(i < 3)):
                    st.write(rec['description'])
                    if st.button(f"Apply This Recommendation", key=f"btn_apply_rec_{i}"):
                        action = rec.get("action")
                        params = rec.get("params", {})
                        try:
                            func_map = {
                                "create_datetime_features": create_datetime_features,
                                "create_log_transform": create_log_transform,
                                "create_interaction_features": create_interaction_features,
                                "create_group_aggregate_features": create_group_aggregate_features,
                                "create_binned_features": create_binned_features,
                                "create_text_stats_features": create_text_stats_features,
                                "create_frequency_encoding": create_frequency_encoding,
                            }
                            if action in func_map:
                                out = func_map[action](df, **params)
                                set_dataset(out, action_description=rec['title'])
                                st.success(f"Applied: {rec['title']}")
                                st.rerun()
                        except Exception as e:
                            st.error(str(e))
        else:
            st.success("No automated feature recommendations required for the current dataset structure.")

        st.markdown("---")
        st.markdown("### 🛠️ Custom Feature Generator")

        fe_type = st.selectbox(
            "Select Feature Creation Type",
            ["Datetime Component Extraction", "Mathematical Interaction (Product/Ratio)", "Skewness Transformation (Log1p/Sqrt)", "Group Aggregation", "Binning / Discretization", "Text Statistics"],
            key="fe_custom_type"
        )

        if fe_type == "Datetime Component Extraction":
            cols_avail = df.columns.tolist()
            col_sel = st.selectbox("Select Date/Time Column", cols_avail, key="fe_dt_col")
            if st.button("Extract Datetime Features", key="btn_dt_extract"):
                try:
                    out = create_datetime_features(df, col_sel)
                    set_dataset(out, action_description=f"Extracted datetime features from '{col_sel}'")
                    st.success(f"Extracted datetime features from '{col_sel}'.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        elif fe_type == "Mathematical Interaction (Product/Ratio)":
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    c1 = st.selectbox("First Column", num_cols, key="fe_int_c1")
                with ic2:
                    c2 = st.selectbox("Second Column", [c for c in num_cols if c != c1], key="fe_int_c2")
                with ic3:
                    op = st.selectbox("Operation", ["multiply", "divide", "add", "subtract"], key="fe_int_op")

                if st.button("Create Interaction Feature", key="btn_fe_int"):
                    try:
                        out = create_interaction_features(df, c1, c2, operation=op)
                        set_dataset(out, action_description=f"Created {op} interaction: '{c1}' & '{c2}'")
                        st.success("Interaction feature created.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            else:
                st.info("At least two numerical columns are required for interaction features.")

        elif fe_type == "Skewness Transformation (Log1p/Sqrt)":
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                sc1, sc2 = st.columns(2)
                with sc1:
                    col_sel = st.selectbox("Select Numeric Column", num_cols, key="fe_log_col")
                with sc2:
                    method_sel = st.selectbox("Transformation Method", ["log1p", "sqrt"], key="fe_log_method")

                if st.button("Apply Transformation", key="btn_fe_log"):
                    try:
                        out = create_log_transform(df, col_sel, method=method_sel)
                        set_dataset(out, action_description=f"Applied {method_sel} transform on '{col_sel}'")
                        st.success(f"Applied {method_sel} transformation.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        elif fe_type == "Group Aggregation":
            cat_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if cat_cols and num_cols:
                gc1, gc2 = st.columns(2)
                with gc1:
                    grp_col = st.selectbox("Group By Column (Categorical/Key)", cat_cols, key="fe_grp_cat")
                with gc2:
                    tgt_col = st.selectbox("Target Column to Aggregate", num_cols, key="fe_grp_num")

                if st.button("Create Group Aggregates", key="btn_fe_grp"):
                    try:
                        out = create_group_aggregate_features(df, grp_col, tgt_col)
                        set_dataset(out, action_description=f"Created group aggregates of '{tgt_col}' by '{grp_col}'")
                        st.success("Group aggregate features created.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        elif fe_type == "Binning / Discretization":
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    col_sel = st.selectbox("Column to Bin", num_cols, key="fe_bin_col")
                with bc2:
                    n_bins = st.number_input("Number of Bins", min_value=2, max_value=20, value=4, key="fe_bin_num")
                with bc3:
                    strat = st.selectbox("Binning Strategy", ["quantile", "uniform"], key="fe_bin_strat")

                if st.button("Create Binned Feature", key="btn_fe_bin"):
                    try:
                        out = create_binned_features(df, col_sel, num_bins=n_bins, strategy=strat)
                        set_dataset(out, action_description=f"Created {strat} binning on '{col_sel}'")
                        st.success("Binned feature created.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        elif fe_type == "Text Statistics":
            str_cols = df.columns.tolist()
            col_sel = st.selectbox("Select Text Column", str_cols, key="fe_txt_col")
            if st.button("Extract Text Statistics", key="btn_fe_txt"):
                try:
                    out = create_text_stats_features(df, col_sel)
                    set_dataset(out, action_description=f"Extracted text statistics for '{col_sel}'")
                    st.success("Text length and word count features created.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    st.markdown("### 📋 Current Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

    st.download_button(
        "⬇️ Download Current Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="preprocessed_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    render()

