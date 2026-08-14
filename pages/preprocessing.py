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
    cast_column_type
)

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
        "🔧 Column Operations"
    ])

    # Tab 1: Auto Clean
    with tabs[0]:
        st.subheader("Automated Data Cleaning")
        st.write("Apply recommended pipeline steps automatically to your dataset.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✨ Apply All Recommendations", type="primary", use_container_width=True):
                cleaned = fill_missing_recommended(df)
                cleaned = remove_duplicates(cleaned)
                cleaned, dropped = drop_constant_columns(cleaned)
                set_dataset(cleaned, action_description="Applied Auto-Cleaning pipeline")
                st.success("Automated cleaning applied successfully.")
                st.rerun()

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

    # Tab 2: Missing Values
    with tabs[1]:
        st.subheader("Missing Value Imputation")
        missing_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().any()}
        
        if missing_cols:
            st.write("Columns with missing values:")
            m_df = pd.DataFrame([
                {"Column": col, "Missing Count": cnt, "Missing %": f"{(cnt/len(df))*100:.1f}%", "Dtype": str(df[col].dtype)}
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
                    "Constant Fill": "constant", "Drop Rows with NA": "drop_rows"
                }
                if strategy_sel == "Drop Column":
                    out = drop_columns(df, [target_col])
                    desc = f"Dropped column '{target_col}'"
                else:
                    out = impute_column(df, target_col, strategy=strat_map[strategy_sel], fill_value=const_val)
                    desc = f"Imputed '{target_col}' using {strategy_sel}"
                
                set_dataset(out, action_description=desc)
                st.success(f"Imputation applied to '{target_col}'.")
                st.rerun()
        else:
            st.success("🎉 No missing values detected in the current dataset.")

    # Tab 3: Outliers Handling
    with tabs[2]:
        st.subheader("Outlier Capping & Truncation")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            oc1, oc2, oc3 = st.columns([1.5, 1.5, 1])
            with oc1:
                outlier_col = st.selectbox("Select Numeric Column", numeric_cols, key="out_col")
            with oc2:
                out_method = st.selectbox("Outlier Method", ["IQR Capping (1.5x IQR)", "Z-Score Truncation (3.0 std)"], key="out_method")
            with oc3:
                st.write("") # Alignment
                st.write("")

            if st.button("Apply Outlier Handling", type="primary", key="btn_apply_out"):
                if "IQR" in out_method:
                    out = cap_outliers_iqr(df, outlier_col, factor=1.5)
                    desc = f"Capped outliers in '{outlier_col}' using 1.5x IQR"
                else:
                    out = cap_outliers_zscore(df, outlier_col, threshold=3.0)
                    desc = f"Truncated outliers in '{outlier_col}' using Z-score"

                set_dataset(out, action_description=desc)
                st.success(f"Outlier treatment applied to '{outlier_col}'.")
                st.rerun()
        else:
            st.info("No numeric columns available for outlier treatment.")

    # Tab 4: Categorical Encoding
    with tabs[3]:
        st.subheader("Categorical Encoding")
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10]

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
                    out = encode_categorical(df, selected_enc_cols, method=method_key)
                    set_dataset(out, action_description=f"Encoded {selected_enc_cols} using {enc_method}")
                    st.success("Categorical encoding completed successfully.")
                    st.rerun()
        else:
            st.info("No categorical columns available for encoding.")

    # Tab 5: Scaling & Normalization
    with tabs[4]:
        st.subheader("Feature Scaling & Normalization")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            sc1, sc2 = st.columns(2)
            with sc1:
                scale_cols = st.multiselect("Select Numeric Features to Scale", num_cols, default=num_cols, key="scale_cols_sel")
            with sc2:
                scale_method = st.selectbox("Scaler Type", ["StandardScaler (Mean=0, Std=1)", "MinMaxScaler (Range 0-1)", "RobustScaler (IQR based)"], key="scale_method_sel")

            if st.button("Apply Feature Scaling", type="primary", key="btn_apply_scale"):
                if not scale_cols:
                    st.warning("Select at least one feature to scale.")
                else:
                    m_key = "standard" if "Standard" in scale_method else ("minmax" if "MinMax" in scale_method else "robust")
                    out = scale_features(df, scale_cols, method=m_key)
                    set_dataset(out, action_description=f"Scaled features using {scale_method}")
                    st.success("Feature scaling applied.")
                    st.rerun()
        else:
            st.info("No numeric features available for scaling.")

    # Tab 6: Column Operations
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

        with col_op2:
            st.markdown("#### Convert Column Data Type")
            target_cast_col = st.selectbox("Column to Cast", df.columns.tolist(), key="cast_col_sel")
            new_type_sel = st.selectbox("New Type", ["numeric", "datetime", "category", "string"], key="cast_type_sel")
            if st.button("Cast Column Type", key="btn_cast_col"):
                out = cast_column_type(df, target_cast_col, new_type_sel)
                set_dataset(out, action_description=f"Cast '{target_cast_col}' to {new_type_sel}")
                st.success(f"Cast '{target_cast_col}' to {new_type_sel}.")
                st.rerun()

    st.markdown("### 📋 Current Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)
