import streamlit as st
import pandas as pd
import numpy as np
from src.ui import page_header
from src.app_state import set_dataset

def render():
    page_header("📤 Upload Dataset", "Import CSV or Excel datasets or load built-in sample datasets to get started.")

    u1, u2 = st.columns([2, 1])

    with u1:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="upload_file_widget"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                if st.button("🚀 Load Uploaded Dataset", type="primary", use_container_width=True):
                    st.session_state.raw_dataset = df.copy()
                    st.session_state.selected_features = df.columns.tolist()
                    set_dataset(df, save_history=False, action_description="Loaded uploaded dataset")
                    st.success("Dataset loaded successfully! You can now proceed to Analysis & Preprocessing.")
                    st.rerun()

            except Exception as error:
                st.error(f"Unable to load file: {error}")

    with u2:
        st.markdown(
            '''
            <div class="card">
                <h3>🧪 Quick Sample Datasets</h3>
                <p>Don't have a dataset ready? Load a sample synthetic dataset instantly:</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        if st.button("📊 Load Sample Housing Dataset (Regression)", use_container_width=True, key="btn_sample_reg"):
            np.random.seed(42)
            n = 300
            sample_df = pd.DataFrame({
                "House_ID": [f"H_{i:04d}" for i in range(n)],
                "Square_Feet": np.random.randint(600, 4500, n),
                "Bedrooms": np.random.randint(1, 6, n),
                "Bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], n),
                "Age_Years": np.random.randint(0, 50, n),
                "Neighborhood": np.random.choice(["Downtown", "Suburbs", "Uptown", "Waterfront"], n),
                "Price": np.random.randint(120000, 850000, n)
            })
            # Inject a few missing values and duplicates for demo
            sample_df.loc[10:15, "Square_Feet"] = np.nan
            sample_df.loc[25:28, "Neighborhood"] = np.nan
            sample_df = pd.concat([sample_df, sample_df.iloc[:4]], ignore_index=True)

            st.session_state.raw_dataset = sample_df.copy()
            st.session_state.selected_features = sample_df.columns.tolist()
            set_dataset(sample_df, save_history=False, action_description="Loaded sample housing dataset")
            st.success("Sample housing dataset loaded!")
            st.rerun()

        if st.button("🎯 Load Sample Customer Churn Dataset (Classification)", use_container_width=True, key="btn_sample_clf"):
            np.random.seed(42)
            n = 300
            sample_df = pd.DataFrame({
                "Customer_ID": [f"C_{i:04d}" for i in range(n)],
                "Tenure_Months": np.random.randint(1, 72, n),
                "Monthly_Charges": np.random.uniform(20.0, 120.0, n).round(2),
                "Total_Charges": np.random.uniform(100.0, 8000.0, n).round(2),
                "Contract_Type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], n),
                "Payment_Method": np.random.choice(["Electronic Check", "Mailed Check", "Credit Card", "Bank Transfer"], n),
                "Churn": np.random.choice(["No", "Yes"], n, p=[0.75, 0.25])
            })
            sample_df.loc[5:9, "Total_Charges"] = np.nan
            sample_df = pd.concat([sample_df, sample_df.iloc[:3]], ignore_index=True)

            st.session_state.raw_dataset = sample_df.copy()
            st.session_state.selected_features = sample_df.columns.tolist()
            set_dataset(sample_df, save_history=False, action_description="Loaded sample churn dataset")
            st.success("Sample customer churn dataset loaded!")
            st.rerun()
