import streamlit as st
from src.app_state import get_dataset

def render():
    st.markdown(
        '''
        <div class="hero">
            <span class="badge">AUTONOMOUS DATA & ML PLATFORM</span>
            <h1>DataPilot AI</h1>
            <p>
                Upload raw datasets, perform automated health audits, execute one-click smart recommendations,
                engineer advanced features, train machine learning & clustering pipelines, and generate executive reports.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    df = get_dataset()
    if df is not None:
        st.subheader("🟢 Active Dataset Quick Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Columns", df.shape[1])
        k3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
        k4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
        st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("🚀 Complete End-to-End Workflow")

    columns = st.columns(4)
    cards = [
        ("📤", "1. Upload & Import", "Support CSV and Excel files with instant schema preview."),
        ("🔍", "2. Audit & Recommend", "Auto-detect data quality issues with 1-click action buttons."),
        ("🧹", "3. Preprocess & Clean", "Impute missing values, cap outliers, scale and encode features."),
        ("⚙️", "4. Feature Engineering", "Transform features, generate interactions, and select top K features.")
    ]

    for column, (icon, title, text) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="card"><div class="kpi-icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True
            )

    columns2 = st.columns(4)
    cards2 = [
        ("📊", "5. Visualization Studio", "19+ interactive Plotly charts with custom dark themes."),
        ("🤖", "6. ML & Clustering", "Benchmark classification, regression, and K-Means clustering."),
        ("🎯", "7. Prediction Studio", "Run single record forms or batch CSV predictions."),
        ("📄", "8. Executive Reports", "Export complete markdown and formatted HTML reports.")
    ]

    for column, (icon, title, text) in zip(columns2, cards2):
        with column:
            st.markdown(
                f'<div class="card"><div class="kpi-icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True
            )
