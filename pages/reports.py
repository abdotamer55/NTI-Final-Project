import streamlit as st
import pandas as pd
from src.ui import page_header, render_dataset_toolbar
from src.app_state import get_dataset, get_active_features

def render():
    page_header("📄 Executive Audit & Workflow Reports", "Generate, inspect, and export comprehensive HTML and Markdown reports for your project.")

    df = get_dataset()
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    render_dataset_toolbar()

    action_log = st.session_state.get("action_log", [])
    bundle = st.session_state.get("model_bundle")
    active_feats = get_active_features(df)

    # Build Markdown Report
    md_lines = [
        "# DataPilot AI Executive Workflow Report",
        f"**Dataset Shape:** {len(df):,} Rows × {df.shape[1]} Columns",
        f"**Active Features:** {len(active_feats)} features selected",
        f"**Total Missing Cells:** {int(df.isna().sum().sum()):,}",
        f"**Duplicate Rows:** {int(df.duplicated().sum()):,}",
        f"**Memory Footprint:** {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB",
        "\n---",
        "## 🛠️ Data Engineering Action Audit Log"
    ]

    if action_log:
        for idx, act in enumerate(action_log, start=1):
            md_lines.append(f"{idx}. {act}")
    else:
        md_lines.append("- No data transformations applied yet.")

    md_lines.append("\n---")
    md_lines.append("## 🤖 Machine Learning Model Summary")

    if bundle:
        md_lines.append(f"- **Best Model:** {st.session_state.best_model_name}")
        md_lines.append(f"- **Problem Type:** {bundle['problem_type']}")
        md_lines.append(f"- **Target Column:** {bundle['target']}")
        md_lines.append(f"- **Input Features Used:** {', '.join(bundle['features'])}")

        if "leaderboard" in bundle:
            md_lines.append("\n### Model Benchmark Leaderboard:")
            md_lines.append(bundle["leaderboard"].to_markdown(index=False))
    else:
        md_lines.append("- No machine learning model has been trained yet.")

    full_md_report = "\n".join(md_lines)

    # Build HTML Report
    html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DataPilot AI Report</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #080c17; color: #f8fafc; padding: 40px; line-height: 1.6; }}
h1 {{ color: #6d5dfc; font-size: 2.2rem; border-bottom: 2px solid #1e293b; padding-bottom: 10px; }}
h2 {{ color: #22d3ee; font-size: 1.4rem; margin-top: 30px; }}
.card {{ background: #141e36; border: 1px solid rgba(148,163,184,0.2); border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
th, td {{ border: 1px solid rgba(148,163,184,0.2); padding: 10px; text-align: left; }}
th {{ background: #1e293b; color: #6d5dfc; }}
</style>
</head>
<body>
<h1>🚀 DataPilot AI Executive Workflow Report</h1>
<div class="card">
    <h2>📊 Dataset Metrics Summary</h2>
    <p><b>Rows:</b> {len(df):,}<br><b>Columns:</b> {df.shape[1]}<br><b>Active Features:</b> {len(active_feats)}<br><b>Memory:</b> {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB</p>
</div>
<div class="card">
    <h2>🛠️ Action Audit Log</h2>
    <ul>
        {"".join([f"<li>{act}</li>" for act in action_log]) if action_log else "<li>No actions recorded.</li>"}
    </ul>
</div>
<div class="card">
    <h2>🤖 Machine Learning Summary</h2>
    <p><b>Best Model:</b> {st.session_state.get('best_model_name', 'N/A')}<br>
    <b>Target:</b> {st.session_state.get('target_column', 'N/A')}<br>
    <b>Problem Type:</b> {st.session_state.get('problem_type', 'N/A')}</p>
</div>
</body>
</html>"""

    st.subheader("📋 Executive Report Preview")
    st.markdown(full_md_report)

    st.markdown("---")
    st.subheader("⬇ Download Reports")
    r1, r2 = st.columns(2)
    with r1:
        st.download_button(
            "⬇ Download Markdown Report (.md)",
            data=full_md_report,
            file_name="smartprepml_report.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )
    with r2:
        st.download_button(
            "⬇ Download Executive HTML Report (.html)",
            data=html_report,
            file_name="smartprepml_report.html",
            mime="text/html",
            use_container_width=True
        )
