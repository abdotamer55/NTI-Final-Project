from pathlib import Path
import streamlit as st
from src.app_state import undo, redo, reset_dataset, get_dataset

NAVIGATION = [
    ("🏠", "Home"),
    ("📤", "Upload Data"),
    ("🔍", "Analysis"),
    ("🧹", "Preprocessing"),
    ("📊", "Visualization"),
    ("⚙️", "Feature Engineering"),
    ("🤖", "Machine Learning"),
    ("🧪", "Experiments"),
    ("🎯", "Prediction"),
    ("🧩", "Unsupervised Prediction"),
    ("📄", "Reports"),
    ("👥", "Our Team"),
]

def load_css():
    css_path = Path("styles/main.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def render_dataset_toolbar():
    df = get_dataset()
    if df is None:
        return

    hist_len = len(st.session_state.get("history", []))
    redo_len = len(st.session_state.get("redo_stack", []))
    active_features = len(st.session_state.get("selected_features", df.columns.tolist()))

    t1, t2, t3, t4, t5 = st.columns([1.2, 1.2, 1.2, 2.5, 1.5])

    with t1:
        if st.button(f"↩ Undo ({hist_len})", disabled=(hist_len == 0), use_container_width=True, key="tb_undo"):
            if undo():
                st.rerun()
    with t2:
        if st.button(f"↪ Redo ({redo_len})", disabled=(redo_len == 0), use_container_width=True, key="tb_redo"):
            if redo():
                st.rerun()
    with t3:
        if st.button("⟲ Reset Data", use_container_width=True, key="tb_reset"):
            reset_dataset()
            st.rerun()
    with t4:
        st.markdown(
            f'<div style="padding:0.4rem 0.8rem; background:rgba(109,93,252,0.12); border:1px solid rgba(109,93,252,0.3); border-radius:10px; font-size:0.83rem; text-align:center;">'
            f'<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols | <b>Active Features:</b> {active_features}'
            f'</div>',
            unsafe_allow_html=True
        )
    with t5:
        st.download_button(
            "⬇ CSV Snapshot",
            data=df.to_csv(index=False).encode(),
            file_name="dataset_snapshot.csv",
            mime="text/csv",
            use_container_width=True,
            key="tb_download"
        )
    st.markdown("<hr style='margin: 0.8rem 0 1.5rem 0;'>", unsafe_allow_html=True)

import base64

def get_logo_html():
    custom_b64 = st.session_state.get("custom_logo_b64")
    if custom_b64:
        return f'<img src="data:image/png;base64,{custom_b64}" class="brand-logo-full-img" alt="DataPilot AI Logo" />'
    
    logo_path = Path("assets/images/logo.png")
    if logo_path.exists():
        try:
            encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            return f'<img src="data:image/png;base64,{encoded}" class="brand-logo-full-img" alt="DataPilot AI Logo" />'
        except Exception:
            pass
            
    return '<div class="brand-logo-full-img" style="font-size:3rem; text-align:center; padding: 0.5rem;">🚀</div>'

def render_navigation():
    sidebar_open = st.session_state.get("sidebar_open", True)
    nav_button_size = st.session_state.get("nav_button_size", "Medium")
    current_page = st.session_state.get("selected_page", "Home")

    if sidebar_open:
        nav_col, content_col = st.columns([1.25, 5.5], gap="large")
        with nav_col:
            # Header with collapse button
            h_col1, h_col2 = st.columns([3.5, 1])
            with h_col2:
                if st.button("◀", help="Close Sidebar", key="btn_close_sidebar", use_container_width=True):
                    st.session_state.sidebar_open = False
                    st.rerun()

            # Brand Header with Full-Width Top Logo
            logo_html = get_logo_html()
            st.markdown(
                f'''
                <div class="brand-card">
                    <div class="brand-logo-full-container">
                        {logo_html}
                    </div>
                    <div class="brand-info">
                        <h2>DataPilot AI</h2>
                        <p>Autonomous Data & ML Copilot</p>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            # Sidebar & Logo Settings Expander
            with st.expander("⚙️ Sidebar & Logo Settings", expanded=False):
                size_choice = st.select_slider(
                    "Button Size",
                    options=["Compact", "Medium", "Large"],
                    value=nav_button_size,
                    key="slider_nav_size"
                )
                if size_choice != nav_button_size:
                    st.session_state.nav_button_size = size_choice
                    st.rerun()

                uploaded_logo = st.file_uploader(
                    "Upload Custom Logo",
                    type=["png", "jpg", "jpeg", "svg"],
                    key="uploader_custom_logo"
                )
                if uploaded_logo is not None:
                    logo_bytes = uploaded_logo.read()
                    st.session_state.custom_logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
                    st.toast("Logo updated! 🎨", icon="✅")
                    st.rerun()

                if st.session_state.get("custom_logo_b64"):
                    if st.button("Reset Default Logo", key="btn_reset_logo", use_container_width=True):
                        st.session_state.custom_logo_b64 = None
                        st.toast("Restored default DataPilot AI logo.", icon="🔄")
                        st.rerun()

            st.markdown('<div class="nav-label">WORKFLOW STEPS</div>', unsafe_allow_html=True)
            
            # Styled Page Buttons (No radio points!)
            st.markdown(f'<div class="nav-size-{nav_button_size}">', unsafe_allow_html=True)
            for icon, name in NAVIGATION:
                is_active = (current_page == name)
                button_type = "primary" if is_active else "secondary"
                label = f"{icon}  {name}"
                if st.button(label, key=f"nav_btn_{name}", type=button_type, use_container_width=True):
                    if st.session_state.selected_page != name:
                        st.session_state.selected_page = name
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # Dataset status indicator
            df = st.session_state.get("dataset")
            if df is not None:
                mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                missing_total = int(df.isna().sum().sum())
                st.markdown(
                    f'''
                    <div class="nav-status">
                        <div class="nav-status-badge">🟢 Dataset Active</div>
                        <b>Rows:</b> {len(df):,}<br>
                        <b>Cols:</b> {df.shape[1]}<br>
                        <b>Missing:</b> {missing_total:,}<br>
                        <b>Memory:</b> {mem_mb:.2f} MB
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '''
                    <div class="nav-status">
                        <div class="nav-status-badge" style="color:var(--text-muted)">⚪ No Dataset</div>
                        Upload a CSV or Excel file to unlock analysis & modeling features.
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        return content_col
    else:
        # Collapsed mode: Top Bar trigger button to re-open sidebar
        top_bar, content_col = st.columns([1.2, 5.5], gap="large")
        with top_bar:
            if st.button("▶ Open Sidebar", key="btn_open_sidebar", help="Expand Navigation Sidebar", use_container_width=True):
                st.session_state.sidebar_open = True
                st.rerun()
        return content_col

