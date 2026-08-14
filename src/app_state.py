import streamlit as st

def init_state():
    defaults = {
        "dataset": None,
        "raw_dataset": None,
        "history": [],
        "redo_stack": [],
        "action_log": [],
        "analysis": {},
        "recommendations": [],
        "selected_features": None,
        "experiments": [],
        "model_bundle": None,
        "best_model_name": None,
        "target_column": None,
        "problem_type": None,
        "cluster_model": None,
        "sidebar_open": True,
        "nav_button_size": "Medium",
        "custom_logo_b64": None,
        "selected_page": "Home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def set_dataset(df, action_description=None, save_history=True):
    current = st.session_state.get("dataset")
    if save_history and current is not None:
        st.session_state.history.append(current.copy())
        st.session_state.redo_stack.clear()
        if action_description:
            st.session_state.action_log.append(action_description)
    
    st.session_state.dataset = df.copy()
    # Reset selected features if dataset columns changed significantly
    if st.session_state.get("selected_features") is not None:
        valid_cols = [c for c in st.session_state.selected_features if c in df.columns]
        st.session_state.selected_features = valid_cols if valid_cols else df.columns.tolist()

def get_dataset():
    return st.session_state.get("dataset")

def undo():
    if not st.session_state.history:
        return False
    st.session_state.redo_stack.append(st.session_state.dataset.copy())
    st.session_state.dataset = st.session_state.history.pop()
    if st.session_state.action_log:
        st.session_state.action_log.pop()
    return True

def redo():
    if not st.session_state.redo_stack:
        return False
    st.session_state.history.append(st.session_state.dataset.copy())
    st.session_state.dataset = st.session_state.redo_stack.pop()
    return True

def reset_dataset():
    raw = st.session_state.get("raw_dataset")
    if raw is not None:
        st.session_state.dataset = raw.copy()
        st.session_state.history.clear()
        st.session_state.redo_stack.clear()
        st.session_state.action_log.clear()
        st.session_state.selected_features = raw.columns.tolist()

def get_active_features(df=None):
    if df is None:
        df = get_dataset()
    if df is None:
        return []
    if st.session_state.get("selected_features") is not None:
        return [c for c in st.session_state.selected_features if c in df.columns]
    return df.columns.tolist()
