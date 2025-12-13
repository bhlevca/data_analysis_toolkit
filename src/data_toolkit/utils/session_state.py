"""
Session state initialization for the Data Analysis Toolkit
"""

import streamlit as st
from rust_accelerated import is_rust_available


def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = []
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True
    if 'current_tutorial' not in st.session_state:
        st.session_state.current_tutorial = "getting_started"
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'use_rust' not in st.session_state:
        st.session_state.use_rust = is_rust_available()
