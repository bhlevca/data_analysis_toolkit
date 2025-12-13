"""
Advanced Data Analysis Toolkit - Streamlit Version with Plotly
===============================================================

A comprehensive data analysis application with integrated tutorial guidance.
Uses Plotly for interactive, zoomable charts.

Version: 9.2 (Modular Refactor)

This is the main entry point. Tab implementations are in the tabs/ directory.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Advanced Data Analysis Toolkit v9",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the package directory to path for direct running
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import all tab rendering functions from tabs module
from tabs import (
    render_data_tab,
    render_statistical_tab,
    render_statistical_tests_tab,
    render_ml_tab,
    render_neural_networks_tab,
    render_pca_tab,
    render_bayesian_tab,
    render_uncertainty_tab,
    render_nonlinear_tab,
    render_timeseries_tab,
    render_causality_tab,
    render_visualization_tab,
    render_clustering_tab,
    render_anomaly_tab,
    render_signal_analysis_tab,
    render_dimreduction_tab,
    render_tutorial_sidebar,
)

# Import for session state initialization
from rust_accelerated import is_rust_available


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
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


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point"""

    init_session_state()
    render_tutorial_sidebar()

    st.title("📊 Advanced Data Analysis Toolkit")
    st.caption("Version 9.2 - Modular Streamlit Edition with Interactive Plotly Charts")

    # =========================================================================
    # LEVEL 1: Main Category Tabs (6 groups)
    # =========================================================================
    main_tabs = st.tabs([
        "📁 Data",
        "📊 Statistics",
        "🔊 Signal Processing",
        "⏱️ Time Series",
        "🤖 Machine Learning",
        "📈 Visualization"
    ])

    # =========================================================================
    # 📁 DATA GROUP
    # =========================================================================
    with main_tabs[0]:
        render_data_tab()

    # =========================================================================
    # 📊 STATISTICS GROUP (4 subtabs)
    # =========================================================================
    with main_tabs[1]:
        st.markdown("#### 📊 Statistics Group")
        st.caption("Descriptive statistics, hypothesis testing, Bayesian inference, and uncertainty quantification")

        stats_subtabs = st.tabs([
            "📊 Descriptive Statistics",
            "🧪 Hypothesis Tests",
            "📈 Bayesian Inference",
            "🎲 Uncertainty Analysis"
        ])

        with stats_subtabs[0]:
            render_statistical_tab()
        with stats_subtabs[1]:
            render_statistical_tests_tab()
        with stats_subtabs[2]:
            render_bayesian_tab()
        with stats_subtabs[3]:
            render_uncertainty_tab()

    # =========================================================================
    # 🔊 SIGNAL PROCESSING GROUP
    # =========================================================================
    with main_tabs[2]:
        st.markdown("#### 🔊 Signal Processing Group")
        st.caption("Frequency analysis using FFT, PSD, and Wavelet transforms (CWT/DWT)")
        render_signal_analysis_tab()

    # =========================================================================
    # ⏱️ TIME SERIES GROUP (2 subtabs)
    # =========================================================================
    with main_tabs[3]:
        st.markdown("#### ⏱️ Time Series Group")
        st.caption("Temporal pattern analysis, stationarity testing, and causal relationships")

        ts_subtabs = st.tabs([
            "⏱️ Time Series Analysis",
            "🔗 Causality (Granger)"
        ])

        with ts_subtabs[0]:
            render_timeseries_tab()
        with ts_subtabs[1]:
            render_causality_tab()

    # =========================================================================
    # 🤖 MACHINE LEARNING GROUP (7 subtabs)
    # =========================================================================
    with main_tabs[4]:
        st.markdown("#### 🤖 Machine Learning Group")
        st.caption("Supervised learning, neural networks, dimensionality reduction, clustering, and anomaly detection")

        ml_subtabs = st.tabs([
            "🤖 Regression/Classification",
            "🧠 Neural Networks",
            "🔬 PCA (Principal Components)",
            "🎯 Clustering",
            "🚨 Anomaly Detection",
            "📉 Dimensionality Reduction",
            "🔀 Non-Linear Analysis"
        ])

        with ml_subtabs[0]:
            render_ml_tab()
        with ml_subtabs[1]:
            render_neural_networks_tab()
        with ml_subtabs[2]:
            render_pca_tab()
        with ml_subtabs[3]:
            render_clustering_tab()
        with ml_subtabs[4]:
            render_anomaly_tab()
        with ml_subtabs[5]:
            render_dimreduction_tab()
        with ml_subtabs[6]:
            render_nonlinear_tab()

    # =========================================================================
    # 📈 VISUALIZATION GROUP
    # =========================================================================
    with main_tabs[5]:
        st.markdown("#### 📈 Visualization Group")
        st.caption("Interactive charts, scatter plots, distributions, and regression visualization")
        render_visualization_tab()

    st.markdown("---")
    st.caption("💡 All charts are **interactive**: zoom, pan, hover, download!")


if __name__ == "__main__":
    main()
