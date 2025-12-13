"""
Tab module for the Data Analysis Toolkit
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from statistical_analysis import StatisticalAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_statistical_tests_tab():
    """Render advanced statistical tests tab"""
    st.header("🧪 Statistical Hypothesis Tests")
    st.caption("t-tests, ANOVA, Chi-square, normality tests (Shapiro-Wilk), and correlation significance tests")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    stats = StatisticalAnalysis(df)

    st.subheader("Test Distributions & PDFs")

    col1, col2 = st.columns(2)

    st.markdown("---")

    st.subheader("Hypothesis Tests")

    test_type = st.selectbox(
        "Test Type",
        ["Compare 2 Groups", "Compare 3+ Groups", "Chi-Square", "Normality", "Correlation"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if test_type == "Compare 2 Groups":
            col1_test = st.selectbox("Column 1", features, key='col1_test')
            col2_test = st.selectbox("Column 2", [f for f in features if f != col1_test], key='col2_test')
            test_subtype = st.radio("Test", ["Independent t-test", "Paired t-test", "Mann-Whitney U"])

            if st.button("🧪 Run Test", use_container_width=True):
                try:
                    if test_subtype == "Independent t-test":
                        results = stats.ttest_independent(col1_test, col2_test)
                    elif test_subtype == "Paired t-test":
                        results = stats.ttest_paired(col1_test, col2_test)
                    else:
                        results = stats.mann_whitney_u(col1_test, col2_test)
                    st.session_state.analysis_results['hypothesis_test'] = results
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if 'distributions' in st.session_state.analysis_results:
        st.subheader("Distribution Fitting Results")
        dist_results = st.session_state.analysis_results['distributions']

        if 'distributions' in dist_results:
            for dist_name, dist_data in dist_results['distributions'].items():
                with st.expander(f"**{dist_name.upper()}**"):
                    col1, col2 = st.columns(2)
                    col1.metric("Parameters", str(dist_data.get('params', {}))[:50])
                    col2.metric("KS Statistic", f"{dist_data.get('ks_statistic', 0):.4f}")

    if 'hypothesis_test' in st.session_state.analysis_results:
        st.subheader("Test Results")
        test_results = st.session_state.analysis_results['hypothesis_test']

        col1, col2, col3 = st.columns(3)
        col1.metric("Statistic", f"{test_results.get('statistic', 0):.4f}")
        col2.metric("p-value", f"{test_results.get('p_value', 0):.4f}")

        if test_results.get('p_value', 1) < 0.05:
            col3.success("✅ Significant (p < 0.05)")
        else:
            col3.info("❌ Not Significant (p ≥ 0.05)")


