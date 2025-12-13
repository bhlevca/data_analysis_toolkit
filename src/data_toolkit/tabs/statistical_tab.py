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

def render_statistical_tab():
    """Render statistical analysis tab"""
    st.header("📊 Descriptive Statistics & Correlation Analysis")
    st.caption("Compute summary statistics, correlation matrices, and detect outliers")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first in the Data Loading tab.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns in the Data Loading tab.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    # Initialize analyzer with dataframe
    stats = StatisticalAnalysis(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Descriptive Statistics", use_container_width=True):
            st.session_state.analysis_results['descriptive'] = stats.descriptive_stats(features)

    with col2:
        corr_method = st.selectbox("Correlation Method", ['pearson', 'spearman', 'kendall'])
        if st.button("🔗 Correlation Matrix", use_container_width=True):
            st.session_state.analysis_results['correlation'] = stats.correlation_matrix(features, method=corr_method)

    with col3:
        outlier_method = st.selectbox("Outlier Method", ['iqr', 'zscore'])
        if st.button("🎯 Outlier Detection", use_container_width=True):
            # Correct method name: outlier_detection
            st.session_state.analysis_results['outliers'] = stats.outlier_detection(features, method=outlier_method)

    st.markdown("---")

    # Display results
    if 'descriptive' in st.session_state.analysis_results:
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(st.session_state.analysis_results['descriptive'], use_container_width=True)

    if 'correlation' in st.session_state.analysis_results:
        st.subheader("🔗 Correlation Matrix")
        corr_data = st.session_state.analysis_results['correlation']

        fig = px.imshow(
            corr_data,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Heatmap'
        )
        fig.update_layout(height=600, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    if 'outliers' in st.session_state.analysis_results:
        st.subheader("🎯 Outlier Detection Results")
        outlier_data = st.session_state.analysis_results['outliers']

        # Box plots
        # Use unique column names to avoid conflicts with existing DataFrame columns
        box_data = df[features].melt(var_name='_Feature_', value_name='_Value_')
        fig = px.box(box_data, x='_Feature_', y='_Value_', title='Box Plots with Outliers',
                    template=PLOTLY_TEMPLATE, points='outliers')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        for col, info in outlier_data.items():
            n_outliers = info.get('n_outliers', 0)
            pct = info.get('percentage', 0)
            with st.expander(f"**{col}**: {n_outliers} outliers ({pct:.1f}%)"):
                if 'lower_bound' in info:
                    st.write(f"Lower bound: {info['lower_bound']:.4f}")
                if 'upper_bound' in info:
                    st.write(f"Upper bound: {info['upper_bound']:.4f}")


