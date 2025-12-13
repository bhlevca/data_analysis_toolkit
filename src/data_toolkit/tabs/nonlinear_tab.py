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

from nonlinear_analysis import NonLinearAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_nonlinear_tab():
    """Render non-linear analysis tab"""
    st.header("🔀 Non-Linear Analysis")
    st.caption("Distance correlation, mutual information (MI), polynomial regression, and Gaussian Process Regression (GPR)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    nonlinear = NonLinearAnalysis(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Distance Correlation", use_container_width=True):
            with st.spinner("Computing..."):
                # Correct API: distance_correlation(features, target)
                results = nonlinear.distance_correlation(features, target)
                st.session_state.analysis_results['dist_corr'] = results

    with col2:
        if st.button("🔮 Mutual Information", use_container_width=True):
            with st.spinner("Computing..."):
                # Correct API: mutual_information(features, target)
                results = nonlinear.mutual_information(features, target)
                st.session_state.analysis_results['mutual_info'] = results

    with col3:
        max_degree = st.slider("Max Polynomial Degree", 2, 5, 3)
        if st.button("📈 Polynomial Regression", use_container_width=True):
            with st.spinner("Fitting polynomials..."):
                # Correct API: polynomial_regression(features, target, max_degree)
                results = nonlinear.polynomial_regression(features, target, max_degree)
                st.session_state.analysis_results['polynomial'] = results

    st.markdown("---")

    if 'dist_corr' in st.session_state.analysis_results:
        st.subheader("📊 Distance Correlation vs Pearson")

        dist_corr = st.session_state.analysis_results['dist_corr']

        # Calculate Pearson for comparison
        pearson_corr = {}
        for feat in features:
            pearson_corr[feat] = df[feat].corr(df[target])

        comparison_df = pd.DataFrame({
            'Feature': features,
            'Pearson |r|': [abs(pearson_corr.get(f, 0)) for f in features],
            'Distance Corr': [dist_corr.get(f, 0) for f in features],
        })
        comparison_df['Non-linearity'] = comparison_df['Distance Corr'] - comparison_df['Pearson |r|']

        st.dataframe(comparison_df, use_container_width=True)

        fig = go.Figure(data=[
            go.Bar(name='|Pearson|', x=features, y=comparison_df['Pearson |r|'], marker_color='steelblue'),
            go.Bar(name='Distance Corr', x=features, y=comparison_df['Distance Corr'], marker_color='coral')
        ])
        fig.update_layout(barmode='group', title='Pearson vs Distance Correlation',
                         template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Large difference suggests non-linear relationships!")

    if 'mutual_info' in st.session_state.analysis_results:
        st.subheader("🔮 Mutual Information")
        mi = st.session_state.analysis_results['mutual_info']

        fig = go.Figure(data=[
            go.Bar(x=list(mi.keys()), y=list(mi.values()), marker_color='teal')
        ])
        fig.update_layout(title='Mutual Information Scores', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)

    if 'polynomial' in st.session_state.analysis_results:
        st.subheader("📈 Polynomial Regression Results")
        poly_results = st.session_state.analysis_results['polynomial']

        poly_df = pd.DataFrame([
            {'Degree': deg, 'R²': vals['r2'], 'RMSE': vals['rmse']}
            for deg, vals in poly_results.items()
        ])
        st.dataframe(poly_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=poly_df['Degree'], y=poly_df['R²'], mode='lines+markers', name='R²'))
        fig.update_layout(title='R² vs Polynomial Degree', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)


