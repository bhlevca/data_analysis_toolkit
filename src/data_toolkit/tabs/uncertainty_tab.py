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

from uncertainty_analysis import UncertaintyAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_uncertainty_tab():
    """Render uncertainty analysis tab"""
    st.header("🎲 Uncertainty Analysis")
    st.caption("Bootstrap confidence intervals, residual analysis, Monte Carlo simulation, and prediction intervals")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    uncertainty = UncertaintyAnalysis(df)

    col1, col2 = st.columns(2)

    with col1:
        n_bootstrap = st.slider("Bootstrap Samples", 100, 2000, 500)
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, key='boot_conf')

    with col2:
        n_simulations = st.slider("Monte Carlo Simulations", 100, 5000, 1000)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Bootstrap CI", use_container_width=True):
            with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
                # Correct API: bootstrap_ci(features, target, n_bootstrap, confidence)
                results = uncertainty.bootstrap_ci(features, target, n_bootstrap, confidence)
                st.session_state.analysis_results['bootstrap'] = results

    with col2:
        if st.button("🎯 Residual Analysis", use_container_width=True):
            with st.spinner("Analyzing residuals..."):
                # Correct API: residual_analysis(features, target)
                results = uncertainty.residual_analysis(features, target)
                st.session_state.analysis_results['residuals'] = results

    with col3:
        if st.button("🎲 Monte Carlo", use_container_width=True):
            with st.spinner(f"Running {n_simulations} simulations..."):
                # Correct API: monte_carlo(features, target, n_simulations, confidence)
                results = uncertainty.monte_carlo_analysis(features, target, n_simulations, confidence)
                st.session_state.analysis_results['monte_carlo'] = results

    st.markdown("---")

    if 'bootstrap' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['bootstrap']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🔄 Bootstrap Results")

            boot_df = pd.DataFrame({
                'Feature': results.get('features', []),
                'Mean': results.get('mean_coefs', []),
                'Std Error': results.get('std_coefs', []),
                'CI Lower': results.get('ci_lower', []),
                'CI Upper': results.get('ci_upper', [])
            })
            st.dataframe(boot_df, use_container_width=True)

    if 'residuals' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['residuals']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎯 Residual Analysis")

            col1, col2, col3 = st.columns(3)
            col1.metric("Durbin-Watson", f"{results.get('durbin_watson', 0):.4f}")
            col2.metric("Shapiro p-value", f"{results.get('shapiro_pvalue', 0):.4f}")
            col3.metric("Mean Residual", f"{results.get('mean', 0):.4f}")

            # Residual plots
            residuals = results.get('residuals', [])
            y_pred = results.get('y_pred', [])

            if len(residuals) > 0:
                fig = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Residuals vs Fitted', 'Residual Distribution'))

                fig.add_trace(
                    go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(opacity=0.6)),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)

                fig.add_trace(
                    go.Histogram(x=residuals, nbinsx=30),
                    row=1, col=2
                )

                fig.update_layout(height=400, template=PLOTLY_TEMPLATE, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    if 'monte_carlo' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['monte_carlo']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎲 Monte Carlo Results")

            col1, col2 = st.columns(2)
            col1.metric("Mean Uncertainty", f"{results.get('mean_uncertainty', 0):.4f}")
            col2.metric("Mean CI Width", f"{results.get('mean_ci_width', 0):.4f}")


