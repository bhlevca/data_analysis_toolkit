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

from bayesian_analysis import BayesianAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_bayesian_tab():
    """Render Bayesian analysis tab"""
    st.header("📈 Bayesian Inference & Analysis")
    st.caption("Bayesian regression with posterior distributions, credible intervals (CI), and model comparison using BIC")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    bayesian = BayesianAnalysis(df)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Bayesian Regression", use_container_width=True):
            with st.spinner("Fitting Bayesian model..."):
                # Correct API: bayesian_regression(features, target)
                results = bayesian.bayesian_regression(features, target)
                st.session_state.analysis_results['bayesian'] = results

    with col2:
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
        if st.button("📊 Credible Intervals", use_container_width=True):
            with st.spinner("Computing intervals..."):
                # Correct API: credible_intervals(features, target, confidence)
                results = bayesian.credible_intervals(features, target, confidence)
                st.session_state.analysis_results['credible'] = results

    st.markdown("---")

    if 'bayesian' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['bayesian']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎲 Bayesian Regression Results")

            posterior_mean = results.get('posterior_mean', [])
            feat_names = results.get('features', [])
            ci_lower = results.get('credible_intervals_lower', [])
            ci_upper = results.get('credible_intervals_upper', [])

            if len(posterior_mean) > 0:
                coef_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Posterior Mean': posterior_mean,
                    '95% CI Lower': ci_lower,
                    '95% CI Upper': ci_upper
                })
                st.dataframe(coef_df, use_container_width=True)

                # Plot coefficients with error bars
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=feat_names, y=posterior_mean,
                    error_y=dict(type='data',
                                array=np.array(ci_upper) - np.array(posterior_mean),
                                arrayminus=np.array(posterior_mean) - np.array(ci_lower)),
                    marker_color='steelblue'
                ))
                fig.update_layout(title='Posterior Coefficients with 95% CI',
                                template=PLOTLY_TEMPLATE, height=400)
                st.plotly_chart(fig, use_container_width=True)

    if 'credible' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['credible']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("📊 Credible Intervals")
            st.metric("Coverage", f"{results.get('coverage', 0)*100:.1f}%")
            st.metric("Mean CI Width", f"{results.get('mean_ci_width', 0):.4f}")


