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

from causality_analysis import CausalityAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_causality_tab():
    """Render causality analysis tab"""
    st.header("🔗 Causality Analysis (Granger Causality)")
    st.caption("Test predictive causality with Granger causality tests and lead-lag correlation analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    causality = CausalityAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        max_lag = st.slider("Max Lag", 1, 20, 10, key='causality_lag')
        selected_feature = st.selectbox("Test Feature", features)

    with col2:
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔬 Granger Causality", use_container_width=True):
                with st.spinner("Testing..."):
                    # Correct API: granger_causality([features], target, max_lag)
                    results = causality.granger_causality([selected_feature], target, max_lag)
                    st.session_state.analysis_results['granger'] = results.get(selected_feature, {})
                    st.session_state.analysis_results['granger_feature'] = selected_feature

        with col_b:
            if st.button("⏱️ Lead-Lag Analysis", use_container_width=True):
                with st.spinner("Computing..."):
                    # Correct API: lead_lag_analysis([features], target, max_lag)
                    results = causality.lead_lag_analysis([selected_feature], target, max_lag)
                    st.session_state.analysis_results['lead_lag'] = results.get(selected_feature, {})
                    st.session_state.analysis_results['lead_lag_feature'] = selected_feature

    st.markdown("---")

    if 'granger' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['granger']
        feat = st.session_state.analysis_results.get('granger_feature', 'Feature')

        if 'error' in results:
            st.error(results['error'])
        elif results:
            st.subheader("🔬 Granger Causality Test")
            st.write(f"Testing: Does **{feat}** Granger-cause **{target}**?")

            granger_df = pd.DataFrame([
                {
                    'Lag': lag,
                    'p-value': data.get('ssr_ftest_pvalue', 0),
                    'Significant': '✅ Yes' if data.get('is_significant', False) else '❌ No'
                }
                for lag, data in results.items() if isinstance(data, dict)
            ])
            st.dataframe(granger_df, use_container_width=True)

            # Plot p-values
            lags = [lag for lag in results.keys() if isinstance(results[lag], dict)]
            pvals = [results[lag].get('ssr_ftest_pvalue', 0) for lag in lags]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lags, y=pvals, mode='lines+markers'))
            fig.add_hline(y=0.05, line_dash='dash', line_color='red', annotation_text='p=0.05')
            fig.update_layout(title='Granger Causality p-values', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)

    if 'lead_lag' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['lead_lag']
        feat = st.session_state.analysis_results.get('lead_lag_feature', 'Feature')

        if results:
            st.subheader("⏱️ Lead-Lag Analysis")

            col1, col2 = st.columns(2)
            col1.metric("Best Lag", results.get('best_lag', 0))
            col2.metric("Max Correlation", f"{results.get('best_correlation', 0):.4f}")

            best_lag = results.get('best_lag', 0)
            if best_lag < 0:
                st.info(f"📊 **{feat}** leads **{target}** by {abs(best_lag)} periods")
            elif best_lag > 0:
                st.info(f"📊 **{target}** leads **{feat}** by {best_lag} periods")
            else:
                st.info(f"📊 **{feat}** and **{target}** move together")

            # Plot correlations
            lags = results.get('lags', [])
            corrs = results.get('correlations', [])

            fig = go.Figure()
            fig.add_trace(go.Bar(x=lags, y=corrs, marker_color='steelblue'))
            fig.add_vline(x=best_lag, line_dash='dash', line_color='red')
            fig.update_layout(title='Cross-correlation at Different Lags', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)


