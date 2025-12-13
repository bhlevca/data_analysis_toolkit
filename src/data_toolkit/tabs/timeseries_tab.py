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

from timeseries_analysis import TimeSeriesAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_timeseries_tab():
    """Render time series analysis tab"""
    st.header("⏱️ Time Series Analysis")
    st.caption("ACF/PACF plots, ADF stationarity test, seasonal decomposition, rolling statistics, and ARIMA modeling")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select at least one column.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    ts = TimeSeriesAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("#### Axis Selection")

        # X-axis selection
        x_options = ['Index (Row Number)'] + all_numeric
        x_col = st.selectbox("X-axis (Time/Index)", x_options,
                            help="Select a column for X-axis (typically time/date) or use row index")

        # Y-axis selection
        selected_col = st.selectbox("Y-axis (Value)", features)

        max_lag = st.slider("Max Lag", 5, 50, 20)

    # Plot the time series
    st.subheader("📈 Time Series Plot")
    series = df[selected_col].dropna()

    fig = go.Figure()

    if x_col == 'Index (Row Number)':
        # Use row index as X-axis
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=selected_col))
        fig.update_layout(xaxis_title='Index', yaxis_title=selected_col)
    else:
        # Use selected column as X-axis
        x_data = df[x_col].loc[series.index]  # Match indices with non-null Y values
        fig.add_trace(go.Scatter(x=x_data, y=series, mode='lines', name=selected_col))
        fig.update_layout(xaxis_title=x_col, yaxis_title=selected_col)

    fig.update_layout(title=f'Time Series: {selected_col}', template=PLOTLY_TEMPLATE, height=400)
    st.plotly_chart(fig, use_container_width=True)

    with col2:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("📊 ACF", use_container_width=True):
                # Correct API: acf_analysis(column, lags)
                results = ts.acf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['acf'] = results

        with col_b:
            if st.button("📈 PACF", use_container_width=True):
                # Correct API: pacf_analysis(column, lags)
                results = ts.pacf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['pacf'] = results

        with col_c:
            if st.button("🔬 Stationarity", use_container_width=True):
                # Correct API: stationarity_test([columns])
                results = ts.stationarity_test([selected_col])
                st.session_state.analysis_results['adf'] = results.get(selected_col, {})

        with col_d:
            default_win = min(30, max(1, len(series)//5))
            window = st.number_input("Rolling window (samples)", min_value=1, max_value=max(1, len(series)), value=default_win, step=1)
            if st.button("🔄 Rolling Stats", use_container_width=True):
                # Correct API: rolling_statistics(column, window)
                results = ts.rolling_statistics(selected_col, int(window))
                st.session_state.analysis_results['rolling'] = results

    st.markdown("---")

    if 'adf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['adf']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🔬 Augmented Dickey-Fuller Test")

            col1, col2, col3 = st.columns(3)
            col1.metric("ADF Statistic", f"{results.get('adf_statistic', 0):.4f}")
            col2.metric("p-value", f"{results.get('p_value', 0):.4f}")

            if results.get('is_stationary', False):
                col3.success("✅ Stationary")
            else:
                col3.warning("⚠️ Non-stationary")

    if 'acf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['acf']
        acf_values = results.get('acf', [])
        conf_upper = results.get('conf_int_upper', 0)

        if len(acf_values) > 0:
            st.subheader("📊 ACF")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='steelblue'))
            fig.add_hline(y=conf_upper, line_dash='dash', line_color='red')
            fig.add_hline(y=-conf_upper, line_dash='dash', line_color='red')
            fig.update_layout(title='Autocorrelation Function', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)

    if 'pacf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pacf']
        pacf_values = results.get('pacf', [])
        conf_upper = results.get('conf_int_upper', 0)

        if len(pacf_values) > 0:
            st.subheader("📈 PACF")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='teal'))
            fig.add_hline(y=conf_upper, line_dash='dash', line_color='red')
            fig.add_hline(y=-conf_upper, line_dash='dash', line_color='red')
            fig.update_layout(title='Partial Autocorrelation Function', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)

    if 'rolling' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['rolling']

        if 'error' not in results:
            st.subheader("🔄 Rolling Statistics")

            original = results.get('original', [])
            rolling_mean = results.get('rolling_mean', [])
            rolling_std = results.get('rolling_std', [])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=original, mode='lines', name='Original', opacity=0.7))
            fig.add_trace(go.Scatter(y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='red')))
            fig.add_trace(go.Scatter(y=rolling_std, mode='lines', name='Rolling Std', line=dict(color='green')))
            fig.update_layout(title='Rolling Statistics', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)


