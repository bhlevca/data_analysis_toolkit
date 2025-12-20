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
    st.plotly_chart(fig, width='stretch')

    with col2:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("📊 ACF", width='stretch'):
                # Correct API: acf_analysis(column, lags)
                results = ts.acf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['acf'] = results

        with col_b:
            if st.button("📈 PACF", width='stretch'):
                # Correct API: pacf_analysis(column, lags)
                results = ts.pacf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['pacf'] = results

        with col_c:
            if st.button("🔬 Stationarity", width='stretch'):
                # Correct API: stationarity_test([columns])
                results = ts.stationarity_test([selected_col])
                st.session_state.analysis_results['adf'] = results.get(selected_col, {})

        with col_d:
            default_win = min(30, max(1, len(series)//5))
            window = st.number_input("Rolling window (samples)", min_value=1, max_value=max(1, len(series)), value=default_win, step=1)
            if st.button("🔄 Rolling Stats", width='stretch'):
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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')

    # ===== ARIMA/SARIMA FORECASTING SECTION =====
    st.markdown("---")
    st.subheader("🔮 ARIMA/SARIMA Forecasting")
    st.caption("Autoregressive Integrated Moving Average models for time series forecasting")
    
    with st.expander("📊 Configure Forecasting Model", expanded=False):
        fcol1, fcol2 = st.columns(2)
        
        with fcol1:
            st.markdown("**Model Configuration**")
            use_auto = st.checkbox("Auto-select ARIMA parameters (recommended)", value=True,
                                   help="Automatically find best (p,d,q) parameters using AIC")
            
            if not use_auto:
                st.markdown("Manual ARIMA Order (p, d, q):")
                p_order = st.slider("p (AR order)", 0, 5, 1, help="Autoregressive order")
                d_order = st.slider("d (Differencing)", 0, 2, 1, help="Differencing order")
                q_order = st.slider("q (MA order)", 0, 5, 1, help="Moving average order")
            else:
                p_order, d_order, q_order = None, None, None
            
            use_seasonal = st.checkbox("Include seasonal component (SARIMA)", value=False,
                                       help="Add seasonal terms for data with periodic patterns")
            
            if use_seasonal:
                st.markdown("Seasonal Order (P, D, Q, m):")
                scol1, scol2 = st.columns(2)
                with scol1:
                    P_order = st.slider("P (Seasonal AR)", 0, 2, 1)
                    D_order = st.slider("D (Seasonal diff)", 0, 1, 1)
                with scol2:
                    Q_order = st.slider("Q (Seasonal MA)", 0, 2, 1)
                    m_period = st.selectbox("m (Period)", [4, 7, 12, 24, 52], index=2,
                                           help="4=quarterly, 7=weekly, 12=monthly, 24=hourly, 52=weekly-yearly")
                seasonal_order = (P_order, D_order, Q_order, m_period)
            else:
                seasonal_order = None
        
        with fcol2:
            st.markdown("**Forecast Settings**")
            forecast_steps = st.slider("Forecast horizon (steps)", 5, 100, 20,
                                       help="Number of future time points to predict")
            confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, step=0.01,
                                   help="Confidence interval width")
        
        if st.button("🚀 Run Forecast", width='stretch'):
            with st.spinner("Fitting ARIMA model and generating forecast..."):
                if use_auto:
                    order = None
                else:
                    order = (p_order, d_order, q_order)
                
                results = ts.arima_forecast(
                    selected_col,
                    steps=forecast_steps,
                    order=order,
                    seasonal_order=seasonal_order,
                    confidence_level=confidence
                )
                st.session_state.analysis_results['forecast'] = results
                
                if 'error' in results:
                    st.error(f"Forecast failed: {results['error']}")
                else:
                    order_str = f"ARIMA{results.get('order', (0,0,0))}"
                    if results.get('seasonal_order'):
                        order_str = f"SARIMA{results.get('order')}x{results.get('seasonal_order')}"
                    st.success(f"✅ {order_str} fitted successfully - AIC: {results.get('aic', 0):.2f}")
    
    # Display forecast results
    if 'forecast' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['forecast']
        
        if 'error' not in results:
            st.markdown("#### 📈 Forecast Results")
            
            # Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            order = results.get('order', (0, 0, 0))
            mcol1.metric("Model", f"ARIMA{order}")
            mcol2.metric("AIC", f"{results.get('aic', 0):.2f}")
            mcol3.metric("BIC", f"{results.get('bic', 0):.2f}")
            mcol4.metric("Forecast Steps", len(results.get('forecast', [])))
            
            # Plot forecast
            original = results.get('original_data', [])
            fitted = results.get('fitted_values', [])
            forecast = results.get('forecast', [])
            lower_ci = results.get('lower_ci', [])
            upper_ci = results.get('upper_ci', [])
            conf_level = results.get('confidence_level', 0.95)
            
            n_original = len(original)
            n_forecast = len(forecast)
            
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=list(range(n_original)), y=original,
                mode='lines', name='Original', line=dict(color='blue', width=1.5)
            ))
            
            # Fitted values
            fig.add_trace(go.Scatter(
                x=list(range(n_original)), y=fitted,
                mode='lines', name='Fitted', line=dict(color='green', width=1, dash='dash'),
                opacity=0.7
            ))
            
            # Forecast
            forecast_x = list(range(n_original, n_original + n_forecast))
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast,
                mode='lines', name='Forecast', line=dict(color='red', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(upper_ci) + list(lower_ci)[::-1],
                fill='toself', fillcolor='rgba(255,0,0,0.15)',
                line=dict(color='rgba(255,0,0,0)'),
                name=f'{conf_level*100:.0f}% CI'
            ))
            
            order_str = f"ARIMA{order}"
            if results.get('seasonal_order'):
                order_str = f"SARIMA{order}x{results.get('seasonal_order')}"
            
            fig.update_layout(
                title=f'{order_str} Forecast - {selected_col}',
                xaxis_title='Time Index',
                yaxis_title='Value',
                template=PLOTLY_TEMPLATE,
                height=500,
                hovermode='x unified'
            )
            fig.add_vline(x=n_original - 1, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, width='stretch')
            
            # Forecast table
            with st.expander("📋 Forecast Values"):
                forecast_df = pd.DataFrame({
                    'Step': range(1, n_forecast + 1),
                    'Forecast': forecast,
                    'Lower CI': lower_ci,
                    'Upper CI': upper_ci
                })
                st.dataframe(forecast_df, width='stretch')
            
            # Residual diagnostics
            with st.expander("🔍 Residual Diagnostics"):
                residuals = results.get('residuals', [])
                if len(residuals) > 0:
                    rcol1, rcol2 = st.columns(2)
                    
                    with rcol1:
                        # Residual plot
                        fig_res = go.Figure()
                        fig_res.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals'))
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_res.update_layout(title='Residuals Over Time', template=PLOTLY_TEMPLATE, height=300)
                        st.plotly_chart(fig_res, width='stretch')
                    
                    with rcol2:
                        # Residual histogram
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
                        fig_hist.update_layout(title='Residual Distribution', template=PLOTLY_TEMPLATE, height=300)
                        st.plotly_chart(fig_hist, width='stretch')
                    
                    # Stats
                    scol1, scol2, scol3 = st.columns(3)
                    scol1.metric("Mean", f"{np.mean(residuals):.4f}")
                    scol2.metric("Std Dev", f"{np.std(residuals):.4f}")
                    scol3.metric("Skewness", f"{pd.Series(residuals).skew():.4f}")
            
            # Export
            st.subheader("📥 Export Forecast")
            export_df = pd.DataFrame({
                'Step': range(1, n_forecast + 1),
                'Forecast': forecast,
                'Lower_CI': lower_ci,
                'Upper_CI': upper_ci
            })
            csv_forecast = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv_forecast,
                file_name="arima_forecast.csv",
                mime="text/csv"
            )

    # ===== MULTIVARIATE TIME SERIES SECTION =====
    st.markdown("---")
    st.subheader("🔗 Multivariate Time Series Analysis")
    st.caption("Vector Autoregression (VAR), Vector Error Correction (VECM), and Dynamic Time Warping (DTW)")
    
    with st.expander("📊 VAR / VECM / DTW Analysis", expanded=False):
        multi_analysis_type = st.radio(
            "Analysis Type",
            ["VAR (Vector Autoregression)", "VECM (Error Correction)", "DTW (Dynamic Time Warping)"],
            horizontal=True
        )
        
        if multi_analysis_type == "VAR (Vector Autoregression)":
            st.markdown("**VAR models capture linear interdependencies among multiple time series.**")
            
            # Select multiple columns
            var_cols = st.multiselect(
                "Select columns for VAR",
                features,
                default=features[:min(3, len(features))],
                help="Select 2 or more columns for multivariate analysis"
            )
            
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                var_maxlags = st.slider("Max lags to consider", 1, 20, 10)
            with vcol2:
                var_ic = st.selectbox("Information criterion", ["aic", "bic", "hqic"], index=0)
            
            if len(var_cols) >= 2 and st.button("🔗 Fit VAR Model", width='stretch'):
                with st.spinner("Fitting VAR model..."):
                    results = ts.var_model(var_cols, maxlags=var_maxlags, ic=var_ic)
                    st.session_state.analysis_results['var'] = results
                    
                    if 'error' in results:
                        st.error(f"VAR failed: {results['error']}")
                    else:
                        st.success(f"✅ VAR({results.get('optimal_lag', 1)}) fitted - AIC: {results.get('aic', 0):.2f}")
            elif len(var_cols) < 2:
                st.warning("⚠️ Select at least 2 columns for VAR")
        
        elif multi_analysis_type == "VECM (Error Correction)":
            st.markdown("**VECM is for cointegrated (non-stationary but share equilibrium) time series.**")
            
            vecm_cols = st.multiselect(
                "Select columns for VECM",
                features,
                default=features[:min(3, len(features))],
                help="Select 2 or more columns"
            )
            
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                det_options = {
                    "Constant in cointegration (co)": "co",
                    "Constant inside (ci)": "ci", 
                    "Linear outside (lo)": "lo",
                    "Linear inside (li)": "li",
                    "No deterministic (n)": "n"
                }
                det_choice = st.selectbox("Deterministic terms", list(det_options.keys()), index=0,
                                        help="Type of deterministic terms in VECM")
                det_order = det_options[det_choice]
            with vcol2:
                k_ar_diff = st.slider("Lagged differences", 1, 5, 1)
            
            if len(vecm_cols) >= 2 and st.button("🔗 Fit VECM Model", width='stretch'):
                with st.spinner("Fitting VECM and testing cointegration..."):
                    results = ts.vecm_model(vecm_cols, deterministic=det_order, k_ar_diff=k_ar_diff)
                    st.session_state.analysis_results['vecm'] = results
                    
                    if 'error' in results:
                        st.error(f"VECM failed: {results['error']}")
                    elif 'warning' in results:
                        st.warning(results['warning'])
                    else:
                        st.success(f"✅ VECM fitted - Cointegration rank: {results.get('cointegration_rank', 0)}")
            elif len(vecm_cols) < 2:
                st.warning("⚠️ Select at least 2 columns for VECM")
        
        elif multi_analysis_type == "DTW (Dynamic Time Warping)":
            st.markdown("**DTW measures similarity between time series with different timing/speed.**")
            
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                dtw_col1 = st.selectbox("First time series", features, key="dtw1")
            with dcol2:
                other_cols = [f for f in features if f != dtw_col1]
                dtw_col2 = st.selectbox("Second time series", other_cols if other_cols else features, key="dtw2")
            
            dtw_window = st.slider("Sakoe-Chiba band (0=no constraint)", 0, 50, 0,
                                   help="Constrains warping path. 0 means no constraint.")
            
            if st.button("📏 Compute DTW Distance", width='stretch'):
                with st.spinner("Computing DTW alignment..."):
                    window = dtw_window if dtw_window > 0 else None
                    results = ts.dtw_distance(dtw_col1, dtw_col2, window=window)
                    st.session_state.analysis_results['dtw'] = results
                    
                    if 'error' in results:
                        st.error(f"DTW failed: {results['error']}")
                    else:
                        st.success(f"✅ DTW Distance: {results.get('dtw_distance', 0):.4f} (normalized: {results.get('normalized_distance', 0):.4f})")
    
    # Display VAR results
    if 'var' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['var']
        if 'error' not in results:
            st.markdown("#### 🔗 VAR Model Results")
            
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Optimal Lag", results.get('optimal_lag', 0))
            mcol2.metric("AIC", f"{results.get('aic', 0):.2f}")
            mcol3.metric("BIC", f"{results.get('bic', 0):.2f}")
            
            # Granger causality
            granger = results.get('granger_causality', {})
            if granger:
                st.markdown("**Granger Causality Tests (p < 0.05 = significant)**")
                granger_df = pd.DataFrame([
                    {'Relationship': k, 'p-value': v['p_value'], 'Significant': '✅' if v['significant'] else '❌'}
                    for k, v in granger.items()
                ])
                st.dataframe(granger_df, width='stretch')
            
            # Plot
            import matplotlib.pyplot as plt
            fig = ts.plot_var_results(results)
            if fig:
                st.pyplot(fig, width='stretch')
                plt.close(fig)
    
    # Display VECM results
    if 'vecm' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['vecm']
        if 'error' not in results and 'warning' not in results:
            st.markdown("#### 🔗 VECM Model Results")
            
            st.metric("Cointegration Rank", results.get('cointegration_rank', 0))
            
            # Johansen test results
            st.markdown("**Johansen Cointegration Test**")
            trace_stats = results.get('trace_statistics', [])
            crit_vals = results.get('critical_values_5pct', [])
            johansen_df = pd.DataFrame({
                'Hypothesis': [f"r ≤ {i}" for i in range(len(trace_stats))],
                'Trace Statistic': trace_stats,
                'Critical Value (5%)': crit_vals,
                'Reject H0': ['✅' if t > c else '❌' for t, c in zip(trace_stats, crit_vals)]
            })
            st.dataframe(johansen_df, width='stretch')
    
    # Display DTW results
    if 'dtw' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['dtw']
        if 'error' not in results:
            st.markdown("#### 📏 DTW Results")
            
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("DTW Distance", f"{results.get('dtw_distance', 0):.4f}")
            mcol2.metric("Normalized", f"{results.get('normalized_distance', 0):.4f}")
            mcol3.metric("Path Length", results.get('path_length', 0))
            
            if results.get('euclidean_distance') is not None:
                st.metric("Euclidean Distance", f"{results.get('euclidean_distance', 0):.4f}")
            
            # Plot
            import matplotlib.pyplot as plt
            fig = ts.plot_dtw_alignment(results)
            if fig:
                st.pyplot(fig, width='stretch')
                plt.close(fig)


