"""
Tab module for the Data Analysis Toolkit
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from timeseries_analysis import TimeSeriesAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_signal_analysis_tab():
    """Render Signal Analysis tab (Fourier & Wavelet)"""
    st.header("🔊 Signal Processing: FFT, PSD & Wavelet Analysis")
    st.caption("FFT (Fast Fourier Transform), PSD (Power Spectral Density), CWT (Continuous Wavelet), DWT (Discrete Wavelet)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select a column.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    ts = TimeSeriesAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        # Exclude 'time' column from signal analysis (it's used for sampling rate detection)
        signal_features = [f for f in features if f.lower() != 'time']
        if not signal_features:
            signal_features = features
        selected_col = st.selectbox("Select Time Series Column", signal_features)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["FFT (Fourier)", "Power Spectral Density", "Continuous Wavelet", "Discrete Wavelet"]
        )

        # Auto-detect sampling rate from data
        if 'time' in df.columns:
            time_diff = df['time'].diff().dropna()
            if len(time_diff) > 0:
                avg_dt = time_diff.mean()
                sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 1.0
                st.success(f"✅ Sampling rate: **{sampling_rate:.1f} Hz** (auto-detected from 'time' column)")
            else:
                sampling_rate = 1.0
                st.warning("⚠️ Could not detect sampling rate from 'time' column. Using 1.0 Hz default.")
        else:
            # Calculate from number of samples assuming 1 second duration
            n_samples = len(df[selected_col].dropna())
            sampling_rate = float(n_samples)
            st.info(f"📊 No 'time' column found. Assuming {n_samples} samples over 1 second → {sampling_rate:.1f} Hz")

        wavelet_type = st.selectbox("Wavelet Type", ["morl", "mexh", "gaus1", "gaus2", "cgau1"], index=0, help="Select the wavelet function for CWT.")
        cwt_scales = st.slider("CWT Scales (max)", 16, 256, 64, help="Maximum number of scales for CWT.")
        y_scale = st.selectbox("CWT Y-axis scale", ["log", "linear"], index=0, help="Y-axis scale for wavelet power plot.")
        significance_level = st.slider("Significance level", 0.80, 0.999, 0.95, step=0.01, help="Significance threshold for Torrence & Compo plot.")
        show_coi = st.checkbox("Show COI (Cone of Influence)", value=True, help="Display Cone of Influence on wavelet plot.")

    with col2:
        # Data quality check
        n_samples = len(df[selected_col].dropna())
        nyquist_freq = sampling_rate / 2.0

        if n_samples < 100:
            st.warning(f"⚠️ Only {n_samples} samples - may not be enough for reliable frequency analysis. Consider using test_data/signal_analysis_sample.csv")

        st.info(f"📊 {n_samples} samples at **{sampling_rate:.1f} Hz** → Nyquist: {nyquist_freq:.1f} Hz (max detectable frequency)")

        # Clear cache button
        if st.button("🗑️ Clear Cached Results", help="Clear all previous analysis results"):
            st.session_state.analysis_results = {}
            st.success("✅ Cache cleared! Run analysis again.")
            st.rerun()

        # Only show the relevant button for the selected analysis type
        if analysis_type == "FFT (Fourier)":
            if st.button("🔍 FFT Analysis", width='stretch'):
                with st.spinner("Computing FFT from loaded data..."):
                    # Use the actual data from the selected column
                    results = ts.fourier_transform(selected_col, sampling_rate=float(sampling_rate))
                    st.session_state.analysis_results['fft'] = results
                    st.success(f"✅ FFT computed: {n_samples} samples at **{sampling_rate:.1f} Hz** → Dominant: {results.get('dominant_frequency', 0):.2f} Hz")
        elif analysis_type == "Power Spectral Density":
            if st.button("📊 PSD Analysis", width='stretch'):
                with st.spinner("Computing PSD..."):
                    # Use the actual data from the selected column
                    results = ts.power_spectral_density(selected_col, sampling_rate=float(sampling_rate))
                    st.session_state.analysis_results['psd'] = results
                    st.success(f"✅ PSD computed on {len(df[selected_col].dropna())} samples from column '{selected_col}' at {sampling_rate} Hz")
        elif analysis_type == "Continuous Wavelet":
            if st.button("🌊 CWT Analysis", width='stretch'):
                with st.spinner("Computing Continuous Wavelet Transform..."):
                    results = ts.continuous_wavelet_transform(selected_col, scales=None, wavelet=wavelet_type, sampling_rate=float(sampling_rate))
                    if 'error' in results:
                        st.error(f"CWT failed: {results['error']}")
                    else:
                        st.session_state.analysis_results['cwt'] = results
                        st.session_state.analysis_results['cwt_options'] = {
                            'y_scale': y_scale,
                            'significance_level': significance_level,
                            'show_coi': show_coi,
                            'wavelet_type': wavelet_type
                        }
                        power = results.get('power', np.array([]))
                        st.success(f"✅ CWT computed: {power.shape[1]} time points × {power.shape[0]} scales using '{wavelet_type}' wavelet")
        elif analysis_type == "Discrete Wavelet":
            dwt_wavelet_type = st.selectbox("Wavelet Type (Discrete)", ["db4", "db8", "sym4", "coif1", "haar"], index=0, help="Select discrete wavelet for DWT. Common choices: db4 (Daubechies 4), haar (simplest)")
            level = st.slider("Decomposition Level", 1, 5, 3, help="Level of wavelet decomposition.")
            if st.button("🌀 DWT Analysis", width='stretch'):
                with st.spinner("Computing DWT..."):
                    # Use the actual data from the selected column with discrete wavelet
                    results = ts.discrete_wavelet_transform(selected_col, wavelet=dwt_wavelet_type, level=level)
                    st.session_state.analysis_results['dwt'] = results
                    st.session_state.analysis_results['dwt_wavelet'] = dwt_wavelet_type
                    st.success(f"✅ DWT computed on {len(df[selected_col].dropna())} samples from column '{selected_col}'")

    st.markdown("---")
    # ...existing code...

    # Display results
    # Combined FFT and PSD panel plot
    fft_res = st.session_state.analysis_results.get('fft')
    psd_res = st.session_state.analysis_results.get('psd')
    if fft_res and psd_res and ('error' not in fft_res) and ('error' not in psd_res):
        st.subheader("🔍 FFT & 📊 Power Spectral Density (Combined)")
        col1, col2 = st.columns(2)
        col1.metric("Dominant Frequency (FFT)", f"{fft_res.get('dominant_frequency', 0):.4f}")
        col2.metric("Dominant Frequency (PSD)", f"{psd_res.get('dominant_frequency', 0):.4f}")

        # Prepare subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("FFT Magnitude Spectrum", "Power Spectral Density"))
        # FFT panel
        frequencies = fft_res.get('frequencies', [])
        magnitude = fft_res.get('magnitude', [])
        if len(frequencies) > 0 and len(magnitude) > 0:
            fig.add_trace(
                go.Scatter(x=frequencies[:len(frequencies)//2], y=magnitude[:len(magnitude)//2],
                           mode='lines', fill='tozeroy', name='FFT'),
                row=1, col=1
            )
        # PSD panel
        psd_freq = psd_res.get('frequencies', [])
        psd_vals = psd_res.get('power_spectral_density', [])
        if len(psd_freq) > 0 and len(psd_vals) > 0:
            fig.add_trace(
                go.Scatter(x=psd_freq, y=psd_vals, mode='lines', fill='tozeroy', name='PSD', line=dict(color='orange')),
                row=2, col=1
            )
        fig.update_layout(height=700, template=PLOTLY_TEMPLATE)
        fig.update_xaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Fallback: show FFT or PSD individually if only one is present
        if fft_res and ('error' not in fft_res):
            st.subheader("🔍 FFT Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Dominant Frequency (Hz)", f"{fft_res.get('dominant_frequency', 0):.2f}")
            col2.metric("Peak Power", f"{fft_res.get('peak_power', 0):.2e}")

            # Plot FFT spectrum
            frequencies = fft_res.get('positive_frequencies', [])
            magnitude = fft_res.get('magnitude', [])

            if len(frequencies) > 0 and len(magnitude) > 0:
                # Use only positive frequencies for cleaner plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frequencies,
                    y=magnitude,
                    mode='lines',
                    fill='tozeroy',
                    name='FFT Magnitude',
                    line=dict(color='steelblue')
                ))
                fig.update_layout(
                    title='FFT Magnitude Spectrum (Positive Frequencies)',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Magnitude',
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show top frequencies
                st.write("**Top 5 Dominant Frequencies:**")
                top_freqs = fft_res.get('dominant_frequencies', [])
                top_powers = fft_res.get('dominant_powers', [])
                for i, (f, p) in enumerate(zip(top_freqs[:5], top_powers[:5]), 1):
                    st.write(f"{i}. {f:.2f} Hz - Power: {p:.2e}")
        if psd_res and ('error' not in psd_res):
            st.subheader("📊 Power Spectral Density")
            col1, col2 = st.columns(2)
            col1.metric("Dominant Frequency", f"{psd_res.get('dominant_frequency', 0):.4f}")
            col2.metric("Total Power", f"{psd_res.get('total_power', 0):.4f}")
            psd_freq = psd_res.get('frequencies', [])
            psd_vals = psd_res.get('power_spectral_density', [])
            if len(psd_freq) > 0 and len(psd_vals) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=psd_freq, y=psd_vals, mode='lines', fill='tozeroy', line=dict(color='orange')))
                fig.update_layout(title='Power Spectral Density', xaxis_title='Frequency',
                                yaxis_title='Power', template=PLOTLY_TEMPLATE, height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Export FFT/PSD results
    if fft_res or psd_res:
        st.subheader("📥 Export Spectral Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            if fft_res and 'error' not in fft_res:
                fft_df = pd.DataFrame({
                    'Frequency_Hz': fft_res.get('positive_frequencies', []),
                    'Magnitude': fft_res.get('magnitude', []),
                    'Power': fft_res.get('power', [])
                })
                csv_fft = fft_df.to_csv(index=False)
                st.download_button(
                    label="📥 FFT Results (CSV)",
                    data=csv_fft,
                    file_name="fft_results.csv",
                    mime="text/csv"
                )

        with col2:
            if psd_res and 'error' not in psd_res:
                psd_df = pd.DataFrame({
                    'Frequency_Hz': psd_res.get('frequencies', []),
                    'Power_Spectral_Density': psd_res.get('power_spectral_density', [])
                })
                csv_psd = psd_df.to_csv(index=False)
                st.download_button(
                    label="📥 PSD Results (CSV)",
                    data=csv_psd,
                    file_name="psd_results.csv",
                    mime="text/csv"
                )

    if 'cwt' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['cwt']
        if 'error' not in results:
            st.subheader("🌊 Continuous Wavelet Transform")
            st.info("Time-frequency analysis showing power at each frequency over time")
            try:
                cwt_opts = st.session_state.analysis_results.get('cwt_options', {})
                y_scale_opt = cwt_opts.get('y_scale', 'log')
                signif_opt = cwt_opts.get('significance_level', 0.95)
                show_coi_opt = cwt_opts.get('show_coi', True)
                wavelet_type_opt = cwt_opts.get('wavelet_type', 'morl')

                # Create and display CWT plot
                fig = ts.plot_wavelet_torrence(
                    results,
                    selected_col,
                    y_scale=y_scale_opt,
                    significance_level=signif_opt,
                    show_coi=show_coi_opt,
                    wavelet=wavelet_type_opt
                )
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Export CWT results
                st.subheader("📥 Export CWT Results")
                col1, col2 = st.columns(2)
                with col1:
                    cwt_summary_df = pd.DataFrame({
                        'Scale': results.get('scales', []),
                        'Period': results.get('periods', []),
                        'Global_Power': np.mean(results.get('power', np.array([[0]])), axis=1)
                    })
                    csv_cwt = cwt_summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CWT Summary (CSV)",
                        data=csv_cwt,
                        file_name="cwt_summary.csv",
                        mime="text/csv"
                    )
                with col2:
                    scale_avg_power = np.mean(results.get('power', np.array([[0]])), axis=0)
                    cwt_time_df = pd.DataFrame({
                        'Time': results.get('time', []),
                        'Scale_Averaged_Power': scale_avg_power
                    })
                    csv_cwt_time = cwt_time_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CWT Time Series (CSV)",
                        data=csv_cwt_time,
                        file_name="cwt_time_series.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"CWT plotting failed: {str(e)}")

    if 'dwt' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['dwt']
        if 'error' not in results:
            st.subheader("🌀 Discrete Wavelet Transform")
            try:
                fig = ts.plot_discrete_wavelet(results, selected_col)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("DWT data available but no plot generated")

                # Export DWT results
                st.subheader("📥 Export DWT Results")
                coefficients = results.get('coefficients', [])
                if coefficients:
                    # Create summary of DWT decomposition
                    dwt_summary = []
                    for c in coefficients:
                        dwt_summary.append({
                            'Level': c['level'],
                            'Detail_Length': c['detail_length'],
                            'Approx_Length': c['approximation_length'],
                            'Detail_RMS': np.sqrt(np.mean(np.array(c['detail'])**2)),
                            'Detail_Max': np.max(np.abs(c['detail'])),
                            'Detail_Energy': np.sum(np.array(c['detail'])**2)
                        })
                    dwt_df = pd.DataFrame(dwt_summary)
                    csv_dwt = dwt_df.to_csv(index=False)
                    st.download_button(
                        label="📥 DWT Summary (CSV)",
                        data=csv_dwt,
                        file_name="dwt_summary.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"DWT plotting failed: {str(e)}")


