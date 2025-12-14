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
    # DEBUG: Show current session state for diagnosis
    st.markdown("**[DEBUG] session_state.df type:** {}".format(type(st.session_state.get('df', 'NOT SET'))))
    st.markdown("**[DEBUG] session_state.feature_cols:** {}".format(st.session_state.get('feature_cols', 'NOT SET')))
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
        # Add sliders for method-specific parameters
        if outlier_method == 'iqr':
            iqr_multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, step=0.1, help="Controls the whisker length for IQR outlier detection.")
            zscore_threshold = 3.0  # Not used
        else:
            zscore_threshold = st.slider("Z-score Threshold", 2.0, 5.0, 3.0, step=0.1, help="Threshold for Z-score outlier detection.")
            iqr_multiplier = 1.5  # Not used
        if st.button("🎯 Outlier Detection", use_container_width=True):
            st.session_state.analysis_results['outliers'] = stats.outlier_detection(
                features,
                method=outlier_method,
                iqr_multiplier=iqr_multiplier,
                zscore_threshold=zscore_threshold
            )

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

        for col in features:
            info = outlier_data.get(col, {})
            n_outliers = info.get('n_outliers', 0)
            pct = info.get('percentage', 0)
            st.markdown(f"### {col}: {n_outliers} outliers ({pct:.1f}%)")
            if info:
                if 'lower_bound' in info:
                    st.write(f"Lower bound: {info['lower_bound']:.4f}")
                if 'upper_bound' in info:
                    st.write(f"Upper bound: {info['upper_bound']:.4f}")
            else:
                st.info("No outlier info available for this feature.")

            # Show plot and table in expanders, expanded by default, smaller font
            with st.expander("Show Outlier Plot (Plotly)", expanded=True):
                try:
                    # Plotly version of classic line plot
                    series = df[col]
                    outlier_indices = info.get('outlier_indices', [])
                    lower = info.get('lower_bound', None)
                    upper = info.get('upper_bound', None)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines+markers',
                        name='Data',
                        line=dict(color='royalblue', width=2),
                        marker=dict(size=4)
                    ))
                    if outlier_indices:
                        fig.add_trace(go.Scatter(
                            x=series.index[outlier_indices],
                            y=series.values[outlier_indices],
                            mode='markers',
                            name='Outliers',
                            marker=dict(color='red', size=7, symbol='x')
                        ))
                    if lower is not None:
                        fig.add_hline(y=lower, line_dash='dash', line_color='orange', annotation_text='Lower Bound', annotation_position='bottom left')
                    if upper is not None:
                        fig.add_hline(y=upper, line_dash='dash', line_color='orange', annotation_text='Upper Bound', annotation_position='top left')
                    fig.update_layout(
                        title=f"{col} with Outliers",
                        template=PLOTLY_TEMPLATE,
                        height=300,
                        font=dict(size=16),
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(title_font=dict(size=15), tickfont=dict(size=14)),
                        yaxis=dict(title_font=dict(size=15), tickfont=dict(size=14))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"[PLOT ERROR] Could not plot outlier line for {col}: {e}")

            with st.expander("Show Outlier Table", expanded=True):
                try:
                    outlier_table = stats.outlier_table(col, info)
                    st.dataframe(outlier_table, use_container_width=True)
                    if outlier_table.empty:
                        st.info("No outliers detected.")
                except Exception as e:
                    st.error(f"[TABLE ERROR] Could not display outlier table for {col}: {e}")

        # Box plots (shown at the end, less prominent)
        st.markdown('---')
        st.subheader('Box Plots (All Features)')
        box_data = df[features].melt(var_name='_Feature_', value_name='_Value_')
        fig = px.box(box_data, x='_Feature_', y='_Value_', title='Box Plots with Outliers',
                    template=PLOTLY_TEMPLATE, points='outliers')
        fig.update_layout(
            height=400,
            font=dict(size=16),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(title_font=dict(size=15), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=15), tickfont=dict(size=14))
        )
        st.plotly_chart(fig, use_container_width=True)


