"""
Tab module for the Data Analysis Toolkit
"""

import os
import sys

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

from statistical_analysis import StatisticalAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_statistical_tab():
    """Render statistical analysis tab"""
    st.header("📊 Descriptive Statistics & Correlation Analysis")
    st.caption("Compute summary statistics and correlation matrices")

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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Descriptive Statistics", width='stretch'):
            st.session_state.analysis_results['descriptive'] = stats.descriptive_stats(features)

    with col2:
        corr_method = st.selectbox("Correlation Method", ['pearson', 'spearman', 'kendall'])
        if st.button("🔗 Correlation Matrix", width='stretch'):
            st.session_state.analysis_results['correlation'] = stats.correlation_matrix(features, method=corr_method)

    st.markdown("---")

    # Display results
    if 'descriptive' in st.session_state.analysis_results:
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(st.session_state.analysis_results['descriptive'], width='stretch')

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
        st.plotly_chart(fig, width='stretch')

    # Box plots for quick visual inspection (always show when features selected)
    st.markdown('---')
    st.subheader('📦 Box Plots (Distribution Overview)')
    st.caption("Quick visual inspection of data distributions. For detailed outlier detection (IQR, Z-score, MAD), see the **Data Quality & Outliers** subtab.")
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
    st.plotly_chart(fig, width='stretch')

    # ===== PROBABILITY DISTRIBUTION ANALYSIS =====
    st.markdown("---")
    st.subheader("📊 Probability Distribution Analysis")
    st.caption("Fit distributions, analyze random variables, and test goodness-of-fit")
    
    with st.expander("🎲 Distribution Fitting & Random Variable Analysis", expanded=False):
        dist_col = st.selectbox("Select column for distribution analysis", features, key="dist_col")
        
        dcol1, dcol2 = st.columns(2)
        
        with dcol1:
            st.markdown("**Distribution Fitting**")
            available_dists = ['normal', 't', 'gamma', 'exponential', 'lognormal', 
                              'weibull', 'uniform', 'laplace', 'logistic', 'pareto', 'cauchy']
            selected_dists = st.multiselect(
                "Distributions to fit",
                available_dists,
                default=['normal', 't', 'gamma', 'lognormal'],
                help="Select distributions to fit to your data"
            )
            
            if st.button("📈 Fit Distributions", width='stretch'):
                with st.spinner("Fitting distributions..."):
                    results = stats.fit_extended_distributions(dist_col, selected_dists)
                    st.session_state.analysis_results['dist_fit'] = results
                    
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        best_aic = results.get('best_fit_aic', 'N/A')
                        best_ks = results.get('best_fit_ks', 'N/A')
                        st.success(f"✅ Fitted {len(selected_dists)} distributions - Best (AIC): {best_aic}, Best (KS): {best_ks}")
        
        with dcol2:
            st.markdown("**Random Variable Analysis**")
            
            if st.button("📊 Analyze Random Variable", width='stretch'):
                with st.spinner("Computing moments and quantiles..."):
                    results = stats.random_variable_analysis(dist_col)
                    st.session_state.analysis_results['rv_analysis'] = results
                    
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.success(f"✅ Analysis complete - n={results['n']}")
            
            if st.button("📉 Generate Distribution Plot", width='stretch'):
                with st.spinner("Generating plots..."):
                    import matplotlib.pyplot as plt
                    fig = stats.plot_probability_analysis(dist_col)
                    if fig:
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)
    
    # Display distribution fit results
    if 'dist_fit' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['dist_fit']
        if 'error' not in results:
            st.markdown("#### 📈 Distribution Fitting Results")
            
            # Data summary
            summary = results.get('data_summary', {})
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("n", summary.get('n', 0))
            scol2.metric("Mean", f"{summary.get('mean', 0):.4f}")
            scol3.metric("Skewness", f"{summary.get('skewness', 0):.4f}")
            scol4.metric("Kurtosis", f"{summary.get('kurtosis', 0):.4f}")
            
            # Best fits
            bcol1, bcol2 = st.columns(2)
            bcol1.success(f"**Best by AIC:** {results.get('best_fit_aic', 'N/A')}")
            bcol2.success(f"**Best by KS test:** {results.get('best_fit_ks', 'N/A')}")
            
            # Distribution comparison table
            dist_data = results.get('distributions', {})
            if dist_data:
                table_data = []
                for dist_name, dist_info in dist_data.items():
                    if 'error' not in dist_info:
                        table_data.append({
                            'Distribution': dist_name.title(),
                            'KS Statistic': f"{dist_info.get('ks_statistic', 0):.4f}",
                            'KS p-value': f"{dist_info.get('ks_pvalue', 0):.4f}",
                            'AIC': f"{dist_info.get('aic', float('inf')):.2f}",
                            'BIC': f"{dist_info.get('bic', float('inf')):.2f}",
                            'Good Fit (p>0.05)': '✅' if dist_info.get('ks_pvalue', 0) > 0.05 else '❌'
                        })
                
                if table_data:
                    dist_df = pd.DataFrame(table_data)
                    st.dataframe(dist_df, width='stretch')
                    
                    # Plot AIC comparison
                    aic_data = [(d['Distribution'], float(d['AIC'])) for d in table_data if d['AIC'] != 'inf']
                    if aic_data:
                        aic_df = pd.DataFrame(aic_data, columns=['Distribution', 'AIC'])
                        aic_df = aic_df.sort_values('AIC')
                        fig_aic = px.bar(aic_df, x='Distribution', y='AIC', 
                                        title='Distribution Comparison (AIC - lower is better)',
                                        template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig_aic, width='stretch')
    
    # Display random variable analysis
    if 'rv_analysis' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['rv_analysis']
        if 'error' not in results:
            st.markdown("#### 📊 Random Variable Analysis")
            
            # Moments
            moments = results.get('moments', {})
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Mean", f"{moments.get('mean', 0):.4f}")
            mcol2.metric("Variance", f"{moments.get('variance', 0):.4f}")
            mcol3.metric("Std Dev", f"{moments.get('std', 0):.4f}")
            mcol4.metric("Skewness", f"{moments.get('skewness', 0):.4f}")
            mcol5.metric("Kurtosis", f"{moments.get('excess_kurtosis', 0):.4f}")
            
            # Quantiles
            with st.expander("📊 Quantiles & Percentiles"):
                quantiles = results.get('quantiles', {})
                qcol1, qcol2, qcol3 = st.columns(3)
                with qcol1:
                    st.write("**Lower Tail**")
                    st.write(f"1%: {quantiles.get('1%', 0):.4f}")
                    st.write(f"5%: {quantiles.get('5%', 0):.4f}")
                    st.write(f"10%: {quantiles.get('10%', 0):.4f}")
                with qcol2:
                    st.write("**Central**")
                    st.write(f"25% (Q1): {quantiles.get('25%', 0):.4f}")
                    st.write(f"50% (Median): {quantiles.get('50%', 0):.4f}")
                    st.write(f"75% (Q3): {quantiles.get('75%', 0):.4f}")
                with qcol3:
                    st.write("**Upper Tail**")
                    st.write(f"90%: {quantiles.get('90%', 0):.4f}")
                    st.write(f"95%: {quantiles.get('95%', 0):.4f}")
                    st.write(f"99%: {quantiles.get('99%', 0):.4f}")
            
            # Additional stats
            ci = results.get('confidence_interval_95', {})
            st.info(f"**95% CI for Mean:** [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]")
            st.write(f"**Coefficient of Variation:** {results.get('coefficient_of_variation', 0):.2f}%")
            st.write(f"**IQR:** {results.get('iqr', 0):.4f}")


