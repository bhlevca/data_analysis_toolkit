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

from ml_models import MLModels

PLOTLY_TEMPLATE = "plotly_white"

def render_dimreduction_tab():
    """Render dimensionality reduction tab"""
    st.header("📉 Dimensionality Reduction")
    st.caption("PCA, SVD, t-SNE (t-distributed Stochastic Neighbor Embedding), UMAP, and ICA (Independent Component Analysis)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Reduction Method",
            ["PCA", "t-SNE", "UMAP", "SVD", "ICA"]
        )

    with col2:
        max_comp = max(2, min(10, len(features)))
        if max_comp <= 2:
            max_comp = 3  # Ensure slider min < max
        n_components = st.slider("Components", 2, max_comp, 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        if method == "PCA":
            if st.button("🔬 PCA", use_container_width=True):
                with st.spinner("Computing PCA..."):
                    results = ml.pca_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['pca_new'] = results

    with col2:
        if method == "t-SNE":
            if st.button("📊 t-SNE", use_container_width=True):
                with st.spinner("Computing t-SNE..."):
                    results = ml.tsne_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['tsne'] = results

    with col3:
        if method == "UMAP":
            if st.button("🔷 UMAP", use_container_width=True):
                with st.spinner("Computing UMAP..."):
                    results = ml.umap_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['umap'] = results
    # Add SVD and ICA buttons below
    col4, col5 = st.columns(2)
    with col4:
        if method == "SVD":
            if st.button("📐 SVD", use_container_width=True):
                with st.spinner("Computing SVD..."):
                    results = ml.svd_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['svd'] = results

    with col5:
        if method == "ICA":
            if st.button("🔀 ICA", use_container_width=True):
                with st.spinner("Computing ICA..."):
                    results = ml.ica_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['ica'] = results

    st.markdown("---")

    if 'pca_new' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pca_new']
        if 'error' not in results:
            st.subheader("🔬 PCA Results with Feature Vectors")
            total_var = results.get('total_variance_explained', 0)
            st.metric("Variance Explained (Total)", f"{total_var*100:.1f}%")

            explained_var = results.get('explained_variance', [])
            if len(explained_var) > 0:
                fig = px.bar(x=[f'PC{i+1}' for i in range(len(explained_var))], y=explained_var,
                            title='Explained Variance per Component', template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)

            # Enhanced Cartesian biplot with feature vectors
            transformed = results.get('transformed_data')
            components = results.get('components')
            feature_names = results.get('feature_names', [])

            if (transformed is not None and hasattr(transformed, 'shape') and transformed.shape[1] >= 2
                and components is not None and feature_names):

                # Create biplot with vectors
                # Note: PCA components are (n_components, n_features), need to transpose for biplot
                # which expects (n_features, n_components)
                try:
                    components_T = components.T  # Transpose: (n_features, n_components)
                    fig_biplot, vector_info = create_pca_biplot_with_vectors(
                        transformed[:, :2],
                        components_T[:, :2] if components_T.shape[1] >= 2 else components_T,
                        explained_var,
                        feature_names,
                        scale_factor=3.0
                    )
                    st.plotly_chart(fig_biplot, use_container_width=True)

                    # Display insights
                    st.markdown("### 📊 Vector Interpretation Guide")
                    insights = generate_pca_insights(vector_info, explained_var, total_var)
                    st.markdown(insights)

                    # Display detailed vector interpretation
                    with st.expander("🔍 Detailed Vector Analysis"):
                        vector_interp = interpret_vectors(vector_info, feature_names)

                        st.markdown("#### PC Drivers")
                        st.markdown(vector_interp['pc1_drivers'])
                        st.markdown(vector_interp['pc2_drivers'])

                        st.markdown("#### Feature Correlations (Based on Vector Angles)")
                        for corr in vector_interp['correlations']:
                            st.markdown(f"- {corr}")

                        st.markdown("#### Feature Importance")
                        st.markdown(vector_interp['feature_importance'])

                        st.markdown("#### How to Read Vectors")
                        st.markdown("""
                        - **Vector direction**: Shows how feature aligns with PC1/PC2
                        - **Vector length**: Magnitude of contribution (longer = stronger)
                        - **Parallel vectors**: Features are correlated
                        - **Perpendicular vectors**: Features are independent
                        - **Opposite vectors**: Features are negatively correlated
                        """)

                except Exception as e:
                    st.error(f"Error creating PCA biplot: {str(e)}")
                    # Fallback to simple scatter
                    fig_pc = px.scatter(x=transformed[:, 0], y=transformed[:, 1],
                                        title='PCA Projection (PC1 vs PC2)',
                                        labels={'x': 'PC1', 'y': 'PC2'},
                                        template=PLOTLY_TEMPLATE)
                    fig_pc.update_layout(height=500)
                    st.plotly_chart(fig_pc, use_container_width=True)

    if 'tsne' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['tsne']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("📊 t-SNE Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='t-SNE Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    if 'umap' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['umap']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("🔷 UMAP Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='UMAP Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    if 'svd' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['svd']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("📐 SVD Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='SVD Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    if 'ica' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ica']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("🔀 ICA Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='ICA Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)


