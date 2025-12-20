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

def render_clustering_tab():
    """Render clustering analysis tab"""
    st.header("🎯 Clustering Analysis")
    st.caption("K-Means, Hierarchical (Agglomerative), DBSCAN, and Gaussian Mixture Model (GMM) clustering")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    st.subheader("Clustering Methods Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox("Method", ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"])

    with col2:
        if method == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            n_init = st.slider("Initializations", 5, 20, 10)
        elif method == "Hierarchical":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        elif method == "DBSCAN":
            eps = st.slider("Eps", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 20, 5)
        elif method == "Gaussian Mixture":
            n_clusters = st.slider("Number of Components", 2, 10, 3)
            cov_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])

    with col3:
        if st.button("🎯 Run Clustering", width='stretch'):
            with st.spinner("Clustering..."):
                try:
                    if method == "K-Means":
                        results = ml.kmeans_clustering(features, n_clusters=n_clusters, n_init=n_init)
                    elif method == "Hierarchical":
                        results = ml.hierarchical_clustering(features, n_clusters=n_clusters, linkage_method=linkage)
                    elif method == "DBSCAN":
                        results = ml.dbscan_clustering(features, eps=eps, min_samples=min_samples)
                    elif method == "Gaussian Mixture":
                        results = ml.gaussian_mixture_model(features, n_components=n_clusters, covariance_type=cov_type)

                    st.session_state.analysis_results['clustering'] = results
                    st.success("✅ Clustering complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if 'clustering' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['clustering']

        if 'error' not in results:
            st.subheader("📊 Clustering Results")

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{results.get('silhouette_score', 0):.4f}")
            col2.metric("Davies-Bouldin", f"{results.get('davies_bouldin_score', 0):.4f}")
            col3.metric("Calinski-Harabasz", f"{results.get('calinski_harabasz_score', 0):.4f}")

            # Cluster visualization (prefer model's 2D projection if available)
            clusters = results.get('clusters', [])
            X_vis = results.get('X_vis')

            if X_vis is not None and hasattr(X_vis, 'shape') and X_vis.shape[1] >= 2:
                fig = px.scatter(x=X_vis[:, 0], y=X_vis[:, 1], color=clusters,
                                 title=f'{method} Clustering Results (2D projection)',
                                 labels={'x': 'Component 1', 'y': 'Component 2'},
                                 template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            elif len(features) >= 2:
                # Fall back to raw feature scatter
                x_data = df[features[0]]
                y_data = df[features[1] if len(features) > 1 else features[0]]

                fig = px.scatter(
                    x=x_data, y=y_data,
                    color=clusters,
                    title=f'{method} Clustering Results (feature space)',
                    labels={f'x': features[0], f'y': features[1] if len(features) > 1 else features[0]},
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

            # Export clustering results
            st.subheader("📥 Export Clustering Results")
            df_cluster_results = df[features].dropna().copy()
            df_cluster_results['Cluster'] = clusters[:len(df_cluster_results)]

            csv_clusters = df_cluster_results.to_csv(index=True)
            st.download_button(
                label="📥 Download Clustering Results (CSV)",
                data=csv_clusters,
                file_name="clustering_results.csv",
                mime="text/csv"
            )


