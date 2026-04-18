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

            # Dendrogram for Hierarchical clustering
            linkage_matrix = results.get('linkage_matrix')
            if linkage_matrix is not None:
                st.subheader("🌳 Dendrogram")
                from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

                n_clust = results.get('n_clusters', 3)
                n_samples = len(linkage_matrix) + 1
                color_thresh = (
                    float(linkage_matrix[-(n_clust - 1), 2])
                    if len(linkage_matrix) >= n_clust else None
                )

                # Build sample labels from the dataframe index
                df_clean = df[features].dropna()
                raw_labels = [str(v) for v in df_clean.index[:n_samples]]

                # Use scipy dendrogram to get the layout data
                dendro_data = scipy_dendrogram(
                    linkage_matrix,
                    labels=raw_labels,
                    no_plot=True,
                    color_threshold=color_thresh,
                )

                # Map scipy colour keys to plotly-friendly colours
                _color_map = {
                    'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c',
                    'C3': '#d62728', 'C4': '#9467bd', 'C5': '#8c564b',
                    'C6': '#e377c2', 'C7': '#7f7f7f', 'C8': '#bcbd22',
                    'C9': '#17becf', 'b': '#1f77b4', 'g': '#2ca02c',
                    'r': '#d62728', 'c': '#17becf', 'm': '#9467bd',
                    'y': '#bcbd22', 'k': '#000000',
                }

                fig_dendro = go.Figure()

                for xk, yk, ck in zip(
                    dendro_data['icoord'],
                    dendro_data['dcoord'],
                    dendro_data['color_list'],
                ):
                    fig_dendro.add_trace(go.Scatter(
                        x=xk, y=yk,
                        mode='lines',
                        line=dict(color=_color_map.get(ck, ck), width=2),
                        hoverinfo='y',
                        showlegend=False,
                    ))

                # Cut-line for chosen number of clusters
                if color_thresh is not None:
                    fig_dendro.add_hline(
                        y=color_thresh, line_dash='dash', line_color='red',
                        annotation_text=f"Cut for {n_clust} clusters",
                        annotation_position='top left',
                    )

                # X-axis tick labels (sample names) — thin out if too many
                tick_positions = list(range(5, 10 * len(dendro_data['ivl']), 10))
                ivl = dendro_data['ivl']
                if len(ivl) > 50:
                    # Show every Nth label so they stay readable
                    step = max(1, len(ivl) // 30)
                    shown_pos = tick_positions[::step]
                    shown_lbl = [ivl[i] for i in range(0, len(ivl), step)]
                else:
                    shown_pos = tick_positions
                    shown_lbl = ivl

                fig_dendro.update_layout(
                    title='Hierarchical Clustering Dendrogram',
                    xaxis=dict(
                        title='Samples',
                        tickvals=shown_pos,
                        ticktext=shown_lbl,
                        tickangle=90,
                        tickfont=dict(size=max(7, min(11, 500 // max(len(shown_lbl), 1)))),
                    ),
                    yaxis=dict(title='Distance'),
                    template=PLOTLY_TEMPLATE,
                    height=550,
                )

                st.plotly_chart(fig_dendro, use_container_width=True)

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


