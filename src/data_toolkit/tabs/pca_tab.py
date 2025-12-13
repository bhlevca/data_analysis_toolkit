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
from pca_visualization import create_pca_biplot_with_vectors, generate_pca_insights, interpret_vectors

PLOTLY_TEMPLATE = "plotly_white"

def render_pca_tab():
    """Render PCA analysis tab with comprehensive visualizations"""
    st.header("🔬 PCA (Principal Component Analysis)")

    # Educational introduction
    with st.expander("ℹ️ About PCA - The Foundation of Multivariate Data Analysis", expanded=False):
        st.markdown("""
        **PCA is the mother method for Multivariate Data Analysis (MVDA)**

        PCA finds **lines, planes, and hyper-planes** in K-dimensional space that best approximate
        the data in a least-squares sense. As Pearson described it: *"finding lines and planes of
        closest fit to systems of points in space"*.

        **Key Concepts:**
        - **Scores**: The coordinates of observations projected onto the principal component plane
        - **Loadings**: The weights showing how each original variable contributes to each PC
        - **Biplot**: Combines scores and loadings to reveal relationships between observations AND variables
        - **Explained Variance**: How much of the total data variation each PC captures

        **What PCA reveals:**
        - 📊 **Trends & Patterns**: Similar observations cluster together
        - 🎯 **Outliers**: Points far from the main cluster
        - 🔗 **Variable Relationships**: Variables pointing in similar directions are correlated
        - 📉 **Data Structure**: The main "directions" of variation in your data
        """)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    # Settings
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        variance_threshold = st.slider("Variance threshold for component selection", 0.80, 0.99, 0.95,
                                       help="Select components that explain at least this much variance")
    with col_set2:
        scale_loadings = st.slider("Loading vector scale", 1.0, 5.0, 2.5,
                                   help="Scale factor for loading vectors in biplot")

    if st.button("🔬 Run PCA Analysis", use_container_width=True):
        with st.spinner("Running PCA..."):
            results = ml.pca_analysis(features, variance_threshold=variance_threshold)
            st.session_state.analysis_results['pca'] = results

    st.markdown("---")

    if 'pca' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pca']

        if 'error' in results:
            st.error(results['error'])
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Components Selected", results['n_components_selected'],
                         help="Number of PCs needed to explain the variance threshold")
            with col2:
                total_var = results['total_variance_explained']
                st.metric("Variance Explained", f"{total_var*100:.1f}%")
            with col3:
                st.metric("Original Dimensions", len(features))

            # Get data for plotting
            explained_var = results['explained_variance']
            cumsum_var = results['cumulative_variance']
            components = results['components']
            feature_names = results['feature_names']
            scores = results['transformed_data']
            n_components = min(results['n_components_selected'], scores.shape[1])

            # ═══════════════════════════════════════════════════════════════
            # SCREE PLOT & CUMULATIVE VARIANCE
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📊 Variance Explained")

            fig_scree = make_subplots(rows=1, cols=2,
                                      subplot_titles=('Scree Plot (Individual Variance)',
                                                     'Cumulative Variance Explained'))

            # Individual variance bars
            fig_scree.add_trace(
                go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))],
                       y=explained_var * 100,
                       name='Individual',
                       marker_color='steelblue',
                       text=[f'{v*100:.1f}%' for v in explained_var],
                       textposition='outside'),
                row=1, col=1
            )

            # Cumulative line
            fig_scree.add_trace(
                go.Scatter(x=[f'PC{i+1}' for i in range(len(cumsum_var))],
                          y=cumsum_var * 100,
                          mode='lines+markers',
                          name='Cumulative',
                          marker=dict(size=10, color='darkorange'),
                          line=dict(width=3, color='darkorange')),
                row=1, col=2
            )
            # 95% threshold line
            fig_scree.add_hline(y=95, line_dash='dash', line_color='red',
                               annotation_text='95% threshold', row=1, col=2)

            fig_scree.update_layout(height=400, template=PLOTLY_TEMPLATE, showlegend=False)
            fig_scree.update_yaxes(title_text='Variance (%)', row=1, col=1)
            fig_scree.update_yaxes(title_text='Cumulative Variance (%)', row=1, col=2)
            st.plotly_chart(fig_scree, use_container_width=True)

            # ═══════════════════════════════════════════════════════════════
            # 2D SCORE PLOT WITH LOADING VECTORS
            # ═══════════════════════════════════════════════════════════════
            st.subheader("🎯 Score Plot with Loading Vectors")
            st.caption("**Dots**: Observations projected onto PC1-PC2 plane. **Arrows**: How each original variable contributes to the PCs (loading vectors). "
                      "Arrows pointing same direction = correlated variables. Color shows observation's PC1 score.")

            if scores.shape[1] >= 2:
                fig_scores = go.Figure()

                # Scatter plot of scores with meaningful colors
                fig_scores.add_trace(go.Scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(scores)),  # Color by observation index
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Observation #', x=1.02)
                    ),
                    text=[f'Obs {i+1}' for i in range(len(scores))],
                    hovertemplate='<b>Observation %{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                    name='Observations'
                ))

                # Add reference lines at origin
                fig_scores.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig_scores.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.5)

                # Add LOADING VECTORS for original variables
                # Scale loadings to fit the score space
                max_score = max(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1])))
                max_loading = np.max(np.abs(components[:2, :]))
                load_scale = max_score * 0.8 / max_loading if max_loading > 0 else 1

                colors = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Loading vector: how this variable projects onto PC1 and PC2
                    load_x = components[0, i] * load_scale  # PC1 loading
                    load_y = components[1, i] * load_scale  # PC2 loading
                    color = colors[i % len(colors)]

                    # Arrow line
                    fig_scores.add_trace(go.Scatter(
                        x=[0, load_x],
                        y=[0, load_y],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'{feature}',
                        hoverinfo='name',
                        showlegend=True
                    ))

                    # Arrowhead
                    fig_scores.add_annotation(
                        x=load_x, y=load_y,
                        ax=0, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=color
                    )

                    # Label at arrow tip
                    fig_scores.add_annotation(
                        x=load_x * 1.1, y=load_y * 1.1,
                        text=feature,
                        showarrow=False,
                        font=dict(size=11, color=color, family='Arial Black')
                    )

                var1 = explained_var[0] * 100
                var2 = explained_var[1] * 100
                fig_scores.update_layout(
                    title=f'Score Plot with Loading Vectors',
                    xaxis_title=f'PC1 ({var1:.1f}% variance explained)',
                    yaxis_title=f'PC2 ({var2:.1f}% variance explained)',
                    template=PLOTLY_TEMPLATE,
                    height=600,
                    xaxis=dict(scaleanchor='y', scaleratio=1),
                    legend=dict(title='Original Variables', yanchor='top', y=0.99, xanchor='left', x=1.15)
                )
                st.plotly_chart(fig_scores, use_container_width=True)

                with st.expander("📖 How to interpret this plot"):
                    st.markdown("""
                    **This is a combined Score Plot + Loading Vectors visualization:**

                    - **Dots (Observations)**: Each dot is a data point projected onto PC1-PC2 space
                      - Color indicates observation number (low=purple, high=yellow)
                      - Dots close together = similar observations
                      - Dots far from origin = extreme values

                    - **Arrows (Loading Vectors)**: Show how original variables relate to PCs
                      - Arrow direction: which PC direction this variable aligns with
                      - Arrow length: strength of contribution to these PCs
                      - **Arrows pointing same way**: positively correlated variables
                      - **Arrows pointing opposite**: negatively correlated variables
                      - **Arrows at 90°**: uncorrelated variables

                    **Reading tip**: Project a dot perpendicularly onto an arrow to see
                    if that observation has high/low values for that variable.
                    """)

            # ═══════════════════════════════════════════════════════════════
            # BIPLOT - Scores + Loading Vectors
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📐 Biplot: Observations AND Variables")
            st.caption("Combines score plot with loading vectors. Arrows show how each original variable "
                      "contributes to the principal components. Variables pointing in similar directions are correlated.")

            if scores.shape[1] >= 2 and len(feature_names) > 0:
                fig_biplot = go.Figure()

                # Scale scores to fit with loadings
                score_scale = np.max(np.abs(components[:2, :])) * scale_loadings
                scores_scaled = scores[:, :2] / (np.max(np.abs(scores[:, :2])) / score_scale)

                # Plot scores (observations)
                fig_biplot.add_trace(go.Scatter(
                    x=scores_scaled[:, 0],
                    y=scores_scaled[:, 1],
                    mode='markers',
                    marker=dict(size=6, color='steelblue', opacity=0.6),
                    name='Observations',
                    hovertemplate='Obs %{text}<extra></extra>',
                    text=[str(i+1) for i in range(len(scores))]
                ))

                # Plot loading vectors (variables)
                colors = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Arrow from origin to loading
                    loading_x = components[0, i] * scale_loadings
                    loading_y = components[1, i] * scale_loadings

                    color = colors[i % len(colors)]

                    # Arrow line
                    fig_biplot.add_trace(go.Scatter(
                        x=[0, loading_x],
                        y=[0, loading_y],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Arrowhead (using annotation)
                    fig_biplot.add_annotation(
                        x=loading_x, y=loading_y,
                        ax=0, ay=0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=color
                    )

                    # Label
                    fig_biplot.add_trace(go.Scatter(
                        x=[loading_x * 1.15],
                        y=[loading_y * 1.15],
                        mode='text',
                        text=[feature],
                        textfont=dict(size=12, color=color, family='Arial Black'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Reference lines
                fig_biplot.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig_biplot.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.5)

                fig_biplot.update_layout(
                    title='Biplot: Scores (points) + Loadings (vectors)',
                    xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
                    yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
                    template=PLOTLY_TEMPLATE,
                    height=600,
                    xaxis=dict(scaleanchor='y', scaleratio=1)
                )
                st.plotly_chart(fig_biplot, use_container_width=True)

                # Interpretation help
                with st.expander("📖 How to interpret the Biplot"):
                    st.markdown("""
                    **Reading the Biplot:**

                    1. **Observation Points (blue dots)**:
                       - Close points = similar observations
                       - Far from center = extreme/unusual observations

                    2. **Loading Vectors (colored arrows)**:
                       - Arrow length = importance of variable for these PCs
                       - Arrow direction = how the variable relates to the PCs
                       - **Arrows pointing same direction** = positively correlated variables
                       - **Arrows pointing opposite directions** = negatively correlated variables
                       - **Arrows at 90°** = uncorrelated variables

                    3. **Projecting observations onto vectors**:
                       - Drop a perpendicular from an observation to a variable vector
                       - Where it lands indicates the observation's value for that variable
                    """)

            # ═══════════════════════════════════════════════════════════════
            # 3D SCORE PLOT WITH LOADING VECTORS (if 3+ components)
            # ═══════════════════════════════════════════════════════════════
            if scores.shape[1] >= 3:
                st.subheader("🌐 3D Score Plot with Loading Vectors")
                st.caption("**Dots**: Observations in PC1-PC2-PC3 space. **Arrows**: Loading vectors showing how each original variable projects into this 3D space.")

                fig_3d = go.Figure()

                # Add data points with color by observation index
                fig_3d.add_trace(go.Scatter3d(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    z=scores[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=np.arange(len(scores)),
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='Obs #', x=1.02)
                    ),
                    text=[f'Obs {i+1}' for i in range(len(scores))],
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
                    name='Observations'
                ))

                # Add LOADING VECTORS for original variables (scaled to score space)
                max_score_3d = max(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1])), np.max(np.abs(scores[:, 2])))
                max_loading_3d = np.max(np.abs(components[:3, :])) if components.shape[0] >= 3 else 1
                load_scale_3d = max_score_3d * 0.7 / max_loading_3d if max_loading_3d > 0 else 1

                colors_3d = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Loading vector in 3D: how this variable projects onto PC1, PC2, PC3
                    load_x = components[0, i] * load_scale_3d
                    load_y = components[1, i] * load_scale_3d
                    load_z = components[2, i] * load_scale_3d if components.shape[0] >= 3 else 0
                    color = colors_3d[i % len(colors_3d)]

                    # Arrow line
                    fig_3d.add_trace(go.Scatter3d(
                        x=[0, load_x], y=[0, load_y], z=[0, load_z],
                        mode='lines+text',
                        line=dict(color=color, width=5),
                        text=['', feature],
                        textposition='top center',
                        textfont=dict(size=10, color=color),
                        name=feature,
                        hoverinfo='name',
                        showlegend=True
                    ))

                    # Cone arrowhead
                    length = np.sqrt(load_x**2 + load_y**2 + load_z**2)
                    if length > 0:
                        fig_3d.add_trace(go.Cone(
                            x=[load_x], y=[load_y], z=[load_z],
                            u=[load_x/length * 0.2], v=[load_y/length * 0.2], w=[load_z/length * 0.2],
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            sizemode='absolute',
                            sizeref=0.12,
                            hoverinfo='skip',
                            showlegend=False
                        ))

                # Add origin marker
                fig_3d.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=6, color='black', symbol='diamond'),
                    name='Origin',
                    hoverinfo='name',
                    showlegend=True
                ))

                fig_3d.update_layout(
                    title='3D Score Plot with Variable Loading Vectors',
                    scene=dict(
                        xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
                        yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
                        zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
                        aspectmode='data'
                    ),
                    template=PLOTLY_TEMPLATE,
                    height=700,
                    legend=dict(
                        title='Original Variables',
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
                st.plotly_chart(fig_3d, use_container_width=True)

                with st.expander("📖 Understanding the 3D Score Plot"):
                    st.markdown("""
                    **This is a 3D version of the Score Plot + Loading Vectors:**

                    - **Dots (Observations)**: Data points projected onto PC1-PC2-PC3 space
                      - Color indicates observation number
                      - Clusters = groups of similar observations

                    - **Arrows (Loading Vectors)**: Original variables projected onto PC space
                      - Direction shows how variable relates to PC1, PC2, PC3
                      - Length shows strength of contribution
                      - Variables pointing same direction are correlated

                    **Tip**: Rotate the 3D plot to explore the data structure from different angles.
                    """)

            # ═══════════════════════════════════════════════════════════════
            # LOADING MATRIX HEATMAP
            # ═══════════════════════════════════════════════════════════════
            st.subheader("🔥 Loading Matrix Heatmap")
            st.caption("How each original variable contributes to each principal component. "
                      "Red = positive loading, Blue = negative loading.")

            n_show = min(5, components.shape[0])
            loadings_df = pd.DataFrame(
                components[:n_show].T,
                index=feature_names,
                columns=[f'PC{i+1}' for i in range(n_show)]
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=loadings_df.values,
                x=loadings_df.columns,
                y=loadings_df.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(loadings_df.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Loading')
            ))

            fig_heat.update_layout(
                title='Variable Loadings on Principal Components',
                xaxis_title='Principal Component',
                yaxis_title='Original Variable',
                template=PLOTLY_TEMPLATE,
                height=max(300, len(feature_names) * 30)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Loadings table
            with st.expander("📋 Detailed Loading Values"):
                st.dataframe(loadings_df.round(4), use_container_width=True)

            # ═══════════════════════════════════════════════════════════════
            # EXPORT PCA RESULTS
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📥 Export PCA Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Export scores (transformed data)
                scores_df = pd.DataFrame(
                    scores[:, :n_show],
                    columns=[f'PC{i+1}' for i in range(n_show)]
                )
                csv_scores = scores_df.to_csv(index=True)
                st.download_button(
                    label="📥 PC Scores (CSV)",
                    data=csv_scores,
                    file_name="pca_scores.csv",
                    mime="text/csv"
                )

            with col2:
                # Export loadings
                csv_loadings = loadings_df.to_csv(index=True)
                st.download_button(
                    label="📥 Loadings Matrix (CSV)",
                    data=csv_loadings,
                    file_name="pca_loadings.csv",
                    mime="text/csv"
                )

            with col3:
                # Export variance explained
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                    'Variance_Explained': explained_var,
                    'Cumulative_Variance': cumsum_var
                })
                csv_var = var_df.to_csv(index=False)
                st.download_button(
                    label="📥 Variance Explained (CSV)",
                    data=csv_var,
                    file_name="pca_variance.csv",
                    mime="text/csv"
                )


