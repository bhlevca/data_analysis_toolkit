"""
Multivariate Ordination Tab for the Data Analysis Toolkit

Provides PCoA, NMDS, CA, DCA, CCA, and RDA ordination methods
with interactive Plotly visualizations and Mantel test.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from ordination import (
    pcoa, nmds, correspondence_analysis,
    detrended_correspondence_analysis,
    canonical_correspondence_analysis,
    redundancy_analysis, mantel_test,
)
from ecology import distance_matrix as eco_distance_matrix

PLOTLY_TEMPLATE = "plotly_white"


def render_ordination_tab():
    """Render the Multivariate Ordination tab."""
    st.header("🗺️ Multivariate Ordination")
    st.caption("PCoA, NMDS, CA, DCA, CCA, RDA, and Mantel test")

    if st.session_state.df is None:
        st.warning("📁 Please load data first in the Data tab.")
        return

    df = st.session_state.df

    with st.expander("ℹ️ About Ordination Methods", expanded=False):
        st.markdown("""
        ### Ordination Methods

        Ordination reduces multivariate data to a few interpretable axes:

        | Method | Type | Best for |
        |--------|------|----------|
        | **PCoA** | Unconstrained | Any distance metric → Euclidean embedding |
        | **NMDS** | Unconstrained | Preserves rank-order of distances |
        | **CA** | Unconstrained | Species × sites (unimodal response) |
        | **DCA** | Unconstrained | CA with arch-effect removal |
        | **CCA** | Constrained | Species ~ environment relationships |
        | **RDA** | Constrained | Linear response ~ environment |
        | **Mantel** | Association | Correlation between distance matrices |

        ---

        ### 📖 Choosing a Method

        ```
        Do you have environmental variables?
            │
            ├── No  → Unconstrained
            │       Is species response unimodal?
            │           ├── Yes → CA / DCA
            │           └── No  → PCoA / NMDS
            │
            └── Yes → Constrained
                    Is species response unimodal?
                        ├── Yes → CCA
                        └── No  → RDA
        ```

        **PCoA vs NMDS:**
        - PCoA is eigenvalue-based → faster, deterministic
        - NMDS is iterative → better for non-metric distances, check stress < 0.2

        ---

        ### 📋 Data Format

        - **Species matrix**: rows = sites, columns = species abundances
        - **Environmental matrix** (CCA/RDA): rows = sites, columns = environmental variables
        - **Distance matrix** (PCoA/NMDS/Mantel): square symmetric matrix of pairwise distances

        ---

        ### 📊 Interpreting Results

        - **Eigenvalues / % variance**: how much variation each axis explains
        - **Site scores**: position of each sample in reduced space
        - **Species scores** (CA/CCA): species optima along gradients
        - **Biplot arrows** (RDA/CCA): direction and strength of environmental effects
        - **Stress** (NMDS): < 0.05 excellent, < 0.1 good, < 0.2 acceptable, > 0.3 poor
        - **Mantel r**: correlation between distance matrices (−1 to 1); test with permutation p-value
        """)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    # ----- Method selection -----
    method = st.selectbox(
        "Ordination method:",
        ['PCoA', 'NMDS', 'CA', 'DCA', 'CCA', 'RDA', 'Mantel Test'],
        key="ord_method"
    )

    # ----- Column selection -----
    st.subheader("📋 Data Setup")

    if method in ('CCA', 'RDA'):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            response_cols = st.multiselect(
                "Response columns (species/variables):",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="ord_response_cols"
            )
        with col_b:
            env_cols = st.multiselect(
                "Explanatory columns (environment):",
                [c for c in numeric_cols if c not in response_cols],
                key="ord_env_cols"
            )
        with col_c:
            label_col = st.selectbox(
                "Label column (optional):",
                [None] + all_cols,
                key="ord_label_col"
            )
            group_col = st.selectbox(
                "Group column (for colouring):",
                [None] + all_cols,
                key="ord_group_col"
            )
    elif method == 'Mantel Test':
        col_a, col_b = st.columns(2)
        with col_a:
            set1_cols = st.multiselect(
                "Distance matrix 1 columns:",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="ord_set1_cols"
            )
        with col_b:
            set2_cols = st.multiselect(
                "Distance matrix 2 columns:",
                [c for c in numeric_cols if c not in set1_cols],
                key="ord_set2_cols"
            )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            response_cols = st.multiselect(
                "Data columns:",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))],
                key="ord_data_cols"
            )
        with col_b:
            label_col = st.selectbox(
                "Label column (optional):",
                [None] + all_cols,
                key="ord_label_col2"
            )
            group_col = st.selectbox(
                "Group column (for colouring):",
                [None] + all_cols,
                key="ord_group_col2"
            )

    # ----- Distance metric (for PCoA/NMDS) -----
    if method in ('PCoA', 'NMDS'):
        dist_metric = st.selectbox(
            "Distance metric:",
            ['bray_curtis', 'jaccard', 'sorensen', 'euclidean'],
            key="ord_dist_metric"
        )
    else:
        dist_metric = None

    st.markdown("---")

    # ============================
    # MANTEL TEST
    # ============================
    if method == 'Mantel Test':
        if not set1_cols or not set2_cols:
            st.info("Select columns for both distance matrices.")
            return

        mantel_method = st.selectbox("Correlation method:", ['pearson', 'spearman'], key="mantel_method")
        mantel_perms = st.slider("Permutations:", 99, 9999, 999, step=100, key="mantel_perms")

        if st.button("Run Mantel Test", type="primary"):
            with st.spinner("Computing..."):
                data1 = df[set1_cols].values.astype(float)
                data2 = df[set2_cols].values.astype(float)
                dm1 = eco_distance_matrix(data1, metric='bray_curtis')
                dm2 = eco_distance_matrix(data2, metric='bray_curtis')
                result = mantel_test(dm1, dm2, method=mantel_method, permutations=mantel_perms)

            st.success(f"**Mantel r = {result['statistic']:.4f}**, p = {result['p_value']:.4f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Mantel r", f"{result['statistic']:.4f}")
            col2.metric("p-value", f"{result['p_value']:.4f}")
            col3.metric("Permutations", result['n_permutations'])

            st.markdown(f"**Method**: {result['method'].title()} correlation")
            if result['p_value'] < 0.05:
                st.success("✅ Significant correlation between the two distance matrices (p < 0.05)")
            else:
                st.info("No significant correlation at α = 0.05")
        return

    # ============================
    # ORDINATION METHODS
    # ============================
    if method in ('CCA', 'RDA'):
        if not response_cols or not env_cols:
            st.info("Select response AND explanatory columns.")
            return
        cols_for_labels = response_cols
    else:
        if not response_cols or len(response_cols) < 2:
            st.info("Select at least 2 data columns.")
            return
        cols_for_labels = response_cols

    # Labels and groups
    if method in ('CCA', 'RDA'):
        lbl_col = label_col
        grp_col = group_col
    else:
        lbl_col = label_col if 'label_col' in dir() else None
        grp_col = group_col if 'group_col' in dir() else None

    # Resolve variable references from different column layouts
    try:
        lbl_col_name = lbl_col
    except NameError:
        lbl_col_name = None
    try:
        grp_col_name = grp_col
    except NameError:
        grp_col_name = None

    if lbl_col_name and lbl_col_name in df.columns:
        labels = df[lbl_col_name].astype(str).tolist()
    else:
        labels = [f"Sample_{i}" for i in range(len(df))]

    groups = df[grp_col_name].astype(str).tolist() if grp_col_name and grp_col_name in df.columns else None

    if st.button("🔄 Run Ordination", type="primary", key="run_ordination"):
        with st.spinner(f"Running {method}..."):
            data = df[response_cols].dropna().values.astype(float)

            try:
                if method == 'PCoA':
                    if dist_metric == 'euclidean':
                        from scipy.spatial.distance import pdist, squareform
                        dm = squareform(pdist(data, metric='euclidean'))
                    else:
                        dm = eco_distance_matrix(data, metric=dist_metric)
                    result = pcoa(dm, n_components=min(3, data.shape[0] - 1))
                    axis_label = "PCo"
                    variance = result['explained_variance']

                elif method == 'NMDS':
                    if dist_metric == 'euclidean':
                        from scipy.spatial.distance import pdist, squareform
                        dm = squareform(pdist(data, metric='euclidean'))
                    else:
                        dm = eco_distance_matrix(data, metric=dist_metric)
                    result = nmds(dm, n_components=2)
                    axis_label = "NMDS"
                    variance = None

                elif method == 'CA':
                    result = correspondence_analysis(data, n_components=min(3, min(data.shape) - 1))
                    axis_label = "CA"
                    variance = result['explained_inertia']

                elif method == 'DCA':
                    result = detrended_correspondence_analysis(data, n_components=2)
                    axis_label = "DCA"
                    variance = None

                elif method == 'CCA':
                    env_data = df[env_cols].dropna().values.astype(float)
                    result = canonical_correspondence_analysis(data, env_data, n_components=2)
                    axis_label = "CCA"
                    variance = result['explained_inertia']

                elif method == 'RDA':
                    env_data = df[env_cols].dropna().values.astype(float)
                    result = redundancy_analysis(data, env_data, n_components=2)
                    axis_label = "RDA"
                    variance = result['explained_variance']

            except Exception as e:
                st.error(f"Error: {e}")
                return

        # ----- Plot results -----
        if method == 'NMDS':
            coords = result['coordinates']
            stress = result['stress']
            st.metric("Stress", f"{stress:.4f}")
            if stress < 0.05:
                st.success("Excellent fit (stress < 0.05)")
            elif stress < 0.1:
                st.success("Good fit (stress < 0.1)")
            elif stress < 0.2:
                st.warning("Fair fit (stress < 0.2)")
            else:
                st.error("Poor fit (stress ≥ 0.2) — consider more dimensions")
        else:
            coords = result.get('coordinates', result.get('row_scores', result.get('site_scores')))

        n_pts = min(len(labels), coords.shape[0])
        plot_labels = labels[:n_pts]

        # Build axes labels
        ax1_label = f"{axis_label}1"
        ax2_label = f"{axis_label}2"
        if variance is not None and len(variance) >= 2:
            ax1_label += f" ({variance[0]*100:.1f}%)"
            ax2_label += f" ({variance[1]*100:.1f}%)"

        plot_df = pd.DataFrame({
            ax1_label: coords[:n_pts, 0],
            ax2_label: coords[:n_pts, 1],
            'Label': plot_labels,
        })
        if groups:
            plot_df['Group'] = groups[:n_pts]
            fig = px.scatter(
                plot_df, x=ax1_label, y=ax2_label,
                color='Group', text='Label',
                title=f"{method} Ordination",
                template=PLOTLY_TEMPLATE
            )
        else:
            fig = px.scatter(
                plot_df, x=ax1_label, y=ax2_label,
                text='Label',
                title=f"{method} Ordination",
                template=PLOTLY_TEMPLATE
            )

        fig.update_traces(textposition='top center', textfont_size=9)

        # Add biplot arrows for CCA/RDA
        if method in ('CCA', 'RDA') and 'biplot_scores' in result:
            biplot = result['biplot_scores']
            # Scale arrows
            max_coord = np.abs(coords[:n_pts, :2]).max()
            for j, col_name in enumerate(env_cols):
                if j < biplot.shape[0]:
                    fig.add_annotation(
                        x=biplot[j, 0] * max_coord * 0.8,
                        y=biplot[j, 1] * max_coord * 0.8,
                        ax=0, ay=0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2, arrowsize=1.5,
                        arrowcolor='red',
                        text=col_name,
                        font=dict(color='red', size=10)
                    )

        # Add species scores for CA/CCA/DCA
        if method in ('CA', 'DCA', 'CCA') and 'col_scores' in result:
            sp_scores = result.get('col_scores', result.get('species_scores'))
            if sp_scores is not None:
                n_sp = min(sp_scores.shape[0], len(response_cols))
                for j in range(n_sp):
                    fig.add_trace(go.Scatter(
                        x=[sp_scores[j, 0]], y=[sp_scores[j, 1]],
                        mode='markers+text',
                        marker=dict(symbol='cross', size=8, color='gray'),
                        text=[response_cols[j]],
                        textposition='bottom center',
                        textfont=dict(size=8, color='gray'),
                        name=response_cols[j],
                        showlegend=False
                    ))

        fig.update_layout(
            height=600,
            xaxis=dict(zeroline=True, zerolinecolor='lightgray'),
            yaxis=dict(zeroline=True, zerolinecolor='lightgray'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----- Summary statistics -----
        if method in ('PCoA', 'CA', 'CCA', 'RDA') and variance is not None:
            st.subheader("Eigenvalues & Explained Variance")
            eig_df = pd.DataFrame({
                'Axis': [f"{axis_label}{i+1}" for i in range(len(variance))],
                'Eigenvalue': result['eigenvalues'][:len(variance)],
                'Explained (%)': variance * 100,
                'Cumulative (%)': np.cumsum(variance) * 100,
            })
            st.dataframe(eig_df.style.format({'Eigenvalue': '{:.4f}', 'Explained (%)': '{:.1f}', 'Cumulative (%)': '{:.1f}'}))

            # Scree plot
            fig_scree = px.bar(
                eig_df, x='Axis', y='Explained (%)',
                title='Scree Plot',
                template=PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        if method in ('CCA', 'RDA'):
            if method == 'CCA':
                st.metric("Total Inertia", f"{result['total_inertia']:.4f}")
                st.metric("Constrained Inertia", f"{result['constrained_inertia']:.4f}")
                st.metric("% Explained", f"{result['constrained_inertia']/result['total_inertia']*100:.1f}%")
            else:
                st.metric("R²", f"{result['r_squared']:.4f}")
                st.metric("Constrained Variance", f"{result['constrained_variance']:.4f}")
                st.metric("Total Variance", f"{result['total_variance']:.4f}")

        # Coordinates table
        with st.expander("📄 Ordination Scores"):
            score_df = pd.DataFrame(
                coords[:n_pts, :2],
                columns=[f"{axis_label}1", f"{axis_label}2"],
                index=plot_labels
            )
            st.dataframe(score_df.style.format('{:.4f}'))
