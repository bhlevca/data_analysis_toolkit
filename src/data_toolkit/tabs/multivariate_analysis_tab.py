"""
Multivariate Analysis Tab for the Data Analysis Toolkit

Provides PERMANOVA, ANOSIM, SIMPER, MANOVA, Hotelling T², and
Discriminant Analysis with interactive UI.
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

from multivariate_analysis import (
    permanova, anosim, simper, manova,
    hotelling_t2, discriminant_analysis,
)
from ecology import distance_matrix as eco_distance_matrix

PLOTLY_TEMPLATE = "plotly_white"


def render_multivariate_analysis_tab():
    """Render the Multivariate Tests tab."""
    st.header("📊 Multivariate Hypothesis Tests")
    st.caption("PERMANOVA, ANOSIM, SIMPER, MANOVA, Hotelling T², Discriminant Analysis")

    if st.session_state.df is None:
        st.warning("📁 Please load data first in the Data tab.")
        return

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    with st.expander("ℹ️ About Multivariate Tests", expanded=False):
        st.markdown("""
        ### Multivariate Hypothesis Tests

        | Test | Purpose | Assumptions |
        |------|---------|-------------|
        | **PERMANOVA** | Multivariate ANOVA on distance matrices | No distributional assumptions |
        | **ANOSIM** | Rank-based dissimilarity among groups | Ranked distances |
        | **SIMPER** | Species contributions to between-group dissimilarity | Bray-Curtis distances |
        | **MANOVA** | Parametric multivariate ANOVA | Multivariate normality |
        | **Hotelling T²** | Two-sample multivariate t-test | Multivariate normality |
        | **LDA / CVA** | Canonical variate analysis / discriminant analysis | Linear separability |

        ---

        ### 📖 Choosing a Test

        ```
        Are your data normally distributed (multivariate)?
            │
            ├── Yes (or large n)
            │       How many groups?
            │           ├── 2 → Hotelling T²
            │           └── 3+ → MANOVA
            │
            └── No / Unknown
                    ├── Test group differences → PERMANOVA
                    ├── Rank-based test        → ANOSIM
                    ├── Which species differ?   → SIMPER
                    └── Classify into groups    → LDA
        ```

        ---

        ### 📊 Interpreting Results

        | Statistic | Meaning |
        |-----------|---------|
        | **PERMANOVA F** | Larger = greater between-group vs within-group variance |
        | **ANOSIM R** | −1 to 1; R > 0 = groups differ; R ≈ 0 = no difference |
        | **SIMPER %** | Species contribution to total between-group dissimilarity |
        | **Wilks' Λ** (MANOVA) | 0–1; smaller = stronger group effect |
        | **Hotelling T²** | Multivariate analogue of t-statistic |
        | **LDA accuracy** | Correct classification rate (cross-validated) |

        **p-values**: All tests report permutation or F-distribution p-values.
        Reject H₀ (groups are identical) when p < 0.05.

        ---

        ### 📋 Data Format

        - **Response variables**: numeric columns (species abundances, measurements, etc.)
        - **Grouping variable**: categorical column identifying group membership
        - For SIMPER: abundance data preferred (not presence/absence)
        - Minimum 2 groups required; at least 5 observations per group recommended
        """)

    test_method = st.selectbox(
        "Test method:",
        ['PERMANOVA', 'ANOSIM', 'SIMPER', 'MANOVA', 'Hotelling T²', 'Discriminant Analysis (LDA)'],
        key="mvtest_method"
    )

    # ----- Column selection -----
    st.subheader("📋 Data Setup")
    col_a, col_b = st.columns(2)
    with col_a:
        data_cols = st.multiselect(
            "Response variables (numeric):",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="mvtest_data_cols"
        )
    with col_b:
        group_col = st.selectbox(
            "Grouping variable:",
            [None] + [c for c in all_cols if c not in data_cols],
            key="mvtest_group_col"
        )

    if not data_cols or not group_col:
        st.info("Select response variables AND a grouping variable.")
        return

    # Clean data
    keep_cols = data_cols + [group_col]
    df_clean = df[keep_cols].dropna()
    data = df_clean[data_cols].values.astype(float)
    groups_arr = df_clean[group_col].values
    unique_groups = np.unique(groups_arr)

    if len(unique_groups) < 2:
        st.error("Need at least 2 groups. Check your grouping variable.")
        return

    st.markdown("---")

    # ============================
    # PERMANOVA
    # ============================
    if test_method == 'PERMANOVA':
        dist_metric = st.selectbox("Distance metric:", ['bray_curtis', 'jaccard', 'euclidean'], key="perm_dist")
        n_perms = st.slider("Permutations:", 99, 9999, 999, step=100, key="perm_nperms")

        if st.button("Run PERMANOVA", type="primary"):
            with st.spinner("Computing PERMANOVA..."):
                if dist_metric == 'euclidean':
                    from scipy.spatial.distance import pdist, squareform
                    dm = squareform(pdist(data, metric='euclidean'))
                else:
                    dm = eco_distance_matrix(data, metric=dist_metric)
                result = permanova(dm, groups_arr, permutations=n_perms)

            col1, col2, col3 = st.columns(3)
            col1.metric("Pseudo-F", f"{result['F_statistic']:.4f}")
            col2.metric("p-value", f"{result['p_value']:.4f}")
            col3.metric("R²", f"{result['R2']:.4f}")

            if result['p_value'] < 0.05:
                st.success("✅ Significant difference among groups (p < 0.05)")
            else:
                st.info("No significant difference at α = 0.05")

            st.markdown(f"""
            **Results Summary:**
            - SS_between = {result['SS_between']:.4f}
            - SS_within = {result['SS_within']:.4f}
            - SS_total = {result['SS_total']:.4f}
            - Permutations: {result['n_permutations']}
            """)

    # ============================
    # ANOSIM
    # ============================
    elif test_method == 'ANOSIM':
        dist_metric = st.selectbox("Distance metric:", ['bray_curtis', 'jaccard', 'euclidean'], key="anosim_dist")
        n_perms = st.slider("Permutations:", 99, 9999, 999, step=100, key="anosim_nperms")

        if st.button("Run ANOSIM", type="primary"):
            with st.spinner("Computing ANOSIM..."):
                if dist_metric == 'euclidean':
                    from scipy.spatial.distance import pdist, squareform
                    dm = squareform(pdist(data, metric='euclidean'))
                else:
                    dm = eco_distance_matrix(data, metric=dist_metric)
                result = anosim(dm, groups_arr, permutations=n_perms)

            col1, col2, col3 = st.columns(3)
            col1.metric("R statistic", f"{result['R_statistic']:.4f}")
            col2.metric("p-value", f"{result['p_value']:.4f}")
            col3.metric("Permutations", result['n_permutations'])

            st.markdown("""
            **Interpreting R:**
            - R = 1: Complete separation between groups
            - R = 0: No separation (random)
            - R < 0: More dissimilarity within groups than between
            """)

            if result['p_value'] < 0.05:
                st.success("✅ Significant group separation (p < 0.05)")
            else:
                st.info("No significant separation at α = 0.05")

    # ============================
    # SIMPER
    # ============================
    elif test_method == 'SIMPER':
        if st.button("Run SIMPER Analysis", type="primary"):
            with st.spinner("Computing SIMPER..."):
                result = simper(data, groups_arr, feature_names=data_cols)

            for pair_key, pair_result in result.items():
                st.subheader(f"Groups: {pair_key}")
                col1, col2 = st.columns(2)
                col1.metric("Overall dissimilarity", f"{pair_result['overall_dissimilarity']:.4f}")

                contrib_df = pd.DataFrame(pair_result['contributions'])
                contrib_df = contrib_df.sort_values('contribution_pct', ascending=False)
                st.dataframe(
                    contrib_df.style.format({
                        'mean_contribution': '{:.4f}',
                        'sd': '{:.4f}',
                        'contribution_pct': '{:.1f}',
                        'cumulative_pct': '{:.1f}',
                    })
                )

                # Plot top contributors
                top_n = min(10, len(contrib_df))
                fig = px.bar(
                    contrib_df.head(top_n),
                    x='feature', y='contribution_pct',
                    title=f"Top species contributions ({pair_key})",
                    labels={'feature': 'Species', 'contribution_pct': 'Contribution (%)'},
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============================
    # MANOVA
    # ============================
    elif test_method == 'MANOVA':
        if len(unique_groups) < 2:
            st.error("Need at least 2 groups for MANOVA.")
            return

        if st.button("Run MANOVA", type="primary"):
            with st.spinner("Computing MANOVA..."):
                result = manova(data, groups_arr)

            if result is None:
                st.error("MANOVA computation failed. Check that data has more samples than variables per group.")
                return

            st.subheader("MANOVA Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Wilks' Lambda**")
                w = result['wilks_lambda']
                st.metric("Λ", f"{w['statistic']:.4f}")
                st.metric("F-approx", f"{w['F_approx']:.4f}")
                st.metric("p-value", f"{w['p_value']:.4f}")
                st.metric("df", f"({w['df1']:.0f}, {w['df2']:.0f})")

            with col2:
                st.markdown("**Pillai's Trace**")
                p = result['pillai_trace']
                st.metric("V", f"{p['statistic']:.4f}")
                st.metric("F-approx", f"{p['F_approx']:.4f}")
                st.metric("p-value", f"{p['p_value']:.4f}")

            if 'hotelling_lawley' in result:
                st.markdown("**Hotelling-Lawley Trace**")
                hl = result['hotelling_lawley']
                st.metric("T²₀", f"{hl['statistic']:.4f}")
                st.metric("F-approx", f"{hl['F_approx']:.4f}")
                st.metric("p-value", f"{hl['p_value']:.4f}")

            # Overall verdict
            p_val = result['wilks_lambda']['p_value']
            if p_val < 0.05:
                st.success("✅ Significant multivariate effect (Wilks' Λ p < 0.05)")
            else:
                st.info("No significant multivariate effect at α = 0.05")

    # ============================
    # HOTELLING T²
    # ============================
    elif test_method == 'Hotelling T²':
        if len(unique_groups) != 2:
            st.error(f"Hotelling T² requires exactly 2 groups. Found {len(unique_groups)}. "
                     f"Please filter your data or choose a different grouping variable.")
            return

        if st.button("Run Hotelling T²", type="primary"):
            with st.spinner("Computing Hotelling T²..."):
                idx1 = groups_arr == unique_groups[0]
                idx2 = groups_arr == unique_groups[1]
                result = hotelling_t2(data[idx1], data[idx2])

            col1, col2, col3 = st.columns(3)
            col1.metric("T²", f"{result['T2']:.4f}")
            col2.metric("F statistic", f"{result['F_statistic']:.4f}")
            col3.metric("p-value", f"{result['p_value']:.4f}")

            st.markdown(f"**df**: ({result['df1']:.0f}, {result['df2']:.0f})")
            st.markdown(f"**Groups**: {unique_groups[0]} (n={np.sum(idx1)}) vs {unique_groups[1]} (n={np.sum(idx2)})")

            if result['p_value'] < 0.05:
                st.success("✅ Significant difference between groups (p < 0.05)")
            else:
                st.info("No significant difference at α = 0.05")

    # ============================
    # DISCRIMINANT ANALYSIS
    # ============================
    elif test_method == 'Discriminant Analysis (LDA)':
        n_comp = st.slider("Number of components:", 1, min(len(unique_groups) - 1, len(data_cols)), 
                           min(2, len(unique_groups) - 1, len(data_cols)), key="lda_ncomp")

        if st.button("Run Discriminant Analysis", type="primary"):
            with st.spinner("Computing LDA..."):
                result = discriminant_analysis(data, groups_arr, n_components=n_comp)

            if result is None:
                st.error("LDA failed. Check data dimensions and group structure.")
                return

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training accuracy", f"{result['accuracy']:.1%}")
            with col2:
                st.metric("N components", result['n_components'])

            # Explained variance
            if result['explained_variance_ratio'] is not None:
                ev = result['explained_variance_ratio']
                ev_df = pd.DataFrame({
                    'Component': [f"LD{i+1}" for i in range(len(ev))],
                    'Explained (%)': ev * 100,
                    'Cumulative (%)': np.cumsum(ev) * 100,
                })
                st.dataframe(ev_df.style.format({'Explained (%)': '{:.1f}', 'Cumulative (%)': '{:.1f}'}))

            # 2D plot
            coords = result['coordinates']
            if coords.shape[1] >= 2:
                ev = result['explained_variance_ratio']
                ax1 = f"LD1 ({ev[0]*100:.1f}%)" if ev is not None else "LD1"
                ax2 = f"LD2 ({ev[1]*100:.1f}%)" if ev is not None and len(ev) > 1 else "LD2"

                plot_df = pd.DataFrame({
                    ax1: coords[:, 0],
                    ax2: coords[:, 1],
                    'Group': groups_arr,
                })
                fig = px.scatter(
                    plot_df, x=ax1, y=ax2, color='Group',
                    title="Linear Discriminant Analysis",
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            elif coords.shape[1] == 1:
                plot_df = pd.DataFrame({
                    'LD1': coords[:, 0],
                    'Group': groups_arr,
                })
                fig = px.histogram(
                    plot_df, x='LD1', color='Group',
                    barmode='overlay', opacity=0.6,
                    title="Linear Discriminant Analysis (1D)",
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig, use_container_width=True)

            # Classification report
            if result['classification_report'] is not None:
                st.subheader("Classification Report")
                st.text(result['classification_report'])

            # Confusion matrix
            if result['confusion_matrix'] is not None:
                st.subheader("Confusion Matrix")
                cm = result['confusion_matrix']
                cm_df = pd.DataFrame(cm, index=unique_groups, columns=unique_groups)
                fig_cm = px.imshow(
                    cm_df, text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix",
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig_cm, use_container_width=True)
