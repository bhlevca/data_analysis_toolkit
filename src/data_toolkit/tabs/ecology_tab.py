"""
Ecology & Community Analysis Tab for the Data Analysis Toolkit

Provides alpha/beta diversity, rarefaction, species accumulation,
and SHE analysis via an interactive Streamlit interface.
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

from ecology import (
    all_alpha_diversity, distance_matrix, whittaker_beta,
    rarefaction_curve, species_accumulation, she_analysis,
    analyze_community_dataframe,
)

PLOTLY_TEMPLATE = "plotly_white"


def render_ecology_tab():
    """Render the Ecology & Community Analysis tab."""
    st.header("🌿 Ecology & Community Analysis")
    st.caption("Alpha/beta diversity, rarefaction, species accumulation curves, and SHE analysis")

    if st.session_state.df is None:
        st.warning("📁 Please load data first in the Data tab.")
        return

    df = st.session_state.df

    with st.expander("ℹ️ About Community Ecology Analysis", expanded=False):
        st.markdown("""
        ### Community Ecology Methods

        This module analyses species × sites (samples) data:

        **Alpha Diversity** — richness and evenness within a single sample:
        - Shannon H', Simpson (1-D), Inverse Simpson, Fisher's α
        - Margalef, Menhinick, Berger-Parker, Pielou evenness
        - Chao1 and ACE richness estimators

        **Beta Diversity** — dissimilarity between samples:
        - Bray-Curtis, Jaccard, Sørensen, Morisita-Horn
        - Whittaker β_W

        **Rarefaction** — species richness at standardised sample sizes

        **Species Accumulation** — cumulative richness as samples added

        **SHE Analysis** — distinguishes log-normal vs. broken-stick community patterns

        ---

        ### 📋 Data Format

        Each row is a **sample (site)** and each species column contains
        **abundance counts** (integers) or **presence/absence** (0/1):

        | site | sp_A | sp_B | sp_C | sp_D |
        |------|------|------|------|------|
        | S1   | 12   | 0    | 5    | 3    |
        | S2   | 0    | 8    | 7    | 1    |
        | S3   | 4    | 3    | 0    | 9    |

        ---

        ### 📖 Interpreting Results

        | Index | Range | Higher means… |
        |-------|-------|---------------|
        | **Shannon H'** | 0 – ~5 | More diverse community |
        | **Simpson 1-D** | 0 – 1 | More evenly distributed |
        | **Chao1** | ≥ observed S | More undetected species |
        | **Pielou J'** | 0 – 1 | More even abundance |
        | **Fisher's α** | 0 – ∞ | Greater diversity (sample-size robust) |

        **Beta diversity** values range from **0** (identical communities)
        to **1** (no shared species). Use Bray-Curtis for abundance data
        and Jaccard/Sørensen for presence/absence data.

        ---

        ### 🔄 Typical Workflow

        1. Select species columns → run **Alpha Diversity**
        2. Compare sites → run **Beta Diversity** (distance matrix + heatmap)
        3. Standardise sampling effort → **Rarefaction Curves**
        4. Assess sampling completeness → **Species Accumulation**
        5. Diagnose community model → **SHE Analysis**
        """)

    # ----- Column selection -----
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    st.subheader("📋 Data Setup")
    col_a, col_b = st.columns(2)

    with col_a:
        species_cols = st.multiselect(
            "Species / abundance columns:",
            numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
            key="eco_species_cols"
        )
    with col_b:
        sample_col = st.selectbox(
            "Sample label column (optional):",
            [None] + all_cols,
            key="eco_sample_col"
        )
        beta_metric = st.selectbox(
            "Beta diversity metric:",
            ['bray_curtis', 'jaccard', 'sorensen', 'morisita_horn'],
            key="eco_beta_metric"
        )

    if not species_cols or len(species_cols) < 2:
        st.info("Select at least 2 species columns to begin analysis.")
        return

    # ----- Subtabs -----
    eco_tabs = st.tabs([
        "📊 Alpha Diversity",
        "🔢 Beta Diversity",
        "📈 Rarefaction",
        "📉 Species Accumulation",
        "🔍 SHE Analysis"
    ])

    data = df[species_cols].values.astype(float)
    if sample_col and sample_col in df.columns:
        labels = df[sample_col].astype(str).tolist()
    else:
        labels = [f"Sample_{i}" for i in range(data.shape[0])]

    # ===== ALPHA DIVERSITY =====
    with eco_tabs[0]:
        st.subheader("Alpha Diversity Indices")

        alpha_rows = []
        for i in range(data.shape[0]):
            row = all_alpha_diversity(data[i])
            row['Sample'] = labels[i]
            alpha_rows.append(row)
        alpha_df = pd.DataFrame(alpha_rows)
        cols_order = ['Sample'] + [c for c in alpha_df.columns if c != 'Sample']
        alpha_df = alpha_df[cols_order]

        st.dataframe(alpha_df.style.format({c: '{:.4f}' for c in alpha_df.select_dtypes(include='number').columns}), use_container_width=True)

        # Bar chart of selected index
        index_choice = st.selectbox(
            "Index to plot:",
            [c for c in alpha_df.columns if c != 'Sample'],
            key="eco_alpha_index"
        )
        fig = px.bar(
            alpha_df, x='Sample', y=index_choice,
            title=f"{index_choice} by Sample",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.markdown("**Summary across samples:**")
        summary = alpha_df.drop(columns=['Sample']).describe().T
        st.dataframe(summary.style.format('{:.4f}'), use_container_width=True)

    # ===== BETA DIVERSITY =====
    with eco_tabs[1]:
        st.subheader("Beta Diversity Matrix")
        st.caption(f"Metric: {beta_metric}")

        dm = distance_matrix(data, metric=beta_metric)
        beta_df = pd.DataFrame(dm, index=labels, columns=labels)

        # Heatmap
        fig = px.imshow(
            dm, x=labels, y=labels,
            color_continuous_scale='RdYlBu_r',
            title=f"Pairwise {beta_metric.replace('_', ' ').title()} Dissimilarity",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(beta_df.style.format('{:.4f}'), use_container_width=True)

        # Whittaker beta
        wb = whittaker_beta(data)
        st.metric("Whittaker β_W", f"{wb:.4f}")
        st.caption("β_W = (γ / mean α) − 1. Higher values = more species turnover between samples.")

    # ===== RAREFACTION =====
    with eco_tabs[2]:
        st.subheader("Individual-based Rarefaction Curves")

        n_steps = st.slider("Curve resolution:", 20, 200, 50, key="eco_rare_steps")

        fig = go.Figure()
        for i in range(data.shape[0]):
            sizes, expected = rarefaction_curve(data[i], steps=n_steps)
            fig.add_trace(go.Scatter(
                x=sizes, y=expected,
                mode='lines', name=labels[i]
            ))
        fig.update_layout(
            title="Rarefaction Curves",
            xaxis_title="Number of Individuals",
            yaxis_title="Expected Species Richness",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Curves flattening early → well-sampled community. "
                "Steep curves → more species expected with further sampling.")

    # ===== SPECIES ACCUMULATION =====
    with eco_tabs[3]:
        st.subheader("Sample-based Species Accumulation")

        n_perms = st.slider("Permutations:", 10, 500, 100, step=10, key="eco_acc_perms")

        acc_x, acc_mean, acc_std = species_accumulation(data, permutations=n_perms)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acc_x, y=acc_mean + acc_std,
            mode='lines', line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=acc_x, y=acc_mean - acc_std,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(68,114,196,0.2)',
            name='±1 SD'
        ))
        fig.add_trace(go.Scatter(
            x=acc_x, y=acc_mean,
            mode='lines+markers', name='Mean richness',
            line=dict(color='rgb(68,114,196)', width=2)
        ))
        fig.update_layout(
            title="Species Accumulation Curve",
            xaxis_title="Number of Samples",
            yaxis_title="Cumulative Species Richness",
            template=PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 If the curve plateaus, the community is well-sampled. "
                f"Observed total γ-richness: **{int(np.sum(np.any(data > 0, axis=0)))}** species.")

    # ===== SHE ANALYSIS =====
    with eco_tabs[4]:
        st.subheader("SHE Analysis")
        st.caption("Plots ln(S), H', and ln(E) against ln(N) to distinguish community patterns")

        sample_idx = st.selectbox(
            "Select sample:", range(len(labels)),
            format_func=lambda i: labels[i],
            key="eco_she_sample"
        )

        she = she_analysis(data[sample_idx], steps=50)

        if len(she['ln_N']) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=she['ln_N'], y=she['ln_S'],
                mode='lines+markers', name='ln(S)',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=she['ln_N'], y=she['H'],
                mode='lines+markers', name="H'",
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=she['ln_N'], y=she['ln_E'],
                mode='lines+markers', name='ln(E)',
                line=dict(color='red')
            ), secondary_y=True)
            fig.update_layout(
                title=f"SHE Analysis — {labels[sample_idx]}",
                xaxis_title="ln(N)",
                template=PLOTLY_TEMPLATE
            )
            fig.update_yaxes(title_text="ln(S) / H'", secondary_y=False)
            fig.update_yaxes(title_text="ln(E)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - **All three increasing linearly** → log-series distribution
            - **ln(S) and H' increase, ln(E) flat** → log-normal distribution
            - **ln(S) flat, H' decreasing** → broken-stick / niche pre-emption
            """)
        else:
            st.warning("Not enough data for SHE analysis.")
