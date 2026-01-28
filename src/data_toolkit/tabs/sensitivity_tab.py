"""
Sensitivity Analysis Tab for Streamlit GUI.

Provides interface for:
- Morris Screening (Elementary Effects)
- Sobol Sensitivity Indices
- One-At-a-Time (OAT) Analysis

# =============================================================================
# ⚠️  IMPORTANT WARNING FOR AI ASSISTANTS AND DEVELOPERS  ⚠️
# =============================================================================
# DO NOT use `use_container_width=True` with st.plotly_chart() or st.dataframe()!
# This parameter was DEPRECATED and removed after 2025-12-31.
#
# CORRECT usage:
#   st.plotly_chart(fig, width='stretch')    # instead of use_container_width=True
#   st.plotly_chart(fig, width='content')    # instead of use_container_width=False
#   st.dataframe(df, width='stretch')        # instead of use_container_width=True
#
# This applies to ALL Streamlit display functions that previously used use_container_width.
# =============================================================================
"""

import os
# Import from parent package
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_analysis import CARTAnalysis, sensitivity_to_cart_workflow
from sensitivity_analysis import (SensitivityAnalysis,
                                  analyze_dataframe_sensitivity)


def render_sensitivity_tab():
    """Render sensitivity analysis tab."""
    st.header("🎯 Sensitivity Analysis")
    st.caption("Global sensitivity analysis to understand how input parameters affect model outputs")

    df = st.session_state.get('df')

    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for sensitivity analysis.")
        return

    # Configuration
    st.subheader("⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Target Variable (Output)",
            numeric_cols,
            key="sens_target"
        )

        available_features = [c for c in numeric_cols if c != target]
        features = st.multiselect(
            "Input Parameters (Features)",
            available_features,
            default=available_features[:min(5, len(available_features))],
            key="sens_features"
        )

    with col2:
        method = st.selectbox(
            "Analysis Method",
            ["Morris Screening", "Sobol Indices", "One-At-a-Time (OAT)"],
            key="sens_method",
            help="""
            **Morris Screening**: Fast screening method to identify important factors.
            Good for initial exploration with many parameters.

            **Sobol Indices**: Variance-based method providing first-order and total-order indices.
            More computationally expensive but gives interaction effects.

            **OAT (One-At-a-Time)**: Simple local sensitivity by varying each parameter individually.
            Fast but doesn't capture interactions.
            """
        )

        n_samples = st.slider(
            "Number of Samples/Trajectories",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="sens_samples",
            help="More samples = more accurate but slower"
        )

    if len(features) < 1:
        st.warning("Please select at least 1 input parameter.")
        return

    st.markdown("---")

    # Run Analysis
    col1, col2, col3 = st.columns(3)

    with col1:
        run_button = st.button("🚀 Run Sensitivity Analysis", type="primary", key="run_sensitivity_analysis")

    if run_button:
        with st.spinner(f"Running {method}..."):
            try:
                # Map method names
                method_map = {
                    "Morris Screening": "morris",
                    "Sobol Indices": "sobol",
                    "One-At-a-Time (OAT)": "oat"
                }

                results = analyze_dataframe_sensitivity(
                    df, target, features,
                    method=method_map[method],
                    n_samples=n_samples,
                    seed=42
                )

                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                    return

                st.session_state['sensitivity_results'] = results
                st.session_state['sensitivity_method'] = method
                st.success(f"✅ {method} analysis complete!")

            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display Results
    if 'sensitivity_results' in st.session_state:
        results = st.session_state['sensitivity_results']
        method_used = st.session_state.get('sensitivity_method', 'Unknown')

        st.markdown("---")
        st.subheader(f"📊 Results: {method_used}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Visualization
            if 'Morris' in method_used:
                _plot_morris_results(results)
            elif 'Sobol' in method_used:
                _plot_sobol_results(results)
            elif 'OAT' in method_used:
                _plot_oat_results(results)

        with col2:
            # Summary table
            st.subheader("📋 Parameter Ranking")

            if 'ranking' in results:
                ranking_df = pd.DataFrame({
                    'Rank': range(1, len(results['ranking']) + 1),
                    'Parameter': results['ranking']
                })

                # Add scores based on method
                if 'mu_star' in results:
                    ranking_df['μ* (Importance)'] = [
                        f"{results['mu_star'].get(p, 0):.4f}" for p in results['ranking']
                    ]
                if 'ST' in results:
                    ranking_df['Total Effect (ST)'] = [
                        f"{results['ST'].get(p, 0):.4f}" for p in results['ranking']
                    ]
                if 'gradients' in results:
                    ranking_df['Gradient'] = [
                        f"{results['gradients'].get(p, 0):.4f}" for p in results['ranking']
                    ]

                st.dataframe(ranking_df, hide_index=True, width='stretch')

            # Classification (Morris)
            if 'classification' in results:
                st.subheader("🏷️ Classification")
                for param, cls in results['classification'].items():
                    if cls == 'negligible':
                        st.write(f"⚪ {param}: Negligible")
                    elif cls == 'linear':
                        st.write(f"🔵 {param}: Linear effect")
                    else:
                        st.write(f"🔴 {param}: Nonlinear/Interaction")

        # Model coefficients (for reference)
        if 'model_coefficients' in results:
            with st.expander("📐 Linear Model Coefficients (Reference)"):
                coef_df = pd.DataFrame({
                    'Parameter': list(results['model_coefficients'].keys()),
                    'Coefficient': list(results['model_coefficients'].values())
                })
                st.dataframe(coef_df, hide_index=True)
                st.caption("Note: Sensitivity analysis uses a fitted linear model. "
                          "These coefficients show the underlying relationships.")

        # CART Workflow Section (after sensitivity results)
        st.markdown("---")
        _render_cart_workflow_section(df, target, features, results)

    else:
        # CART Workflow Section (standalone - without prior sensitivity analysis)
        st.markdown("---")
        st.subheader("🌳 CART Analysis Workflow")
        st.info("💡 **Tip**: Run Sensitivity Analysis first to identify important parameters, "
               "or use CART directly with all selected features below.")
        _render_cart_workflow_section(df, target, features, None)

    # Educational content
    with st.expander("ℹ️ About Sensitivity Analysis Methods"):
        st.markdown("""
        ### Morris Screening (Elementary Effects)
        - **Purpose**: Fast screening to identify important/unimportant factors
        - **Key Statistics**:
          - **μ* (mu-star)**: Mean of absolute elementary effects - overall importance
          - **σ (sigma)**: Standard deviation of effects - indicates nonlinearity or interactions
        - **Interpretation**: High μ* = important; High σ/μ* ratio = nonlinear or interactive effects

        ### Sobol Sensitivity Indices
        - **Purpose**: Variance-based decomposition of output uncertainty
        - **Key Statistics**:
          - **S1 (First-order)**: Direct contribution of each parameter
          - **ST (Total-order)**: Total contribution including interactions
        - **Interpretation**: ST >> S1 indicates strong interaction effects

        ### One-At-a-Time (OAT)
        - **Purpose**: Simple local sensitivity analysis
        - **Key Statistics**:
          - **Gradient**: Rate of change of output w.r.t. input
          - **Elasticity**: Normalized sensitivity (% change output / % change input)
        - **Limitation**: Doesn't capture interactions between parameters

        ### References
        - Morris, M.D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments
        - Saltelli, A. et al. (2010). Variance based sensitivity analysis of model output
        """)


def _plot_morris_results(results):
    """Plot Morris screening results using Plotly."""
    if 'mu_star' not in results or 'sigma' not in results:
        st.warning("Morris results not available")
        return

    param_names = results.get('param_names', list(results['mu_star'].keys()))
    mu_star = [results['mu_star'].get(p, 0) for p in param_names]
    sigma = [results['sigma'].get(p, 0) for p in param_names]

    # Classification colors
    colors = []
    for p in param_names:
        cls = results.get('classification', {}).get(p, 'unknown')
        if cls == 'negligible':
            colors.append('gray')
        elif cls == 'linear':
            colors.append('blue')
        else:
            colors.append('red')

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=mu_star,
        y=sigma,
        mode='markers+text',
        text=param_names,
        textposition='top center',
        marker=dict(size=15, color=colors, opacity=0.7, line=dict(width=1, color='black')),
        hovertemplate="<b>%{text}</b><br>μ*: %{x:.4f}<br>σ: %{y:.4f}<extra></extra>"
    ))

    # Add reference line σ = 0.5 * μ*
    max_val = max(max(mu_star) if mu_star else 1, max(sigma) if sigma else 1) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, 0.5 * max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='σ = 0.5μ*',
        showlegend=True
    ))

    fig.update_layout(
        title="Morris Screening Plot",
        xaxis_title="μ* (Mean of |Elementary Effects|)",
        yaxis_title="σ (Std of Elementary Effects)",
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig, width='stretch')

    # Also show bar chart
    fig2 = px.bar(
        x=param_names, y=mu_star,
        title="Parameter Importance (μ*)",
        labels={'x': 'Parameter', 'y': 'μ* (Mean |EE|)'},
        template="plotly_white"
    )
    fig2.update_traces(marker_color=colors)
    st.plotly_chart(fig2, width='stretch')


def _plot_sobol_results(results):
    """Plot Sobol indices using Plotly."""
    if 'S1' not in results or 'ST' not in results:
        st.warning("Sobol results not available")
        return

    param_names = results.get('param_names', list(results['S1'].keys()))
    S1 = [results['S1'].get(p, 0) for p in param_names]
    ST = [results['ST'].get(p, 0) for p in param_names]

    fig = go.Figure()

    x = list(range(len(param_names)))
    width = 0.35

    fig.add_trace(go.Bar(
        x=[i - width/2 for i in x],
        y=S1,
        width=width,
        name='First-order (S1)',
        marker_color='steelblue'
    ))

    fig.add_trace(go.Bar(
        x=[i + width/2 for i in x],
        y=ST,
        width=width,
        name='Total-order (ST)',
        marker_color='darkorange'
    ))

    # Add confidence intervals if available
    if 'S1_conf' in results:
        s1_err = [[S1[i] - results['S1_conf'][p][0] for i, p in enumerate(param_names)],
                  [results['S1_conf'][p][1] - S1[i] for i, p in enumerate(param_names)]]
        fig.data[0].error_y = dict(type='data', symmetric=False, array=s1_err[1], arrayminus=s1_err[0])

    if 'ST_conf' in results:
        st_err = [[ST[i] - results['ST_conf'][p][0] for i, p in enumerate(param_names)],
                  [results['ST_conf'][p][1] - ST[i] for i, p in enumerate(param_names)]]
        fig.data[1].error_y = dict(type='data', symmetric=False, array=st_err[1], arrayminus=st_err[0])

    fig.update_layout(
        title="Sobol Sensitivity Indices",
        xaxis=dict(tickmode='array', tickvals=x, ticktext=param_names, tickangle=45),
        yaxis_title="Sensitivity Index",
        template="plotly_white",
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig, width='stretch')

    # Interaction indicator
    st.subheader("🔗 Interaction Analysis")
    interaction_df = pd.DataFrame({
        'Parameter': param_names,
        'S1': S1,
        'ST': ST,
        'ST - S1': [ST[i] - S1[i] for i in range(len(param_names))],
        'Interaction %': [f"{(ST[i] - S1[i]) / max(ST[i], 0.001) * 100:.1f}%" for i in range(len(param_names))]
    })
    st.dataframe(interaction_df, hide_index=True)
    st.caption("Large ST - S1 values indicate the parameter has strong interactions with others.")


def _plot_oat_results(results):
    """Plot OAT results using Plotly."""
    if 'sweeps' not in results:
        st.warning("OAT results not available")
        return

    param_names = results.get('param_names', list(results['sweeps'].keys()))
    n_params = len(param_names)

    # Determine subplot layout
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{p} (grad: {results['gradients'].get(p, 0):.3f})" for p in param_names]
    )

    for i, name in enumerate(param_names):
        row = i // n_cols + 1
        col = i % n_cols + 1

        sweep = results['sweeps'][name]

        fig.add_trace(
            go.Scatter(
                x=sweep['values'],
                y=sweep['outputs'],
                mode='lines',
                name=name,
                line=dict(width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        # Add base point marker
        base_val = results['base_point'].get(name, sweep['values'][len(sweep['values'])//2])
        base_out = results['base_output']
        fig.add_trace(
            go.Scatter(
                x=[base_val],
                y=[base_out],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                name='Base',
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="One-At-a-Time Sensitivity Sweeps",
        template="plotly_white",
        height=300 * n_rows
    )

    st.plotly_chart(fig, width='stretch')

    # Gradient bar chart
    gradients = [abs(results['gradients'].get(p, 0)) for p in param_names]
    fig2 = px.bar(
        x=param_names, y=gradients,
        title="Absolute Gradients (|∂y/∂x|)",
        labels={'x': 'Parameter', 'y': '|Gradient|'},
        template="plotly_white"
    )
    st.plotly_chart(fig2, width='stretch')


def _render_cart_workflow_section(df, target, features, sens_results):
    """Render CART analysis workflow section."""

    # Check if we have sensitivity results or running standalone
    has_sens_results = sens_results is not None and 'ranking' in sens_results

    with st.expander("🔗 Morris → CART → Monte Carlo Workflow", expanded=True):
        if has_sens_results:
            st.markdown("""
            **Workflow Steps:**
            1. ✅ **Morris Screening** - Completed! Parameters ranked by importance
            2. **CART (Decision Tree)** - Build model using top parameters
            3. **Monte Carlo** - Quantify prediction uncertainty
            """)
        else:
            st.markdown("""
            **Workflow Steps:**
            1. **Morris Screening** - Run sensitivity analysis first (optional)
            2. **CART (Decision Tree)** - Build model using selected parameters
            3. **Monte Carlo** - Quantify prediction uncertainty

            *Note: Without sensitivity analysis, CART will use all selected features.*
            """)

        # Get ranked parameters (from sensitivity results or use all features)
        if has_sens_results:
            ranking = sens_results.get('ranking', features)
            st.success(f"📊 Using {len(ranking)} parameters ranked by sensitivity analysis")
        else:
            ranking = features
            st.info(f"📋 Using all {len(ranking)} selected features (no sensitivity ranking)")

        col1, col2 = st.columns(2)

        with col1:
            top_n = st.slider(
                "Number of Top Parameters",
                min_value=1,
                max_value=len(ranking),
                value=min(10, len(ranking)),
                key="cart_top_n",
                help="Select the top N most important parameters for CART"
            )

            max_depth = st.slider(
                "CART Tree Depth",
                min_value=2,
                max_value=15,
                value=5,
                key="cart_depth",
                help="Maximum depth of the decision tree"
            )

        with col2:
            mc_samples = st.slider(
                "Monte Carlo Samples",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="mc_samples",
                help="Number of Monte Carlo simulation samples"
            )

            min_samples_leaf = st.slider(
                "Min Samples per Leaf",
                min_value=1,
                max_value=50,
                value=5,
                key="cart_min_leaf"
            )

        # Show selected parameters
        selected_params = ranking[:top_n]
        st.write(f"**Selected Parameters:** {', '.join(selected_params)}")

        run_cart = st.button("🚀 Run CART + Monte Carlo", type="primary", key="run_cart")

        if run_cart:
            with st.spinner("Running CART analysis and Monte Carlo simulations..."):
                try:
                    cart = CARTAnalysis(df)

                    # Fit CART model
                    cart_result = cart.cart_regression(
                        selected_params, target,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf
                    )

                    if 'error' in cart_result:
                        st.error(f"CART Error: {cart_result['error']}")
                        return

                    # Generate hypercube and run Monte Carlo
                    hypercube = cart.generate_hypercube(selected_params, n_samples=mc_samples)
                    mc_result = cart.monte_carlo_predictions(hypercube)

                    # Store results
                    st.session_state['cart_result'] = cart_result
                    st.session_state['mc_result'] = mc_result
                    st.session_state['hypercube'] = hypercube

                    st.success("✅ CART and Monte Carlo analysis complete!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

        # Display CART results
        if 'cart_result' in st.session_state:
            cart_result = st.session_state['cart_result']
            mc_result = st.session_state['mc_result']
            hypercube = st.session_state['hypercube']

            st.markdown("---")

            # Metrics
            st.subheader("📊 CART Model Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Train R²", f"{cart_result['metrics']['train_r2']:.3f}")
            col2.metric("Test R²", f"{cart_result['metrics']['test_r2']:.3f}")
            col3.metric("CV R² (mean)", f"{cart_result['metrics']['cv_r2_mean']:.3f}")
            col4.metric("Tree Depth", cart_result['tree_depth'])

            # Feature importance
            st.subheader("📈 CART Feature Importance")
            imp_df = pd.DataFrame(cart_result['importance_ranking'], columns=['Parameter', 'Importance'])

            fig = px.bar(
                imp_df, x='Parameter', y='Importance',
                title='CART Feature Importance',
                template='plotly_white'
            )
            fig.update_traces(marker_color='forestgreen')
            st.plotly_chart(fig, width='stretch')

            # Decision tree rules
            with st.expander("🌲 Decision Tree Rules"):
                cart_obj = CARTAnalysis(df)
                cart_obj.model = cart_result['model']
                cart_obj.feature_names = cart_result['feature_names']
                rules = cart_obj.get_tree_rules()
                st.code(rules)

            # Monte Carlo results
            st.subheader("🎲 Monte Carlo Simulation Results")

            stats = mc_result['statistics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Prediction", f"{stats['mean']:.3f}")
            col2.metric("Std Dev", f"{stats['std']:.3f}")
            col3.metric("CV (%)", f"{stats['cv']*100:.1f}%")
            col4.metric("95% CI Width", f"{stats['p95'] - stats['p5']:.3f}")

            # Distribution plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=mc_result['predictions'],
                nbinsx=50,
                name='Predictions',
                marker_color='steelblue',
                opacity=0.7
            ))

            # Add lines for mean and percentiles
            fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {stats['mean']:.2f}")
            fig.add_vline(x=stats['p5'], line_dash="dot", line_color="orange",
                         annotation_text=f"P5: {stats['p5']:.2f}")
            fig.add_vline(x=stats['p95'], line_dash="dot", line_color="orange",
                         annotation_text=f"P95: {stats['p95']:.2f}")

            fig.update_layout(
                title=f"Monte Carlo Prediction Distribution (n={mc_result['n_simulations']})",
                xaxis_title=target,
                yaxis_title="Count",
                template="plotly_white"
            )
            st.plotly_chart(fig, width='stretch')

            # Export options
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)

            with col1:
                # Export hypercube
                csv_hypercube = hypercube.to_csv(index=False)
                st.download_button(
                    "📥 Download Hypercube (CSV)",
                    csv_hypercube,
                    file_name="sensitivity_hypercube.csv",
                    mime="text/csv"
                )

            with col2:
                # Export MC results
                mc_df = pd.DataFrame({
                    'prediction': mc_result['predictions']
                })
                mc_df = pd.concat([hypercube.reset_index(drop=True), mc_df], axis=1)
                csv_mc = mc_df.to_csv(index=False)
                st.download_button(
                    "📥 Download MC Results (CSV)",
                    csv_mc,
                    file_name="monte_carlo_results.csv",
                    mime="text/csv"
                )

            st.info("""
            💡 **Tip**: The exported hypercube and Monte Carlo results can be used for:
            - Further analysis in external tools
            - Importing into other model simulations
            - Creating custom visualizations
            - Archiving for reproducibility
            """)

