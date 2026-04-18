"""
Curve Fitting Tab for the Data Analysis Toolkit

Provides power, exponential, logistic, sinusoidal, Gompertz,
RMA regression, GLM, and multi-model comparison.
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

from curve_fitting import (
    power_fit, exponential_fit, logistic_fit, sinusoidal_fit,
    gompertz_fit, rma_regression, glm_fit, compare_fits,
)

PLOTLY_TEMPLATE = "plotly_white"

MODEL_DESCRIPTIONS = {
    'Power (Allometric)': {
        'func': power_fit,
        'equation': 'y = a · x^b',
        'use': 'Allometric scaling, species-area relationships',
    },
    'Exponential (2-param)': {
        'func': lambda x, y: exponential_fit(x, y, three_param=False),
        'equation': 'y = a · e^(bx)',
        'use': 'Growth/decay without asymptote',
    },
    'Exponential (3-param)': {
        'func': lambda x, y: exponential_fit(x, y, three_param=True),
        'equation': 'y = a · e^(bx) + c',
        'use': 'Growth/decay with baseline offset',
    },
    'Logistic (4-param)': {
        'func': logistic_fit,
        'equation': 'y = d + (a − d) / (1 + (x/c)^b)',
        'use': 'Dose-response, population growth, sigmoidal data',
    },
    'Sinusoidal': {
        'func': sinusoidal_fit,
        'equation': 'y = A · sin(2πf·x + φ) + offset',
        'use': 'Seasonal/cyclical patterns',
    },
    'Gompertz': {
        'func': gompertz_fit,
        'equation': 'y = a · e^(−b · e^(−cx))',
        'use': 'Asymmetric sigmoid, tumour growth, biological growth',
    },
    'RMA Regression': {
        'func': rma_regression,
        'equation': 'y = a + b·x (Type II)',
        'use': 'Both variables have measurement error',
    },
    'GLM': {
        'func': None,  # handled separately
        'equation': 'g(E[y]) = Xβ',
        'use': 'Generalized linear model with link function',
    },
}


def render_curve_fitting_tab():
    """Render the Curve Fitting tab."""
    st.header("📈 Curve Fitting & Non-Linear Models")
    st.caption("Specialized curve fitting beyond linear regression")

    if st.session_state.df is None:
        st.warning("📁 Please load data first in the Data tab.")
        return

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return

    with st.expander("ℹ️ Available Models", expanded=False):
        md_rows = ["| Model | Equation | Typical Use |", "|-------|----------|-------------|"]
        for name, info in MODEL_DESCRIPTIONS.items():
            md_rows.append(f"| **{name}** | `{info['equation']}` | {info['use']} |")
        st.markdown("\n".join(md_rows))

        st.markdown("""
        ---

        ### 📖 Choosing a Model

        ```
        What does your data look like?
            │
            ├── Straight line with error in both X and Y
            │       └── RMA Regression (Type II)
            │
            ├── Power-law / allometric scaling
            │       └── Power (y = a·x^b)
            │
            ├── Monotonic growth or decay
            │       ├── With asymptote? → Exponential (3-param)
            │       └── Without?        → Exponential (2-param)
            │
            ├── S-shaped / sigmoidal
            │       ├── Symmetric?  → Logistic (4-param)
            │       └── Asymmetric? → Gompertz
            │
            ├── Cyclical / periodic
            │       └── Sinusoidal
            │
            └── Count / binary / categorical response
                    └── GLM (with appropriate link)
        ```

        ---

        ### 📊 Interpreting Fit Quality

        | Metric | Good Value | Meaning |
        |--------|------------|---------|
        | **R²** | > 0.9 | Proportion of variance explained |
        | **RMSE** | Low (relative to data range) | Typical prediction error |
        | **AIC** | Lower is better | Model quality (penalises complexity) |
        | **BIC** | Lower is better | Stricter complexity penalty |

        In **Multi-Model Comparison** mode, AIC/BIC differences (ΔAIC) are shown.
        Models with ΔAIC < 2 are equally supported; ΔAIC > 10 indicates poor support.

        ---

        ### 💡 Tips

        - Use **Multi-Model Comparison** to let the data decide which model fits best
        - Non-linear fits can be sensitive to initial guesses — the toolkit uses smart defaults
        - Always inspect the residual plot: patterns indicate model misspecification
        - RMA regression is preferred over OLS when both X and Y have measurement error
        """)

    # ---------- mode selection ----------
    mode = st.radio(
        "Mode:",
        ['Single Model Fit', 'Multi-Model Comparison'],
        horizontal=True,
        key="cf_mode"
    )

    # ---------- column selection ----------
    col_a, col_b = st.columns(2)
    with col_a:
        x_col = st.selectbox("X (independent) variable:", numeric_cols, key="cf_x")
    with col_b:
        y_col = st.selectbox("Y (dependent) variable:",
                             [c for c in numeric_cols if c != x_col] or numeric_cols,
                             key="cf_y")

    # Clean data
    df_clean = df[[x_col, y_col]].dropna()
    x = df_clean[x_col].values.astype(float)
    y = df_clean[y_col].values.astype(float)

    if len(x) < 3:
        st.error("Need at least 3 data points.")
        return

    st.info(f"Using {len(x)} data points after removing NaN.")
    st.markdown("---")

    # ============================
    # SINGLE MODEL FIT
    # ============================
    if mode == 'Single Model Fit':
        model_name = st.selectbox("Model:", list(MODEL_DESCRIPTIONS.keys()), key="cf_model")

        # GLM-specific options
        glm_family = None
        glm_link = None
        glm_extra_cols = None
        if model_name == 'GLM':
            col1, col2 = st.columns(2)
            with col1:
                glm_family = st.selectbox("Family:", ['gaussian', 'poisson', 'binomial', 'gamma', 'inverse_gaussian'], key="cf_glm_fam")
            with col2:
                link_options = {
                    'gaussian': ['identity', 'log'],
                    'poisson': ['log', 'identity'],
                    'binomial': ['logit', 'probit'],
                    'gamma': ['inverse', 'log', 'identity'],
                    'inverse_gaussian': ['inverse_squared', 'inverse', 'log'],
                }
                glm_link = st.selectbox("Link function:", link_options.get(glm_family, ['identity']), key="cf_glm_link")

            glm_extra_cols = st.multiselect(
                "Additional predictors (optional):",
                [c for c in numeric_cols if c not in (x_col, y_col)],
                key="cf_glm_extra"
            )

        if st.button("🔄 Fit Model", type="primary", key="cf_fit"):
            with st.spinner(f"Fitting {model_name}..."):
                try:
                    if model_name == 'GLM':
                        if glm_extra_cols:
                            X_glm = df_clean[[x_col] + glm_extra_cols].values
                        else:
                            X_glm = x
                        result = glm_fit(X_glm, y, family=glm_family, link=glm_link)
                    else:
                        func = MODEL_DESCRIPTIONS[model_name]['func']
                        result = func(x, y)
                except Exception as e:
                    st.error(f"Fitting failed: {e}")
                    return

            # ----- Display results -----
            st.subheader("Results")

            # Metrics
            cols = st.columns(4)
            cols[0].metric("R²", f"{result['r_squared']:.4f}")
            cols[1].metric("Adj. R²", f"{result.get('adj_r_squared', 0):.4f}")
            cols[2].metric("RMSE", f"{result.get('rmse', 0):.4f}")
            cols[3].metric("AIC", f"{result.get('aic', 0):.1f}")

            # Parameters
            if 'parameters' in result:
                st.subheader("Fitted Parameters")
                params = result['parameters']
                if isinstance(params, dict):
                    param_df = pd.DataFrame([params]).T
                    param_df.columns = ['Value']
                    st.dataframe(param_df.style.format('{:.6f}'))
                else:
                    st.write(params)

            # Equation
            st.markdown(f"**Model**: {MODEL_DESCRIPTIONS[model_name]['equation']}")

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                name='Data', marker=dict(size=6, opacity=0.6)
            ))

            if 'x_fit' in result and 'y_fit' in result:
                fig.add_trace(go.Scatter(
                    x=result['x_fit'], y=result['y_fit'],
                    mode='lines', name=model_name,
                    line=dict(color='red', width=2)
                ))

            fig.update_layout(
                title=f"{model_name} Fit",
                xaxis_title=x_col,
                yaxis_title=y_col,
                template=PLOTLY_TEMPLATE,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            if 'y_fit' in result and 'x_fit' in result:
                # Interpolate fitted values at data points
                from scipy.interpolate import interp1d
                try:
                    f_interp = interp1d(result['x_fit'], result['y_fit'],
                                        kind='linear', fill_value='extrapolate')
                    y_pred = f_interp(x)
                    residuals = y - y_pred

                    with st.expander("📉 Residual Diagnostics"):
                        res_fig = go.Figure()
                        res_fig.add_trace(go.Scatter(
                            x=y_pred, y=residuals, mode='markers',
                            marker=dict(size=5, opacity=0.6)
                        ))
                        res_fig.add_hline(y=0, line_dash='dash', line_color='gray')
                        res_fig.update_layout(
                            title="Residuals vs Fitted",
                            xaxis_title="Fitted values",
                            yaxis_title="Residuals",
                            template=PLOTLY_TEMPLATE,
                            height=400,
                        )
                        st.plotly_chart(res_fig, use_container_width=True)
                except Exception:
                    pass

            # GLM summary
            if model_name == 'GLM' and 'summary' in result:
                with st.expander("📄 GLM Summary"):
                    st.text(result['summary'])

    # ============================
    # MULTI-MODEL COMPARISON
    # ============================
    else:
        st.subheader("Multi-Model Comparison")
        st.write("Fit all applicable models and compare using AIC/BIC/R².")

        models_to_try = st.multiselect(
            "Models to compare:",
            [k for k in MODEL_DESCRIPTIONS if k != 'GLM'],
            default=[k for k in MODEL_DESCRIPTIONS if k not in ('GLM', 'Sinusoidal')],
            key="cf_compare_models"
        )

        if not models_to_try:
            st.info("Select at least one model.")
            return

        if st.button("🔄 Compare Models", type="primary", key="cf_compare"):
            results = {}
            fit_data = {}

            progress = st.progress(0)
            for i, model_name in enumerate(models_to_try):
                try:
                    func = MODEL_DESCRIPTIONS[model_name]['func']
                    result = func(x, y)
                    results[model_name] = result
                    fit_data[model_name] = result
                except Exception as e:
                    st.warning(f"⚠️ {model_name}: {e}")
                progress.progress((i + 1) / len(models_to_try))

            if not results:
                st.error("No models converged.")
                return

            # Build comparison table
            comp_rows = []
            for name, r in results.items():
                comp_rows.append({
                    'Model': name,
                    'R²': r.get('r_squared', np.nan),
                    'Adj. R²': r.get('adj_r_squared', np.nan),
                    'RMSE': r.get('rmse', np.nan),
                    'AIC': r.get('aic', np.nan),
                    'BIC': r.get('bic', np.nan),
                    'N params': r.get('n_params', np.nan),
                })

            comp_df = pd.DataFrame(comp_rows).sort_values('AIC')
            st.dataframe(
                comp_df.style.format({
                    'R²': '{:.4f}',
                    'Adj. R²': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'AIC': '{:.1f}',
                    'BIC': '{:.1f}',
                    'N params': '{:.0f}',
                }).highlight_min(subset=['AIC', 'BIC', 'RMSE'], color='lightgreen')
                 .highlight_max(subset=['R²', 'Adj. R²'], color='lightgreen')
            )

            best_model = comp_df.iloc[0]['Model']
            st.success(f"🏆 Best model by AIC: **{best_model}**")

            # Overlay plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                name='Data', marker=dict(size=6, color='black', opacity=0.5)
            ))

            colors = px.colors.qualitative.Set2
            for idx, (name, r) in enumerate(results.items()):
                if 'x_fit' in r and 'y_fit' in r:
                    fig.add_trace(go.Scatter(
                        x=r['x_fit'], y=r['y_fit'],
                        mode='lines', name=name,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ))

            fig.update_layout(
                title="Model Comparison",
                xaxis_title=x_col,
                yaxis_title=y_col,
                template=PLOTLY_TEMPLATE,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            # AIC bar chart
            fig_aic = px.bar(
                comp_df, x='Model', y='AIC',
                title='AIC Comparison (lower is better)',
                template=PLOTLY_TEMPLATE,
                color='AIC',
                color_continuous_scale='RdYlGn_r',
            )
            st.plotly_chart(fig_aic, use_container_width=True)
