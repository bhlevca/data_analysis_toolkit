"""
Bayesian Analysis Tab for the Data Analysis Toolkit.

Provides UI for:
- Bayesian regression with configurable priors
- Load / upload / save prior specification files (JSON)
- Prior distribution visualisation
- Posterior coefficient distributions
- Credible intervals (Bayesian, NOT frequentist confidence intervals)
- Prior sensitivity analysis
- Model comparison via BIC / Bayes factor approximation
- Prediction & coefficient export (CSV download)
"""

import io
import json
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pathlib import Path

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from bayesian_analysis import BayesianAnalysis

PLOTLY_TEMPLATE = "plotly_white"


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for st.download_button."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()


def render_bayesian_tab():
    """Render Bayesian analysis tab."""
    st.header("📈 Bayesian Inference & Analysis")
    st.caption(
        "Bayesian regression with explicit priors, posterior distributions, "
        "and credible intervals.  Unlike frequentist methods, Bayesian analysis "
        "incorporates prior knowledge and produces probability statements about parameters."
    )

    # --- Bayesian vs Frequentist explainer ---
    with st.expander("ℹ️ Bayesian vs Frequentist — what's different?", expanded=False):
        st.markdown("""
| Feature | Frequentist | Bayesian |
|---------|------------|----------|
| **Probability** | Long-run frequency | Degree of belief (updated by data) |
| **Prior knowledge** | Not used | Incorporated via priors |
| **Key metrics** | p-values, Confidence Intervals | Posterior distribution, **Credible Intervals** |
| **Goal** | Evaluate inferences given assumptions | Produce inferences given data + priors |

**Credible intervals** give the probability that the parameter lies within
the interval *given the observed data*, whereas confidence intervals only
guarantee long-run coverage under repeated sampling.
""")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    bayesian = BayesianAnalysis(df)

    n_params = len(features) + 1  # intercept + features

    # ==================================================================
    # Prior Configuration — three modes: preset file, upload, manual
    # ==================================================================
    st.subheader("🔧 Prior Configuration")
    st.caption(
        "The prior encodes your beliefs *before* seeing the data.  "
        "You can load a prior from a **preset file**, **upload** your own "
        "JSON file, or configure the prior **manually**."
    )

    prior_source = st.radio(
        "Prior source",
        ["📂 Preset file", "📤 Upload JSON", "✏️ Manual entry"],
        horizontal=True,
        key="prior_source_radio",
    )

    # Resolved prior values
    prior_mean = None
    prior_precision = 0.01
    noise_var_estimate = None
    loaded_prior_name = None

    # ── Preset files ─────────────────────────────────────────────────
    if prior_source == "📂 Preset file":
        presets = BayesianAnalysis.list_prior_presets()
        if not presets:
            st.info("No preset prior files found in `test_data/priors/`. "
                    "Create JSON files there or upload one below.")
        else:
            preset_names = [p['name'] for p in presets]
            chosen_idx = st.selectbox(
                "Select prior preset",
                range(len(preset_names)),
                format_func=lambda i: f"{preset_names[i]} (precision={presets[i]['prior_precision']})",
            )
            chosen = presets[chosen_idx]
            try:
                prior_data = BayesianAnalysis.load_prior(chosen['path'])
                loaded_prior_name = prior_data.get('name', chosen['name'])

                # Show what was loaded
                with st.expander(f"📋 Prior details: {loaded_prior_name}", expanded=True):
                    st.markdown(f"**Description:** {prior_data.get('description', 'N/A')}")
                    st.markdown(f"**Precision:** {prior_data['prior_precision']}")
                    st.markdown(f"**Noise var:** {prior_data.get('noise_var_estimate', 'estimate from data')}")

                    param_names = ['Intercept'] + list(prior_data.get('features', features))
                    mean_vals = prior_data['prior_mean']
                    # Show table
                    prior_table = pd.DataFrame({
                        'Parameter': param_names[:len(mean_vals)],
                        'Prior Mean': np.round(mean_vals, 4),
                    })
                    st.dataframe(prior_table, use_container_width=True)

                # Check dimensionality match
                if len(prior_data['prior_mean']) != n_params:
                    st.warning(
                        f"⚠️ Prior has {len(prior_data['prior_mean'])} parameters "
                        f"but current data needs {n_params} (intercept + {len(features)} features).  "
                        f"The prior will be padded/truncated."
                    )
                    # Pad or truncate
                    pm = np.zeros(n_params)
                    n_copy = min(len(prior_data['prior_mean']), n_params)
                    pm[:n_copy] = prior_data['prior_mean'][:n_copy]
                    prior_mean = pm
                else:
                    prior_mean = prior_data['prior_mean']
                prior_precision = prior_data['prior_precision']
                noise_var_estimate = prior_data.get('noise_var_estimate')
            except Exception as e:
                st.error(f"Could not load prior: {e}")

    # ── Upload JSON ──────────────────────────────────────────────────
    elif prior_source == "📤 Upload JSON":
        uploaded = st.file_uploader(
            "Upload a prior JSON file",
            type=['json'],
            help="See test_data/priors/README.md for the expected format",
        )
        if uploaded is not None:
            try:
                content = json.loads(uploaded.read().decode('utf-8'))
                # Write to temp file so load_prior can validate
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False, encoding='utf-8'
                ) as tmp:
                    json.dump(content, tmp)
                    tmp_path = tmp.name
                prior_data = BayesianAnalysis.load_prior(tmp_path)
                loaded_prior_name = prior_data.get('name', 'Uploaded prior')

                st.success(f"✅ Loaded: **{loaded_prior_name}**")
                st.markdown(f"*{prior_data.get('description', '')}*")

                if len(prior_data['prior_mean']) != n_params:
                    st.warning(
                        f"⚠️ Prior has {len(prior_data['prior_mean'])} params "
                        f"but data needs {n_params}. Padding/truncating."
                    )
                    pm = np.zeros(n_params)
                    n_copy = min(len(prior_data['prior_mean']), n_params)
                    pm[:n_copy] = prior_data['prior_mean'][:n_copy]
                    prior_mean = pm
                else:
                    prior_mean = prior_data['prior_mean']
                prior_precision = prior_data['prior_precision']
                noise_var_estimate = prior_data.get('noise_var_estimate')

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            except Exception as e:
                st.error(f"Invalid prior file: {e}")
        else:
            st.info("Upload a JSON file with prior_mean, prior_precision, etc.")

    # ── Manual entry ─────────────────────────────────────────────────
    else:
        col_prior1, col_prior2 = st.columns(2)
        with col_prior1:
            prior_precision = st.select_slider(
                "Prior precision (strength)",
                options=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
                value=0.01,
                help=(
                    "0.001 = very weak (data-driven), "
                    "~n/σ² ≈ equal weight of prior and data, "
                    "1000 = very strong (prior dominates). "
                    "Features are internally standardised so this scale "
                    "is comparable across all predictors."
                ),
            )
        with col_prior2:
            use_custom_noise = st.checkbox("Specify noise variance")
            if use_custom_noise:
                noise_var_estimate = st.number_input(
                    "Noise variance (σ²)", min_value=0.01, value=4.0, step=0.5,
                    help="Known or assumed noise variance. Leave unchecked to estimate from data."
                )

        st.markdown("**Prior mean** for each parameter:")
        param_names = ['Intercept'] + list(features)
        cols = st.columns(min(len(param_names), 4))
        manual_means = []
        for i, name in enumerate(param_names):
            with cols[i % len(cols)]:
                val = st.number_input(
                    f"{name}", value=0.0, step=0.1, format="%.2f",
                    key=f"prior_mean_{i}",
                )
                manual_means.append(val)
        prior_mean = np.array(manual_means)

        # Offer to save the configured prior
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            save_name = st.text_input("Prior name (for saving)", value="my_custom_prior")
        with save_col2:
            if st.button("💾 Save Prior", use_container_width=True):
                save_path = Path("test_data/priors") / f"{save_name}.json"
                try:
                    BayesianAnalysis.save_prior(
                        path=save_path,
                        prior_mean=prior_mean,
                        prior_precision=prior_precision,
                        noise_var_estimate=noise_var_estimate,
                        features=features,
                        target=target,
                        name=save_name,
                        description=f"User-configured prior (precision={prior_precision})",
                    )
                    st.success(f"✅ Saved to `{save_path}`")
                except Exception as e:
                    st.error(f"Could not save: {e}")

    n_samples = st.number_input(
        "Posterior samples", 500, 10000, 2000, step=500,
        help="Number of draws from the posterior distribution",
    )

    # ── Prior distribution visualisation & download ──────────────────
    st.subheader("📐 Prior Distribution Preview")
    st.caption(
        "Visualises your chosen prior for each parameter.  "
        "The prior is N(mean, σ²/precision) for each coefficient.  "
        "A wider curve means more uncertainty (weaker prior)."
    )

    # Estimate sigma2 for the prior visualisation
    _sigma2_est = 4.0  # default fallback
    if df is not None and target in df.columns:
        _y_tmp = df[target].dropna()
        if len(_y_tmp) > 10:
            _sigma2_est = float(_y_tmp.var())

    _pm = prior_mean if prior_mean is not None else np.zeros(n_params)
    _pp = max(prior_precision, 1e-8)
    _prior_var = _sigma2_est / _pp
    _prior_std = np.sqrt(_prior_var)

    param_names_viz = ['Intercept'] + list(features)
    n_show_prior = min(4, len(param_names_viz))
    prior_cols = st.columns(n_show_prior)
    for i in range(n_show_prior):
        with prior_cols[i]:
            mu_i = float(_pm[i]) if i < len(_pm) else 0.0
            x_range = np.linspace(mu_i - 4 * _prior_std, mu_i + 4 * _prior_std, 200)
            from scipy.stats import norm as _norm
            y_range = _norm.pdf(x_range, loc=mu_i, scale=_prior_std)
            fig_prior = go.Figure()
            fig_prior.add_trace(go.Scatter(
                x=x_range, y=y_range, mode='lines',
                fill='tozeroy', fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='orange', width=2),
                name='Prior',
            ))
            fig_prior.add_vline(x=mu_i, line_color='orange', line_dash='dash')
            fig_prior.update_layout(
                title=param_names_viz[i],
                xaxis_title='Value', yaxis_title='Density',
                showlegend=False, height=250,
                template=PLOTLY_TEMPLATE,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_prior, use_container_width=True)

    # Download current prior as JSON
    _prior_download = {
        'name': loaded_prior_name or 'Current prior',
        'description': f'Prior precision={_pp}, estimated_sigma2={_sigma2_est:.2f}',
        'features': list(features),
        'target': target,
        'prior_mean': [float(v) for v in _pm],
        'prior_precision': float(_pp),
        'noise_var_estimate': noise_var_estimate,
    }
    st.download_button(
        "⬇️ Download current prior (JSON)",
        data=json.dumps(_prior_download, indent=2),
        file_name="bayesian_prior.json",
        mime="application/json",
    )

    st.markdown("---")

    # ── Action buttons ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        run_regression = st.button("🎲 Bayesian Regression", use_container_width=True)
    with col2:
        confidence = st.slider("Credible interval level", 0.80, 0.99, 0.95, 0.01)
        run_credible = st.button("📊 Credible Intervals", use_container_width=True)
    with col3:
        run_sensitivity = st.button("🔍 Prior Sensitivity", use_container_width=True)

    # ── Run analyses ─────────────────────────────────────────────────
    if run_regression:
        with st.spinner("Sampling from posterior..."):
            results = bayesian.bayesian_regression(
                features, target,
                n_samples=int(n_samples),
                prior_mean=prior_mean,
                prior_precision=prior_precision,
                noise_var_estimate=noise_var_estimate,
            )
            st.session_state.analysis_results['bayesian'] = results

    if run_credible:
        with st.spinner("Computing credible intervals..."):
            results = bayesian.credible_intervals(features, target, confidence)
            st.session_state.analysis_results['credible'] = results

    if run_sensitivity:
        with st.spinner("Running prior sensitivity grid..."):
            results = bayesian.prior_sensitivity(features, target)
            st.session_state.analysis_results['prior_sensitivity'] = results

    st.markdown("---")

    # ── Display: Bayesian Regression ─────────────────────────────────
    if 'bayesian' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['bayesian']
        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎲 Bayesian Regression Results")

            # Show prior info
            prior_info = results.get('prior_info', {})
            if prior_info:
                st.info(f"**Prior used:** {prior_info.get('description', 'N/A')}")

            posterior_mean = results.get('posterior_mean', [])
            feat_names = results.get('features', [])
            ci_lower = results.get('credible_intervals_lower', [])
            ci_upper = results.get('credible_intervals_upper', [])
            ols_coefs = results.get('ols_coefficients', [])
            shrinkage = results.get('shrinkage_factor', [])

            if len(posterior_mean) > 0:
                # ── Coefficient table with OLS comparison & shrinkage ──
                coef_df = pd.DataFrame({
                    'Parameter': feat_names,
                    'OLS Estimate': np.round(ols_coefs, 4) if len(ols_coefs) else ['—'] * len(feat_names),
                    'Posterior Mean': np.round(posterior_mean, 4),
                    '95% CI Lower': np.round(ci_lower, 4),
                    '95% CI Upper': np.round(ci_upper, 4),
                    'CI Width': np.round(np.array(ci_upper) - np.array(ci_lower), 4),
                    'Shrinkage': np.round(shrinkage, 4) if len(shrinkage) else ['—'] * len(feat_names),
                })
                st.dataframe(coef_df, use_container_width=True)

                # ── Summary metrics ──────────────────────────────────
                r2 = results.get('r2_posterior_mean', None)
                noise_var = results.get('noise_variance', None)
                n_obs = results.get('n_observations', None)
                prec_info = results.get('prior_info', {}).get('prior_precision', None)

                metric_cols = st.columns(4)
                if r2 is not None:
                    metric_cols[0].metric("R² (posterior mean)", f"{r2:.4f}")
                if noise_var is not None:
                    metric_cols[1].metric("Noise variance (σ²)", f"{noise_var:.4f}")
                if n_obs is not None:
                    data_prec = n_obs / (noise_var or 1.0)
                    metric_cols[2].metric("Data precision (n/σ²)", f"{data_prec:.1f}")
                if prec_info is not None:
                    metric_cols[3].metric("Prior precision", f"{prec_info:.4g}")

                # ── Download coefficients ────────────────────────────
                st.download_button(
                    "⬇️ Download coefficients (CSV)",
                    data=_to_csv_bytes(coef_df),
                    file_name="bayesian_coefficients.csv",
                    mime="text/csv",
                    key="dl_coefs",
                )

                # ── Coefficient bar chart with credible intervals ────
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=feat_names,
                    y=posterior_mean,
                    error_y=dict(
                        type='data',
                        array=np.array(ci_upper) - np.array(posterior_mean),
                        arrayminus=np.array(posterior_mean) - np.array(ci_lower),
                    ),
                    marker_color='steelblue',
                    name='Posterior Mean',
                ))
                # Overlay OLS as markers for comparison
                if len(ols_coefs) > 0:
                    fig.add_trace(go.Scatter(
                        x=feat_names,
                        y=np.array(ols_coefs),
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='diamond'),
                        name='OLS Estimate',
                    ))
                fig.update_layout(
                    title='Posterior Coefficients with 95% Credible Intervals (red ◆ = OLS)',
                    yaxis_title='Coefficient value',
                    template=PLOTLY_TEMPLATE,
                    height=400,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)

                # ── Posterior distribution histograms ─────────────────
                samples = results.get('posterior_samples')
                if samples is not None and len(feat_names) > 0:
                    st.subheader("📊 Posterior Distributions")
                    n_show = min(4, len(feat_names))
                    cols = st.columns(n_show)
                    for i in range(n_show):
                        with cols[i]:
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(
                                x=samples[:, i], nbinsx=50,
                                marker_color='steelblue', opacity=0.7,
                                name='Posterior',
                            ))
                            fig_hist.add_vline(x=posterior_mean[i], line_color='red',
                                               line_dash='dash', annotation_text='Mean')
                            fig_hist.add_vline(x=ci_lower[i], line_color='green',
                                               line_dash='dash', annotation_text='2.5%')
                            fig_hist.add_vline(x=ci_upper[i], line_color='green',
                                               line_dash='dash', annotation_text='97.5%')
                            fig_hist.update_layout(
                                title=feat_names[i],
                                showlegend=False,
                                height=300,
                                template=PLOTLY_TEMPLATE,
                                margin=dict(l=40, r=20, t=40, b=30),
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                # ── Residual diagnostics ─────────────────────────────
                residuals = results.get('residuals')
                predictions = results.get('predictions')
                y_actual = results.get('y_actual')
                if residuals is not None and predictions is not None:
                    st.subheader("🔍 Residual Diagnostics")
                    diag_c1, diag_c2 = st.columns(2)
                    with diag_c1:
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            x=predictions, y=residuals,
                            mode='markers', marker=dict(size=4, color='steelblue', opacity=0.5),
                        ))
                        fig_resid.add_hline(y=0, line_dash='dash', line_color='red')
                        fig_resid.update_layout(
                            title='Residuals vs Fitted',
                            xaxis_title='Fitted values',
                            yaxis_title='Residuals',
                            template=PLOTLY_TEMPLATE, height=350,
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                    with diag_c2:
                        fig_qq = go.Figure()
                        sorted_resid = np.sort(residuals)
                        n_r = len(sorted_resid)
                        from scipy.stats import norm as _norm_qq
                        theoretical = _norm_qq.ppf(np.linspace(0.01, 0.99, n_r))
                        fig_qq.add_trace(go.Scatter(
                            x=theoretical, y=sorted_resid,
                            mode='markers', marker=dict(size=4, color='steelblue'),
                        ))
                        lim = max(abs(theoretical.min()), abs(theoretical.max()))
                        resid_scale = sorted_resid.std()
                        fig_qq.add_trace(go.Scatter(
                            x=[-lim, lim], y=[-lim * resid_scale, lim * resid_scale],
                            mode='lines', line=dict(color='red', dash='dash'),
                            showlegend=False,
                        ))
                        fig_qq.update_layout(
                            title='Q-Q Plot (residuals)',
                            xaxis_title='Theoretical quantiles',
                            yaxis_title='Observed quantiles',
                            template=PLOTLY_TEMPLATE, height=350,
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)

                    # ── Download predictions ─────────────────────────
                    if y_actual is not None:
                        pred_df = pd.DataFrame({
                            'actual': y_actual,
                            'predicted': predictions,
                            'residual': residuals,
                            'ci_lower': predictions - 1.96 * np.sqrt(results.get('noise_variance', 1)),
                            'ci_upper': predictions + 1.96 * np.sqrt(results.get('noise_variance', 1)),
                        })
                        st.download_button(
                            "⬇️ Download predictions (CSV)",
                            data=_to_csv_bytes(pred_df),
                            file_name="bayesian_predictions.csv",
                            mime="text/csv",
                            key="dl_preds",
                        )

    # ── Display: Credible Intervals ──────────────────────────────────
    if 'credible' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['credible']
        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("📊 Posterior Predictive Credible Intervals")
            st.caption(
                "These are **credible intervals**, not confidence intervals. "
                "A 95% credible interval means there is a 95% posterior probability "
                "that the true value lies within the interval."
            )

            cov = results.get('coverage', 0)
            width = results.get('mean_ci_width', 0)
            conf = results.get('confidence', 0.95)
            m_alpha = results.get('model_alpha', None)
            m_lambda = results.get('model_lambda', None)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Coverage", f"{cov*100:.1f}%")
            c2.metric("Mean CI Width", f"{width:.4f}")
            if m_alpha is not None:
                c3.metric("Noise precision (α)", f"{m_alpha:.2f}")
            if m_lambda is not None:
                c4.metric("Weight precision (λ)", f"{m_lambda:.2f}")

            # Actual vs Predicted with credible band
            y_actual = results.get('y_actual')
            y_pred = results.get('y_pred')
            ci_lo = results.get('ci_lower')
            ci_hi = results.get('ci_upper')

            if y_actual is not None and y_pred is not None:
                order = np.argsort(y_actual)
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Scatter(
                    x=y_actual[order], y=ci_hi[order],
                    mode='lines', line=dict(width=0), showlegend=False,
                ))
                fig_ci.add_trace(go.Scatter(
                    x=y_actual[order], y=ci_lo[order],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(70,130,180,0.2)',
                    name=f'{conf*100:.0f}% Credible Interval',
                ))
                fig_ci.add_trace(go.Scatter(
                    x=y_actual[order], y=y_pred[order],
                    mode='markers', marker=dict(size=4, color='steelblue'),
                    name='Predictions',
                ))
                lims = [y_actual.min(), y_actual.max()]
                fig_ci.add_trace(go.Scatter(
                    x=lims, y=lims, mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect prediction',
                ))
                fig_ci.update_layout(
                    title=f'Actual vs Predicted with {conf*100:.0f}% Credible Intervals',
                    xaxis_title='Actual', yaxis_title='Predicted',
                    template=PLOTLY_TEMPLATE, height=500,
                )
                st.plotly_chart(fig_ci, use_container_width=True)

                # Download credible interval predictions
                ci_export = pd.DataFrame({
                    'actual': y_actual,
                    'predicted': y_pred,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'ci_width': ci_hi - ci_lo,
                })
                st.download_button(
                    "⬇️ Download credible intervals (CSV)",
                    data=_to_csv_bytes(ci_export),
                    file_name="bayesian_credible_intervals.csv",
                    mime="text/csv",
                    key="dl_ci",
                )

    # ── Display: Prior Sensitivity ───────────────────────────────────
    if 'prior_sensitivity' in st.session_state.analysis_results:
        sens_results = st.session_state.analysis_results['prior_sensitivity']
        if sens_results:
            st.subheader("🔍 Prior Sensitivity Analysis")
            st.caption(
                "Shows how posterior estimates change as the prior precision varies.  "
                "If results are stable across priors, the data dominates (good).  "
                "If results shift strongly, the prior matters and should be chosen carefully."
            )

            sens_df = pd.DataFrame([
                {
                    'Prior Precision': r['prior_precision'],
                    'R²': round(r['r2'], 4),
                    'Noise Var': round(r['noise_variance'], 4),
                    **{f'β_{i}': round(m, 4) for i, m in enumerate(r['posterior_mean'])}
                }
                for r in sens_results
            ])
            st.dataframe(sens_df, use_container_width=True)

            st.download_button(
                "⬇️ Download sensitivity table (CSV)",
                data=_to_csv_bytes(sens_df),
                file_name="bayesian_sensitivity.csv",
                mime="text/csv",
                key="dl_sens",
            )

            # Line chart showing coefficient stability
            fig_sens = go.Figure()
            n_coefs = len(sens_results[0]['posterior_mean'])
            precs = [r['prior_precision'] for r in sens_results]
            for j in range(n_coefs):
                means = [r['posterior_mean'][j] for r in sens_results]
                name = f"β_{j} ({'Intercept' if j == 0 else features[j-1]})"
                fig_sens.add_trace(go.Scatter(
                    x=precs, y=means, mode='lines+markers', name=name,
                ))
                # Add OLS reference as dashed line if available
                if 'ols_coefficients' in sens_results[0]:
                    ols_val = sens_results[0]['ols_coefficients'][j]
                    fig_sens.add_trace(go.Scatter(
                        x=[precs[0], precs[-1]], y=[ols_val, ols_val],
                        mode='lines', line=dict(dash='dot', width=1),
                        showlegend=False, opacity=0.4,
                    ))
            fig_sens.update_layout(
                title='Posterior Means vs Prior Precision',
                xaxis_title='Prior Precision', xaxis_type='log',
                yaxis_title='Posterior Mean',
                template=PLOTLY_TEMPLATE, height=400,
            )
            st.plotly_chart(fig_sens, use_container_width=True)

    # ==================================================================
    # MCMC Sampling (non-conjugate priors)
    # ==================================================================
    st.markdown("---")
    st.subheader("🔗 MCMC Sampling (Advanced)")
    st.caption(
        "Markov Chain Monte Carlo allows **non-Normal priors** "
        "(Laplace for sparsity, Student-t for robustness, Cauchy for heavy tails, etc.).  "
        "This uses Metropolis-within-Gibbs sampling — slower than conjugate but far more flexible."
    )

    with st.expander("⚙️ MCMC Configuration", expanded=False):
        from mcmc import MCMCSampler, Prior, SUPPORTED_PRIORS

        mcmc_c1, mcmc_c2, mcmc_c3 = st.columns(3)
        with mcmc_c1:
            mcmc_n_iter = st.number_input(
                "Total iterations", 1000, 50000, 5000, step=1000,
                help="Total MCMC iterations per chain (includes warmup)",
                key="mcmc_n_iter",
            )
        with mcmc_c2:
            mcmc_n_warmup = st.number_input(
                "Warmup (burn-in)", 200, 20000, 1000, step=200,
                help="Iterations to discard as burn-in",
                key="mcmc_n_warmup",
            )
        with mcmc_c3:
            mcmc_n_chains = st.selectbox(
                "Chains", [1, 2, 3, 4], index=1,
                help="More chains = better convergence diagnostics (R-hat needs ≥2)",
                key="mcmc_n_chains",
            )

        # Per-parameter prior configuration
        st.markdown("**Prior distributions per parameter:**")
        dist_options = sorted(SUPPORTED_PRIORS - {"inverse_gamma", "half_normal", "half_cauchy"})
        param_names_mcmc = ["Intercept"] + list(features)

        mcmc_priors = {}
        prior_cols = st.columns(min(len(param_names_mcmc), 4))
        for i, pname in enumerate(param_names_mcmc):
            with prior_cols[i % len(prior_cols)]:
                st.markdown(f"**{pname}**")
                dist = st.selectbox(
                    f"Distribution", dist_options,
                    index=dist_options.index("normal"),
                    key=f"mcmc_dist_{i}",
                    label_visibility="collapsed",
                )
                loc = st.number_input(
                    "Location", value=0.0, step=0.5,
                    key=f"mcmc_loc_{i}",
                    help="Mean / centre of the prior",
                )
                scale = st.number_input(
                    "Scale", value=10.0, min_value=0.01, step=1.0,
                    key=f"mcmc_scale_{i}",
                    help="Spread of the prior (std for Normal, diversity for Laplace)",
                )
                mcmc_priors[pname] = Prior(
                    distribution=dist, loc=loc, scale=scale,
                )

        # sigma2 prior (always positive)
        st.markdown("**Noise variance (σ²) prior:**")
        sig_c1, sig_c2, sig_c3 = st.columns(3)
        with sig_c1:
            sig_dist = st.selectbox(
                "σ² distribution",
                ["inverse_gamma", "half_normal", "half_cauchy"],
                index=0,
                key="mcmc_sig_dist",
            )
        with sig_c2:
            sig_a = st.number_input("Shape (a)", value=1.0, min_value=0.01, step=0.5, key="mcmc_sig_a")
        with sig_c3:
            sig_b = st.number_input("Scale (b)", value=1.0, min_value=0.01, step=0.5, key="mcmc_sig_b")
        mcmc_priors["sigma2"] = Prior(distribution=sig_dist, a=sig_a, b=sig_b, scale=sig_b)

        run_mcmc = st.button("🔗 Run MCMC", use_container_width=True)

    if run_mcmc:
        with st.spinner(f"Running MCMC ({mcmc_n_iter} iterations × {mcmc_n_chains} chains)..."):
            sampler = MCMCSampler(df)
            mcmc_result = sampler.run(
                features, target,
                priors=mcmc_priors,
                n_iter=int(mcmc_n_iter),
                n_warmup=int(mcmc_n_warmup),
                n_chains=int(mcmc_n_chains),
            )
            st.session_state.analysis_results['mcmc'] = mcmc_result

    # ── Display MCMC results ─────────────────────────────────────────
    if 'mcmc' in st.session_state.analysis_results:
        mcmc_res = st.session_state.analysis_results['mcmc']
        if 'error' in mcmc_res:
            st.error(mcmc_res['error'])
        else:
            st.subheader("🔗 MCMC Results")

            # Convergence diagnostics
            diag = mcmc_res.get('diagnostics', {})
            r_hat = diag.get('r_hat', [])
            ess = diag.get('ess', [])
            converged = diag.get('converged', True)
            diag_warnings = diag.get('warnings', [])

            if not converged:
                for w in diag_warnings:
                    st.warning(f"⚠️ {w}")
            else:
                st.success("✅ Chains converged (R-hat ≤ 1.1, ESS ≥ 100 for all parameters)")

            acc_rate = mcmc_res.get('acceptance_rate', 0)
            st.caption(f"Average acceptance rate: {acc_rate:.1%}")

            # Parameter summary table
            pnames = mcmc_res.get('param_names', [])
            post_mean = mcmc_res.get('posterior_mean', [])
            post_std = mcmc_res.get('posterior_std', [])
            ci_lo = mcmc_res.get('credible_intervals_lower', [])
            ci_hi = mcmc_res.get('credible_intervals_upper', [])
            priors_used = mcmc_res.get('priors_used', {})

            if len(post_mean) > 0:
                summary_rows = []
                for i, name in enumerate(pnames):
                    row = {
                        'Parameter': name,
                        'Posterior Mean': round(float(post_mean[i]), 4),
                        'Posterior Std': round(float(post_std[i]), 4),
                        '95% CI Lower': round(float(ci_lo[i]), 4),
                        '95% CI Upper': round(float(ci_hi[i]), 4),
                    }
                    if i < len(r_hat):
                        row['R-hat'] = round(float(r_hat[i]), 3)
                    if i < len(ess):
                        row['ESS'] = int(ess[i])
                    prior_info = priors_used.get(name, {})
                    row['Prior'] = prior_info.get('distribution', 'N/A')
                    summary_rows.append(row)

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True)

                mcmc_r2 = mcmc_res.get('r2', None)
                noise_v = mcmc_res.get('noise_variance', None)
                mc1, mc2, mc3 = st.columns(3)
                if mcmc_r2 is not None:
                    mc1.metric("R²", f"{mcmc_r2:.4f}")
                if noise_v is not None:
                    mc2.metric("σ² (posterior mean)", f"{noise_v:.4f}")
                mc3.metric("Chains × Iterations", f"{mcmc_res.get('n_chains', 0)} × {mcmc_res.get('n_iter', 0)}")

                st.download_button(
                    "⬇️ Download MCMC summary (CSV)",
                    data=_to_csv_bytes(summary_df),
                    file_name="mcmc_summary.csv",
                    mime="text/csv",
                    key="dl_mcmc_summary",
                )

            # Trace plots + posterior histograms
            chains = mcmc_res.get('chains')
            if chains is not None and len(pnames) > 0:
                st.subheader("📈 Trace Plots & Posteriors")
                n_show_mcmc = min(4, len(pnames))
                for i in range(n_show_mcmc):
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        fig_trace = go.Figure()
                        for c_idx in range(chains.shape[0]):
                            fig_trace.add_trace(go.Scatter(
                                y=chains[c_idx, :, i],
                                mode='lines', opacity=0.7,
                                name=f'Chain {c_idx + 1}',
                            ))
                        fig_trace.update_layout(
                            title=f'Trace: {pnames[i]}',
                            xaxis_title='Iteration (post-warmup)',
                            yaxis_title='Value',
                            template=PLOTLY_TEMPLATE,
                            height=250,
                            margin=dict(l=40, r=20, t=40, b=30),
                            showlegend=(i == 0),
                        )
                        st.plotly_chart(fig_trace, use_container_width=True)
                    with tc2:
                        combined_samples = chains[:, :, i].ravel()
                        fig_post = go.Figure()
                        fig_post.add_trace(go.Histogram(
                            x=combined_samples, nbinsx=60,
                            marker_color='steelblue', opacity=0.7,
                        ))
                        fig_post.add_vline(x=float(post_mean[i]), line_color='red', line_dash='dash')
                        if i < len(ci_lo):
                            fig_post.add_vline(x=float(ci_lo[i]), line_color='green', line_dash='dash')
                            fig_post.add_vline(x=float(ci_hi[i]), line_color='green', line_dash='dash')
                        fig_post.update_layout(
                            title=f'Posterior: {pnames[i]}',
                            xaxis_title='Value',
                            yaxis_title='Count',
                            template=PLOTLY_TEMPLATE,
                            height=250,
                            margin=dict(l=40, r=20, t=40, b=30),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_post, use_container_width=True)

            # Download MCMC predictions
            mcmc_preds = mcmc_res.get('predictions')
            mcmc_resids = mcmc_res.get('residuals')
            mcmc_y = mcmc_res.get('y_actual')
            if mcmc_preds is not None and mcmc_y is not None:
                pred_export = pd.DataFrame({
                    'actual': mcmc_y,
                    'predicted': mcmc_preds,
                    'residual': mcmc_resids,
                })
                st.download_button(
                    "⬇️ Download MCMC predictions (CSV)",
                    data=_to_csv_bytes(pred_export),
                    file_name="mcmc_predictions.csv",
                    mime="text/csv",
                    key="dl_mcmc_preds",
                )
