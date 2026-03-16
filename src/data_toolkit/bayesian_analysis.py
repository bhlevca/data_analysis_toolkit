"""
Bayesian Analysis Module
========================
Bayesian regression with explicit priors, posterior distributions,
credible intervals, posterior predictive checks, and model comparison.

Key Bayesian concepts implemented:
- Prior specification (informative or weakly informative)
- Posterior computation via conjugate Normal-Inverse-Gamma prior
- Credible intervals (NOT confidence intervals - these are Bayesian)
- Posterior predictive distribution for new data
- Model comparison via marginal likelihood (Bayes factor approximation)
- Prior sensitivity analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from scipy import stats as sp_stats
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BayesianAnalysis:
    """
    Bayesian statistical analysis using conjugate Normal-Inverse-Gamma priors.

    Bayesian vs Frequentist
    -----------------------
    +-----------------------+----------------------------+-------------------------------+
    | Feature               | Frequentist                | Bayesian                      |
    +-----------------------+----------------------------+-------------------------------+
    | Probability           | Long-run frequency         | Degree of belief (updated)    |
    | Prior knowledge       | Not used                   | Incorporated via priors       |
    | Key metrics           | p-values, confidence int.  | Posterior, credible intervals  |
    | Goal                  | Evaluate under assumptions | Infer given data + priors     |
    +-----------------------+----------------------------+-------------------------------+
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.model = None
        self.posterior_samples = None
        self._last_prior = None  # store prior info for inspection

    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze."""
        self.df = df

    # ------------------------------------------------------------------
    # Prior specification: load / save / list
    # ------------------------------------------------------------------
    @staticmethod
    def load_prior(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a prior specification from a JSON file.

        Expected JSON structure::

            {
                "name": "Informative prior for test data",
                "description": "Domain-knowledge prior ...",
                "features": ["predictor_1", "predictor_2", "predictor_3"],
                "target": "response_homoscedastic",
                "prior_mean": [5.0, 2.0, -1.5, 0.8],
                "prior_precision": 1.0,
                "noise_var_estimate": 4.0
            }

        ``prior_mean`` is [intercept, beta_1, ..., beta_p].
        ``prior_precision`` is a positive scalar (larger = stronger prior).
        ``noise_var_estimate`` is optional (null -> empirical Bayes).

        Returns
        -------
        dict  with at least 'prior_mean' and 'prior_precision' keys.
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            prior = json.load(f)

        # Validate required fields
        if 'prior_mean' not in prior:
            raise ValueError(f"Prior file {path.name} missing 'prior_mean'")
        if 'prior_precision' not in prior:
            raise ValueError(f"Prior file {path.name} missing 'prior_precision'")

        # Convert to numpy where useful
        prior['prior_mean'] = np.array(prior['prior_mean'], dtype=float)
        prior['prior_precision'] = float(prior['prior_precision'])
        if prior.get('noise_var_estimate') is not None:
            prior['noise_var_estimate'] = float(prior['noise_var_estimate'])
        else:
            prior['noise_var_estimate'] = None

        return prior

    @staticmethod
    def save_prior(
        path: Union[str, Path],
        prior_mean: Union[list, np.ndarray],
        prior_precision: float,
        noise_var_estimate: float = None,
        features: List[str] = None,
        target: str = None,
        name: str = "Custom prior",
        description: str = "",
    ) -> Path:
        """
        Save a prior specification to a JSON file.

        Parameters
        ----------
        path : str or Path
            Where to write the JSON file.
        prior_mean : array-like
            Prior mean vector [intercept, beta_1, ..., beta_p].
        prior_precision : float
            Prior precision scalar.
        noise_var_estimate : float or None
            Fixed noise variance (None -> estimate from data).
        features / target : list[str] / str
            Column names for documentation.
        name / description : str
            Human-readable labels.

        Returns
        -------
        Path  of the written file.
        """
        path = Path(path)
        prior_obj = {
            'name': name,
            'description': description,
            'features': features or [],
            'target': target or '',
            'prior_mean': list(np.asarray(prior_mean, dtype=float)),
            'prior_precision': float(prior_precision),
            'noise_var_estimate': noise_var_estimate,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prior_obj, f, indent=2)
        return path

    @staticmethod
    def list_prior_presets(directory: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """
        List available prior JSON files in a directory.

        Parameters
        ----------
        directory : str or Path or None
            Folder to scan.  None -> ``test_data/priors`` relative to
            the workspace root (auto-detected).

        Returns
        -------
        list of dicts, each with 'path', 'name', 'description',
        'prior_precision', 'n_params'.
        """
        if directory is None:
            # Try common locations
            for candidate in [
                Path('test_data/priors'),
                Path(__file__).resolve().parents[3] / 'test_data' / 'priors',
            ]:
                if candidate.is_dir():
                    directory = candidate
                    break
            else:
                return []
        directory = Path(directory)
        presets = []
        for p in sorted(directory.glob('*.json')):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                presets.append({
                    'path': str(p),
                    'name': obj.get('name', p.stem),
                    'description': obj.get('description', ''),
                    'prior_precision': obj.get('prior_precision'),
                    'n_params': len(obj.get('prior_mean', [])),
                })
            except Exception:
                continue
        return presets

    # ------------------------------------------------------------------
    # Core Bayesian Regression  (conjugate Normal-Inverse-Gamma)
    # ------------------------------------------------------------------
    def bayesian_regression(
        self,
        features: List[str],
        target: str,
        n_samples: int = 2000,
        prior_mean: np.ndarray = None,
        prior_precision: float = 0.01,
        noise_var_estimate: float = None,
    ) -> Dict[str, Any]:
        """
        Bayesian linear regression with conjugate Normal-Inverse-Gamma prior.

        Features are internally **standardised** (zero mean, unit variance)
        before the prior precision matrix is applied, so that
        ``prior_precision`` is on a comparable scale for every feature.
        Results are transformed back to the original feature scale.

        The model is::

            y = X @ beta + epsilon,   epsilon ~ N(0, sigma^2)

        Prior on beta (standardised space):
            beta* ~ N(prior_mean*, sigma^2 * (prior_precision * I)^{-1})

        If ``noise_var_estimate`` is None the residual variance is estimated
        from an OLS fit (empirical Bayes).

        Parameters
        ----------
        features : list[str]
            Predictor column names.
        target : str
            Response column name.
        n_samples : int
            Number of posterior draws.
        prior_mean : ndarray or None
            Prior mean for [intercept, beta_1, ..., beta_p] in the
            **original** feature scale.
            None -> zero vector (weakly informative).
        prior_precision : float
            Scalar controlling prior tightness (larger = tighter prior).
            Thanks to internal standardisation, this is comparable across
            features:  ~ n/sigma^2 means "as informative as the data".
            Default 0.01 is weakly informative; use 10+ for strong priors.
        noise_var_estimate : float or None
            Known/assumed noise variance.  None -> estimate from data.

        Returns
        -------
        dict with keys:
            posterior_mean, posterior_cov, posterior_samples,
            credible_intervals_lower/upper (95 %),
            features, prior_info, noise_variance, r2_posterior_mean,
            ols_coefficients, shrinkage_factor, predictions, residuals,
            standardization (x_means, x_stds)
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index].values
        n, p = X.shape
        X_raw = X.values

        # ── Standardise features ─────────────────────────────────────
        x_means = X_raw.mean(axis=0)
        x_stds = X_raw.std(axis=0, ddof=0)
        x_stds[x_stds == 0] = 1.0  # avoid division by zero
        X_std = (X_raw - x_means) / x_stds

        # Design matrix with intercept (standardised space)
        X_mat = np.column_stack([np.ones(n), X_std])
        k = p + 1  # intercept + features

        # ── Prior (in standardised space) ────────────────────────────
        if prior_mean is None:
            prior_mean_orig = np.zeros(k)
        else:
            prior_mean_orig = np.asarray(prior_mean, dtype=float)

        # Convert original-scale prior mean → standardised space
        # y = a + b1*x1 + ... = a + b1*(mu1 + s1*z1) + ...
        #   = (a + sum(bj*muj)) + sum(bj*sj)*zj
        prior_mean_std = np.zeros(k)
        prior_mean_std[0] = prior_mean_orig[0] + np.sum(
            prior_mean_orig[1:k] * x_means[:p]
        )
        prior_mean_std[1:] = prior_mean_orig[1:k] * x_stds[:p]

        Lambda_0 = np.eye(k) * prior_precision  # prior precision matrix

        # ── Noise variance ───────────────────────────────────────────
        beta_ols_std = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        resid_ols = y - X_mat @ beta_ols_std
        ols_sigma2 = float(np.var(resid_ols, ddof=k))

        if noise_var_estimate is None:
            noise_var_estimate = ols_sigma2
        tau = 1.0 / noise_var_estimate  # noise precision

        # ── Posterior (conjugate update, standardised space) ─────────
        Lambda_n = Lambda_0 + tau * (X_mat.T @ X_mat)
        post_cov_std = np.linalg.inv(Lambda_n)
        post_mean_std = post_cov_std @ (
            Lambda_0 @ prior_mean_std + tau * X_mat.T @ y
        )

        # ── Sample from posterior (standardised space) ───────────────
        samples_std = np.random.multivariate_normal(
            post_mean_std, post_cov_std, n_samples,
        )

        # ── Transform back to original scale ─────────────────────────
        # T maps standardised params → original-scale params
        # intercept_orig = a* - sum(b*_j * mu_j / s_j)
        # beta_j_orig = b*_j / s_j
        T = np.zeros((k, k))
        T[0, 0] = 1.0
        for j in range(p):
            T[0, j + 1] = -x_means[j] / x_stds[j]
            T[j + 1, j + 1] = 1.0 / x_stds[j]

        posterior_mean = T @ post_mean_std
        posterior_cov = T @ post_cov_std @ T.T
        self.posterior_samples = (T @ samples_std.T).T  # (n_samples, k)

        # 95 % credible intervals (original scale)
        ci = np.percentile(self.posterior_samples, [2.5, 97.5], axis=0)

        # ── OLS coefficients (original scale) for comparison ─────────
        ols_orig = T @ beta_ols_std

        # ── Predictions & residuals (original scale) ─────────────────
        X_mat_orig = np.column_stack([np.ones(n), X_raw])
        y_hat = X_mat_orig @ posterior_mean
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        # ── Shrinkage: how much the prior moved estimates from OLS ───
        shrinkage = np.where(
            np.abs(ols_orig - prior_mean_orig[:k]) > 1e-12,
            (posterior_mean - ols_orig) / (prior_mean_orig[:k] - ols_orig),
            0.0,
        )

        self._last_prior = {
            'prior_mean': prior_mean_orig.tolist(),
            'prior_precision': prior_precision,
            'noise_variance': noise_var_estimate,
            'description': (
                f"Normal prior (standardised): β* ~ N(m, σ²/{prior_precision:.4g}·I), "
                f"σ² = {noise_var_estimate:.4f}"
            ),
        }

        return {
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'posterior_samples': self.posterior_samples,
            'credible_intervals_lower': ci[0],
            'credible_intervals_upper': ci[1],
            'features': ['Intercept'] + list(features),
            'prior_info': self._last_prior,
            'noise_variance': noise_var_estimate,
            'r2_posterior_mean': r2,
            # new diagnostics
            'ols_coefficients': ols_orig,
            'shrinkage_factor': shrinkage,
            'predictions': y_hat,
            'residuals': residuals,
            'y_actual': y,
            'standardization': {
                'x_means': x_means,
                'x_stds': x_stds,
            },
            'data_precision_per_obs': tau,
            'n_observations': n,
        }

    # ------------------------------------------------------------------
    # Credible Intervals (prediction-level)
    # ------------------------------------------------------------------
    def credible_intervals(
        self,
        features: List[str],
        target: str,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Posterior predictive credible intervals for observed data.

        Uses sklearn BayesianRidge which places a Gamma hyper-prior on both
        the noise precision (alpha) and the weight precision (lambda), then
        estimates full posterior via evidence maximisation.

        The returned intervals are **credible intervals** - they express the
        probability that the true value lies within the interval *given the
        data and prior*, as opposed to frequentist confidence intervals.

        Parameters
        ----------
        confidence : float
            Width of the credible interval (e.g. 0.95 for 95 %).

        Returns
        -------
        dict with y_pred, y_std, ci_lower, ci_upper, coverage,
        mean_ci_width, model_alpha, model_lambda
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]

        self.model = BayesianRidge(compute_score=True)
        self.model.fit(X, y)

        y_pred, y_std = self.model.predict(X, return_std=True)

        # Credible interval bounds from posterior predictive
        # The posterior predictive is Student-t, well-approximated by Normal
        # for n >> p.  We use the posterior std directly.
        alpha = 1 - confidence
        hi_pct = 1 - alpha / 2

        # Degrees of freedom for posterior predictive t-distribution
        n = len(y)
        dof = max(n - X.shape[1] - 1, 1)
        t_val = sp_stats.t.ppf(hi_pct, df=dof)

        ci_lower = y_pred - t_val * y_std
        ci_upper = y_pred + t_val * y_std

        # Empirical coverage
        coverage = float(np.mean((y.values >= ci_lower) & (y.values <= ci_upper)))

        return {
            'y_pred': y_pred,
            'y_std': y_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'coverage': coverage,
            'mean_ci_width': float(np.mean(ci_upper - ci_lower)),
            'y_actual': y.values,
            'confidence': confidence,
            # Bayesian hyper-parameters learned
            'model_alpha': float(self.model.alpha_),   # noise precision
            'model_lambda': float(self.model.lambda_),  # weight precision
        }

    # ------------------------------------------------------------------
    # Posterior Distributions  (full coefficient posteriors)
    # ------------------------------------------------------------------
    def posterior_distributions(
        self,
        features: List[str],
        target: str,
        n_samples: int = 2000,
    ) -> Dict[str, Any]:
        """
        Return posterior distributions for every coefficient.

        Uses BayesianRidge to learn hyper-parameters, then draws from the
        approximate posterior N(coef_, sigma^2 * (X'X + lambda*I)^{-1}).

        Returns
        -------
        dict with keys: coefficients (name->mean), intercept,
        posterior_samples (n_samples x p+1), features,
        alpha, lambda_, credible_intervals_95
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]

        model = BayesianRidge(compute_score=True)
        model.fit(X, y)

        # Approximate posterior covariance for coefficients
        sigma2 = 1.0 / model.alpha_     # noise variance
        XtX = X.values.T @ X.values
        reg = model.lambda_ / model.alpha_ * np.eye(X.shape[1])
        cov = sigma2 * np.linalg.inv(XtX + reg * X.shape[0])

        # Draw posterior samples  [intercept, coefs...]
        mean_full = np.concatenate([[model.intercept_], model.coef_])
        # Intercept variance ~ sigma2 / n
        n = len(y)
        cov_full = np.zeros((len(mean_full), len(mean_full)))
        cov_full[0, 0] = sigma2 / n
        cov_full[1:, 1:] = cov

        samples = np.random.multivariate_normal(mean_full, cov_full, n_samples)
        ci = np.percentile(samples, [2.5, 97.5], axis=0)

        feat_names = ['Intercept'] + list(features)

        return {
            'alpha': float(model.alpha_),
            'lambda_': float(model.lambda_),
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': float(model.intercept_),
            'scores': model.scores_.tolist() if hasattr(model, 'scores_') and model.scores_ is not None else None,
            'posterior_samples': samples,
            'features': feat_names,
            'credible_intervals_95': {
                feat_names[i]: (float(ci[0, i]), float(ci[1, i]))
                for i in range(len(feat_names))
            },
        }

    # ------------------------------------------------------------------
    # Posterior Predictive Distribution
    # ------------------------------------------------------------------
    def posterior_predictive(
        self,
        features: List[str],
        target: str,
        new_data: pd.DataFrame = None,
        n_samples: int = 2000,
    ) -> Dict[str, Any]:
        """
        Generate posterior predictive samples for new observations.

        For each posterior draw of beta, generate a prediction including
        observation noise.  This gives the full predictive distribution,
        not just point estimates.

        Features are internally standardised (same as bayesian_regression).

        Parameters
        ----------
        new_data : DataFrame or None
            New observations to predict. None -> use training data.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X_train = self.df[features].dropna()
        y_train = self.df[target].loc[X_train.index].values
        n, p = X_train.shape
        X_raw = X_train.values

        # Standardise
        x_means = X_raw.mean(axis=0)
        x_stds = X_raw.std(axis=0, ddof=0)
        x_stds[x_stds == 0] = 1.0
        X_std = (X_raw - x_means) / x_stds
        X_mat = np.column_stack([np.ones(n), X_std])
        k = p + 1

        # OLS for noise estimate
        beta_ols = np.linalg.lstsq(X_mat, y_train, rcond=None)[0]
        resid = y_train - X_mat @ beta_ols
        sigma2 = float(np.var(resid, ddof=k))

        # Conjugate posterior in standardised space
        prior_prec = 0.01
        Lambda_0 = np.eye(k) * prior_prec
        tau = 1.0 / sigma2
        Lambda_n = Lambda_0 + tau * (X_mat.T @ X_mat)
        post_cov = np.linalg.inv(Lambda_n)
        post_mean = post_cov @ (tau * X_mat.T @ y_train)

        # Transform matrix: standardised → original
        T = np.zeros((k, k))
        T[0, 0] = 1.0
        for j in range(p):
            T[0, j + 1] = -x_means[j] / x_stds[j]
            T[j + 1, j + 1] = 1.0 / x_stds[j]

        # Sample betas in original scale
        samples_std = np.random.multivariate_normal(post_mean, post_cov, n_samples)
        beta_samples = (T @ samples_std.T).T  # (n_samples, k)

        # Prepare prediction matrix (original scale, with intercept)
        if new_data is not None:
            X_new = np.column_stack([np.ones(len(new_data)), new_data[features].values])
        else:
            X_new = np.column_stack([np.ones(n), X_raw])

        # Predictive samples: y* = X_new @ beta + eps
        pred_samples = X_new @ beta_samples.T  # (n_new, n_samples)
        pred_samples += np.random.normal(0, np.sqrt(sigma2), pred_samples.shape)

        pred_mean = pred_samples.mean(axis=1)
        pred_std = pred_samples.std(axis=1)
        ci = np.percentile(pred_samples, [2.5, 97.5], axis=1)

        return {
            'predictive_mean': pred_mean,
            'predictive_std': pred_std,
            'credible_lower': ci[0],
            'credible_upper': ci[1],
            'predictive_samples': pred_samples,
            'noise_variance': sigma2,
        }

    # ------------------------------------------------------------------
    # Model Comparison (BIC-based Bayes-factor approximation)
    # ------------------------------------------------------------------
    def bayesian_model_comparison(
        self,
        features: List[str],
        target: str,
    ) -> List[Dict[str, Any]]:
        """
        Compare candidate models via BIC (approximation to log marginal
        likelihood / Bayes factor).

        Lower BIC ~ higher marginal likelihood ~ better Bayesian model.
        The difference in BIC between two models approximates
        2 * log(Bayes Factor).
        """
        if self.df is None:
            return []

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        n = len(y)

        results = []
        for i in range(1, len(features) + 1):
            for combo in combinations(range(len(features)), i):
                X_sub = X.iloc[:, list(combo)]
                model = BayesianRidge()
                model.fit(X_sub, y)

                k = len(combo) + 1
                y_pred = model.predict(X_sub)
                rss = float(np.sum((y.values - y_pred) ** 2))
                bic = n * np.log(rss / n) + k * np.log(n)

                feat_names = [features[j] for j in combo]
                results.append({
                    'bic': bic,
                    'features': feat_names,
                    'r2': float(model.score(X_sub, y)),
                    'n_features': len(feat_names),
                })

        results.sort(key=lambda x: x['bic'])
        return results

    # ------------------------------------------------------------------
    # Prior Sensitivity Analysis
    # ------------------------------------------------------------------
    def prior_sensitivity(
        self,
        features: List[str],
        target: str,
    ) -> List[Dict[str, Any]]:
        """
        Systematically vary prior precision to show how the prior
        influences posterior estimates.

        Thanks to internal standardisation, prior_precision is on
        a meaningful scale: 0.001 ≈ data-only, ~n/σ² ≈ equal weight,
        1000+ ≈ prior-only.
        """
        if self.df is None:
            return []

        precisions = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        results = []
        for prec in precisions:
            res = self.bayesian_regression(
                features, target,
                n_samples=500,
                prior_precision=prec,
            )
            if 'error' in res:
                continue
            results.append({
                'prior_precision': prec,
                'posterior_mean': res['posterior_mean'].tolist(),
                'ci_lower': res['credible_intervals_lower'].tolist(),
                'ci_upper': res['credible_intervals_upper'].tolist(),
                'r2': res['r2_posterior_mean'],
                'noise_variance': res['noise_variance'],
                'ols_coefficients': res['ols_coefficients'].tolist(),
            })
        return results

    # ------------------------------------------------------------------
    # Plotting helpers  (Matplotlib - for non-Streamlit use)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # MCMC convenience wrapper
    # ------------------------------------------------------------------
    def mcmc_regression(
        self,
        features: List[str],
        target: str,
        priors: Dict = None,
        n_iter: int = 5000,
        n_warmup: int = 1000,
        n_chains: int = 2,
        seed: int = None,
    ) -> Dict[str, Any]:
        """
        Bayesian regression via MCMC (Metropolis-within-Gibbs).

        Supports non-conjugate priors: Normal, Laplace, Student-t,
        Uniform, Cauchy, Half-Normal, Half-Cauchy, Inverse-Gamma.

        Parameters
        ----------
        priors : dict[str, Prior] or None
            Mapping parameter name -> mcmc.Prior instance.
            None -> default weakly-informative Normal + InvGamma(1,1) for sigma2.
        n_iter : int
            Total iterations per chain (including warmup).
        n_warmup : int
            Burn-in iterations to discard.
        n_chains : int
            Independent chains (for R-hat convergence diagnostics).
        seed : int or None
            Random seed.

        Returns
        -------
        dict — see MCMCSampler.run() for full key listing.
        """
        from .mcmc import MCMCSampler
        sampler = MCMCSampler(self.df)
        return sampler.run(
            features, target,
            priors=priors,
            n_iter=n_iter,
            n_warmup=n_warmup,
            n_chains=n_chains,
            seed=seed,
        )

    def plot_posterior_distributions(self, results: Dict[str, Any], max_plots: int = 3) -> plt.Figure:
        """Histogram of posterior coefficient samples with credible intervals."""
        samples = results.get('posterior_samples')
        features = results.get('features')
        posterior_mean = results.get('posterior_mean')
        ci_lower = results.get('credible_intervals_lower')
        ci_upper = results.get('credible_intervals_upper')

        if samples is None:
            return None

        n_plots = min(max_plots, len(features))
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            axes[i].hist(samples[:, i], bins=50, density=True, alpha=0.7,
                         color='steelblue', edgecolor='white')
            axes[i].axvline(posterior_mean[i], color='red', ls='--', lw=1.5, label='Mean')
            axes[i].axvline(ci_lower[i], color='green', ls='--', lw=1, label='95% CI')
            axes[i].axvline(ci_upper[i], color='green', ls='--', lw=1)
            axes[i].set_title(f'Posterior: {features[i]}')
            axes[i].legend(fontsize=8)

        plt.tight_layout()
        return fig

    def plot_credible_intervals(self, results: Dict[str, Any]) -> plt.Figure:
        """Scatter of actual vs predicted with credible interval bands."""
        y_actual = results.get('y_actual')
        y_pred = results.get('y_pred')
        ci_lower = results.get('ci_lower')
        ci_upper = results.get('ci_upper')
        confidence = results.get('confidence', 0.95)

        if y_actual is None or y_pred is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        order = np.argsort(y_actual)
        ax.scatter(y_actual[order], y_pred[order], alpha=0.5, s=20, label='Predictions')
        ax.fill_between(
            y_actual[order], ci_lower[order], ci_upper[order],
            alpha=0.2, color='steelblue', label=f'{confidence*100:.0f}% Credible Interval',
        )
        lims = [y_actual.min(), y_actual.max()]
        ax.plot(lims, lims, 'r--', lw=1, label='Perfect prediction')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Posterior Predictive with {confidence*100:.0f}% Credible Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
