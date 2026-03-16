"""
Markov Chain Monte Carlo (MCMC) Sampler
=======================================
Metropolis-Hastings and Gibbs sampling for Bayesian linear regression
with non-conjugate priors.

Supports prior distributions: Normal, Laplace, Student-t, Uniform, Cauchy.

MCMC diagnostics: trace plots, R-hat, ESS, autocorrelation.

Usage
-----
::

    from data_toolkit.mcmc import MCMCSampler, Prior

    sampler = MCMCSampler(df)
    priors = {
        'intercept': Prior('normal', loc=0, scale=10),
        'x1': Prior('laplace', loc=0, scale=1),     # sparse
        'x2': Prior('normal', loc=0, scale=5),
        'sigma2': Prior('inverse_gamma', a=1, b=1),
    }
    result = sampler.run(features, target, priors=priors, n_iter=5000)
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# Prior distribution specification
# =====================================================================

SUPPORTED_PRIORS = {
    "normal",
    "laplace",
    "student_t",
    "uniform",
    "cauchy",
    "half_normal",
    "half_cauchy",
    "inverse_gamma",
}


@dataclass
class Prior:
    """
    Specification for a single prior distribution.

    Parameters
    ----------
    distribution : str
        One of: normal, laplace, student_t, uniform, cauchy,
        half_normal, half_cauchy, inverse_gamma.
    loc : float
        Location / mean parameter (used by most distributions).
    scale : float
        Scale / spread parameter.
    df : float
        Degrees of freedom (for student_t only).
    a, b : float
        Shape parameters (for inverse_gamma) or bounds (for uniform).
    """

    distribution: str = "normal"
    loc: float = 0.0
    scale: float = 10.0
    df: float = 3.0
    a: float = 1.0
    b: float = 1.0

    def __post_init__(self):
        d = self.distribution.lower().strip()
        if d not in SUPPORTED_PRIORS:
            raise ValueError(
                f"Unknown prior '{d}'. Supported: {sorted(SUPPORTED_PRIORS)}"
            )
        self.distribution = d

    def log_pdf(self, x: float) -> float:
        """Log-probability density at x."""
        d = self.distribution
        if d == "normal":
            return sp_stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
        elif d == "laplace":
            return sp_stats.laplace.logpdf(x, loc=self.loc, scale=self.scale)
        elif d == "student_t":
            return sp_stats.t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)
        elif d == "uniform":
            return sp_stats.uniform.logpdf(x, loc=self.a, scale=self.b - self.a)
        elif d == "cauchy":
            return sp_stats.cauchy.logpdf(x, loc=self.loc, scale=self.scale)
        elif d == "half_normal":
            if x < 0:
                return -np.inf
            return sp_stats.halfnorm.logpdf(x, loc=0, scale=self.scale)
        elif d == "half_cauchy":
            if x < 0:
                return -np.inf
            return sp_stats.halfcauchy.logpdf(x, loc=0, scale=self.scale)
        elif d == "inverse_gamma":
            if x <= 0:
                return -np.inf
            return sp_stats.invgamma.logpdf(x, a=self.a, scale=self.b)
        return -np.inf

    def sample(self, size: int = 1) -> np.ndarray:
        """Draw random samples from this prior (for initialisation)."""
        d = self.distribution
        if d == "normal":
            return np.random.normal(self.loc, self.scale, size)
        elif d == "laplace":
            return np.random.laplace(self.loc, self.scale, size)
        elif d == "student_t":
            return self.loc + self.scale * np.random.standard_t(self.df, size)
        elif d == "uniform":
            return np.random.uniform(self.a, self.b, size)
        elif d == "cauchy":
            return self.loc + self.scale * np.random.standard_cauchy(size)
        elif d == "half_normal":
            return np.abs(np.random.normal(0, self.scale, size))
        elif d == "half_cauchy":
            return np.abs(self.loc + self.scale * np.random.standard_cauchy(size))
        elif d == "inverse_gamma":
            return sp_stats.invgamma.rvs(a=self.a, scale=self.b, size=size)
        return np.zeros(size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prior":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =====================================================================
# MCMC Sampler
# =====================================================================


class MCMCSampler:
    """
    Metropolis-within-Gibbs MCMC sampler for Bayesian linear regression.

    Model::

        y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2)

    Each coefficient beta_j has its own prior (Normal, Laplace, Student-t,
    Uniform, Cauchy, …).  sigma^2 gets an Inverse-Gamma prior.

    The sampler uses **Gibbs blocks**: each parameter is updated one at a
    time via a Metropolis-Hastings step with an adaptive Normal proposal.

    Parameters
    ----------
    df : DataFrame
        Data.
    """

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.chains: Optional[np.ndarray] = None
        self.param_names: List[str] = []
        self._last_result: Optional[Dict[str, Any]] = None

    def set_data(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    # Core MCMC runner
    # ------------------------------------------------------------------
    def run(
        self,
        features: List[str],
        target: str,
        priors: Dict[str, Prior] = None,
        n_iter: int = 5000,
        n_warmup: int = 1000,
        n_chains: int = 2,
        proposal_scale: float = 0.1,
        thin: int = 1,
        seed: int = None,
    ) -> Dict[str, Any]:
        """
        Run MCMC sampling.

        Parameters
        ----------
        features : list[str]
            Predictor column names.
        target : str
            Response column name.
        priors : dict[str, Prior] or None
            Mapping of parameter name -> Prior.
            Keys: 'intercept', feature names, 'sigma2'.
            None -> default weakly-informative Normal priors + InvGamma for sigma2.
        n_iter : int
            Total iterations **per chain** (including warmup).
        n_warmup : int
            Warmup / burn-in iterations to discard.
        n_chains : int
            Number of independent chains (for R-hat diagnostics).
        proposal_scale : float
            Initial Metropolis proposal std (adapts during warmup).
        thin : int
            Keep every thin-th sample after warmup.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            chains, posterior_mean, posterior_std,
            credible_intervals (95%), param_names, diagnostics,
            acceptance_rate, predictions, residuals, y_actual,
            r2, n_iter, n_warmup, priors_used
        """
        if self.df is None:
            return {"error": "No data loaded"}

        if seed is not None:
            np.random.seed(seed)

        X_df = self.df[features].dropna()
        y = self.df[target].loc[X_df.index].values
        n, p = X_df.shape
        X_raw = X_df.values

        # Standardise features
        x_means = X_raw.mean(axis=0)
        x_stds = X_raw.std(axis=0, ddof=0)
        x_stds[x_stds == 0] = 1.0
        X_std = (X_raw - x_means) / x_stds
        X_mat = np.column_stack([np.ones(n), X_std])
        k = p + 1  # intercept + p features

        param_names = ["Intercept"] + list(features)
        all_params = param_names + ["sigma2"]
        self.param_names = all_params

        # --- Default priors ---
        if priors is None:
            priors = {}
        default_priors = {}
        for name in param_names:
            default_priors[name] = Prior("normal", loc=0.0, scale=10.0)
        default_priors["sigma2"] = Prior("inverse_gamma", a=1.0, b=1.0)
        for name in all_params:
            if name not in priors:
                priors[name] = default_priors[name]

        # --- OLS initialisation (standardised space) ---
        beta_ols = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        resid = y - X_mat @ beta_ols
        sigma2_ols = float(np.var(resid, ddof=k))

        # --- Run chains ---
        all_chains = []
        all_accept = []
        for chain_id in range(n_chains):
            chain, accept_rate = self._run_single_chain(
                X_mat, y, k, priors, param_names,
                beta_init=beta_ols + np.random.normal(0, 0.1, k),
                sigma2_init=max(sigma2_ols * np.random.uniform(0.5, 1.5), 0.01),
                n_iter=n_iter,
                n_warmup=n_warmup,
                proposal_scale=proposal_scale,
                thin=thin,
            )
            all_chains.append(chain)
            all_accept.append(accept_rate)

        # chains shape: (n_chains, n_post_warmup, k+1)
        chains = np.array(all_chains)
        self.chains = chains

        # --- Transform beta samples back to original scale ---
        # T maps standardised betas -> original scale
        T = np.zeros((k, k))
        T[0, 0] = 1.0
        for j in range(p):
            T[0, j + 1] = -x_means[j] / x_stds[j]
            T[j + 1, j + 1] = 1.0 / x_stds[j]

        chains_orig = np.copy(chains)
        for c in range(n_chains):
            beta_samples = chains[c, :, :k]  # (n_samples, k)
            chains_orig[c, :, :k] = (T @ beta_samples.T).T

        # --- Aggregate posterior ---
        combined = chains_orig.reshape(-1, k + 1)  # merge chains
        post_mean = combined.mean(axis=0)
        post_std = combined.std(axis=0)
        ci = np.percentile(combined, [2.5, 97.5], axis=0)

        # --- Predictions (original scale) ---
        X_orig = np.column_stack([np.ones(n), X_raw])
        beta_post_mean = post_mean[:k]
        y_hat = X_orig @ beta_post_mean
        residuals = y - y_hat
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # --- Diagnostics ---
        diagnostics = self.compute_diagnostics(chains_orig)

        priors_used = {name: priors[name].to_dict() for name in all_params}

        self._last_result = {
            "chains": chains_orig,
            "posterior_mean": post_mean,
            "posterior_std": post_std,
            "credible_intervals_lower": ci[0],
            "credible_intervals_upper": ci[1],
            "param_names": all_params,
            "feature_names": param_names,
            "diagnostics": diagnostics,
            "acceptance_rate": float(np.mean(all_accept)),
            "predictions": y_hat,
            "residuals": residuals,
            "y_actual": y,
            "r2": r2,
            "n_iter": n_iter,
            "n_warmup": n_warmup,
            "n_chains": n_chains,
            "priors_used": priors_used,
            "noise_variance": float(post_mean[-1]),
        }
        return self._last_result

    # ------------------------------------------------------------------
    def _run_single_chain(
        self,
        X_mat: np.ndarray,
        y: np.ndarray,
        k: int,
        priors: Dict[str, Prior],
        param_names: List[str],
        beta_init: np.ndarray,
        sigma2_init: float,
        n_iter: int,
        n_warmup: int,
        proposal_scale: float,
        thin: int,
    ) -> tuple:
        """Run one MCMC chain via Metropolis-within-Gibbs."""
        n = len(y)
        n_params = k + 1  # betas + sigma2
        total_accept = np.zeros(n_params)
        total_attempts = np.zeros(n_params)

        # Adaptive proposal scales per parameter
        prop_scales = np.full(n_params, proposal_scale)
        # sigma2 proposal on log-scale → use wider initial
        prop_scales[-1] = 0.2

        # Current state
        beta = beta_init.copy()
        sigma2 = sigma2_init

        # Pre-compute X'X and X'y
        XtX = X_mat.T @ X_mat
        Xty = X_mat.T @ y

        # Storage for post-warmup samples
        n_keep = max((n_iter - n_warmup) // thin, 1)
        samples = np.zeros((n_keep, n_params))
        sample_idx = 0

        def log_likelihood(beta_vec, sig2):
            """Gaussian log-likelihood."""
            resid = y - X_mat @ beta_vec
            return -0.5 * n * np.log(2 * np.pi * sig2) - 0.5 * np.sum(resid**2) / sig2

        def log_prior_all(beta_vec, sig2):
            """Sum of log-priors for all parameters."""
            lp = 0.0
            for j, name in enumerate(param_names):
                lp += priors[name].log_pdf(beta_vec[j])
                if not np.isfinite(lp):
                    return -np.inf
            lp += priors["sigma2"].log_pdf(sig2)
            return lp

        # Initial log-posterior
        current_ll = log_likelihood(beta, sigma2)
        current_lp = log_prior_all(beta, sigma2)
        current_log_post = current_ll + current_lp

        for it in range(n_iter):
            # --- Update each beta_j via Metropolis ---
            for j in range(k):
                beta_prop = beta.copy()
                beta_prop[j] += np.random.normal(0, prop_scales[j])

                lp_prop = priors[param_names[j]].log_pdf(beta_prop[j])
                if not np.isfinite(lp_prop):
                    total_attempts[j] += 1
                    continue

                ll_prop = log_likelihood(beta_prop, sigma2)
                log_post_prop = ll_prop + log_prior_all(beta_prop, sigma2)

                log_alpha = log_post_prop - current_log_post
                if np.log(np.random.uniform()) < log_alpha:
                    beta = beta_prop
                    current_ll = ll_prop
                    current_lp = log_prior_all(beta, sigma2)
                    current_log_post = log_post_prop
                    total_accept[j] += 1
                total_attempts[j] += 1

            # --- Update sigma2 via Metropolis (log-scale proposal) ---
            log_sig2_prop = np.log(sigma2) + np.random.normal(0, prop_scales[-1])
            sig2_prop = np.exp(log_sig2_prop)

            lp_sig2_prop = priors["sigma2"].log_pdf(sig2_prop)
            if np.isfinite(lp_sig2_prop):
                ll_prop = log_likelihood(beta, sig2_prop)
                # Jacobian correction for log-scale proposal
                log_post_prop = (
                    ll_prop
                    + log_prior_all(beta, sig2_prop)
                    + log_sig2_prop  # Jacobian: d(sig2)/d(log_sig2) = sig2
                )
                log_post_curr_adj = current_log_post + np.log(sigma2)

                log_alpha = log_post_prop - log_post_curr_adj
                if np.log(np.random.uniform()) < log_alpha:
                    sigma2 = sig2_prop
                    current_ll = ll_prop
                    current_lp = log_prior_all(beta, sigma2)
                    current_log_post = current_ll + current_lp
                    total_accept[-1] += 1
            total_attempts[-1] += 1

            # --- Adapt proposal during warmup ---
            if it < n_warmup and it > 0 and it % 100 == 0:
                for j in range(n_params):
                    if total_attempts[j] > 0:
                        rate = total_accept[j] / total_attempts[j]
                        # Target acceptance rate ~0.234 for multi-dim
                        if rate < 0.15:
                            prop_scales[j] *= 0.7
                        elif rate > 0.5:
                            prop_scales[j] *= 1.3

            # --- Store sample ---
            if it >= n_warmup and (it - n_warmup) % thin == 0:
                if sample_idx < n_keep:
                    samples[sample_idx, :k] = beta
                    samples[sample_idx, k] = sigma2
                    sample_idx += 1

        # Trim if fewer samples stored
        samples = samples[:sample_idx]

        accept_rate = np.where(
            total_attempts > 0, total_accept / total_attempts, 0.0
        )
        return samples, accept_rate.mean()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_diagnostics(chains: np.ndarray) -> Dict[str, Any]:
        """
        Compute MCMC convergence diagnostics.

        Parameters
        ----------
        chains : ndarray of shape (n_chains, n_samples, n_params)

        Returns
        -------
        dict with r_hat (per param), ess (per param), warnings
        """
        n_chains, n_samples, n_params = chains.shape

        r_hat = np.ones(n_params)
        ess = np.full(n_params, float(n_chains * n_samples))

        if n_chains >= 2 and n_samples > 1:
            for j in range(n_params):
                chain_means = np.array(
                    [chains[c, :, j].mean() for c in range(n_chains)]
                )
                chain_vars = np.array(
                    [chains[c, :, j].var(ddof=1) for c in range(n_chains)]
                )
                overall_mean = chain_means.mean()
                B = n_samples * np.var(chain_means, ddof=1)  # between-chain
                W = np.mean(chain_vars)  # within-chain
                if W > 0:
                    var_hat = (1 - 1 / n_samples) * W + B / n_samples
                    r_hat[j] = np.sqrt(var_hat / W)
                # Effective sample size (simple estimate)
                ess[j] = MCMCSampler._estimate_ess(chains[:, :, j])

        diag_warnings = []
        bad_rhat = np.where(r_hat > 1.1)[0]
        if len(bad_rhat) > 0:
            diag_warnings.append(
                f"R-hat > 1.1 for {len(bad_rhat)} parameter(s) — chains may not have converged"
            )
        low_ess = np.where(ess < 100)[0]
        if len(low_ess) > 0:
            diag_warnings.append(
                f"ESS < 100 for {len(low_ess)} parameter(s) — consider more iterations"
            )

        return {
            "r_hat": r_hat,
            "ess": ess,
            "warnings": diag_warnings,
            "converged": len(diag_warnings) == 0,
        }

    @staticmethod
    def _estimate_ess(chains_param: np.ndarray) -> float:
        """
        Estimate effective sample size for one parameter across chains.

        Uses the initial monotone sequence estimator (simplified).
        chains_param : (n_chains, n_samples)
        """
        combined = chains_param.ravel()
        n = len(combined)
        if n < 4:
            return float(n)
        mean_val = combined.mean()
        centered = combined - mean_val
        var_val = np.var(centered, ddof=1)
        if var_val < 1e-15:
            return float(n)

        # Autocorrelation via FFT
        fft_len = 2 ** int(np.ceil(np.log2(2 * n)))
        fft_c = np.fft.rfft(centered, n=fft_len)
        acf_full = np.fft.irfft(fft_c * np.conj(fft_c))[:n]
        acf = acf_full / acf_full[0]

        # Sum autocorrelations until they go negative (initial positive sequence)
        tau = 1.0
        for lag in range(1, n // 2):
            if acf[lag] < 0:
                break
            tau += 2 * acf[lag]

        return max(float(n / tau), 1.0)

    # ------------------------------------------------------------------
    # Prior file I/O (extended format with distribution types)
    # ------------------------------------------------------------------
    @staticmethod
    def load_mcmc_prior(path: Union[str, Path]) -> Dict[str, Prior]:
        """
        Load MCMC-compatible prior specification from a JSON file.

        Extended format adds ``"distribution"`` per parameter::

            {
                "name": "Sparse Laplace prior",
                "features": ["x1", "x2"],
                "priors": {
                    "intercept": {"distribution": "normal", "loc": 0, "scale": 10},
                    "x1": {"distribution": "laplace", "loc": 0, "scale": 1},
                    "x2": {"distribution": "laplace", "loc": 0, "scale": 1},
                    "sigma2": {"distribution": "inverse_gamma", "a": 1, "b": 1}
                }
            }
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if "priors" not in obj:
            raise ValueError(f"MCMC prior file {path.name} missing 'priors' mapping")

        result = {}
        for name, spec in obj["priors"].items():
            result[name] = Prior.from_dict(spec)
        return result

    @staticmethod
    def save_mcmc_prior(
        path: Union[str, Path],
        priors: Dict[str, Prior],
        features: List[str] = None,
        target: str = None,
        name: str = "MCMC prior",
        description: str = "",
    ) -> Path:
        """Save MCMC prior configuration to JSON."""
        path = Path(path)
        obj = {
            "name": name,
            "description": description,
            "features": features or [],
            "target": target or "",
            "priors": {k: v.to_dict() for k, v in priors.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return path
