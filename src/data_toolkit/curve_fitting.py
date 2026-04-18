"""
Curve Fitting Module for the Data Analysis Toolkit
====================================================

Provides specialized curve fitting: power/allometric, exponential, logistic,
sinusoidal, Reduced Major Axis (RMA/Type II) regression, and GLM wrappers.

Uses scipy.optimize.curve_fit and statsmodels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import warnings


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def _power_func(x, a, b):
    """Power / allometric: y = a * x^b"""
    return a * np.power(np.abs(x), b)


def _exponential_func(x, a, b):
    """Exponential growth/decay: y = a * exp(b*x)"""
    return a * np.exp(b * x)


def _exponential_decay_func(x, a, b, c):
    """Exponential decay with offset: y = a * exp(-b*x) + c"""
    return a * np.exp(-b * x) + c


def _logistic_func(x, L, k, x0, b):
    """Logistic / sigmoid: y = L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b


def _gompertz_func(x, a, b, c):
    """Gompertz growth: y = a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))


def _sinusoidal_func(x, A, f, phi, offset):
    """Sinusoidal: y = A * sin(2π*f*x + φ) + offset"""
    return A * np.sin(2 * np.pi * f * x + phi) + offset


def _gaussian_func(x, a, mu, sigma):
    """Gaussian: y = a * exp(-(x-mu)²/(2σ²))"""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def _compute_fit_stats(x, y, y_pred, n_params):
    """Compute standard fit statistics."""
    n = len(y)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r_squared
    rmse = np.sqrt(ss_res / n)
    aic = n * np.log(ss_res / n) + 2 * n_params if n > 0 and ss_res > 0 else float('inf')
    bic = n * np.log(ss_res / n) + n_params * np.log(n) if n > 0 and ss_res > 0 else float('inf')

    return {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'ss_residual': ss_res,
        'ss_total': ss_tot,
        'n': n,
        'residuals': residuals,
    }


def power_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit power / allometric model: y = a * x^b

    Args:
        x, y: Input data arrays

    Returns:
        Dictionary with 'a', 'b', 'equation', fit stats, 'y_pred', 'x_fit', 'y_fit'
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 positive data points for power fit")

    # Initial guess via log-log linear regression
    log_x, log_y = np.log(x_clean), np.log(y_clean)
    slope, intercept, _, _, _ = scipy_stats.linregress(log_x, log_y)
    p0 = [np.exp(intercept), slope]

    try:
        popt, pcov = curve_fit(_power_func, x_clean, y_clean, p0=p0, maxfev=10000)
    except RuntimeError:
        # Fallback to log-log OLS
        popt = p0
        pcov = np.full((2, 2), np.nan)

    a, b = popt
    y_pred = _power_func(x_clean, a, b)
    stats = _compute_fit_stats(x_clean, y_clean, y_pred, 2)

    # Smooth curve for plotting
    x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_fit = _power_func(x_fit, a, b)

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan]

    return {
        'a': a, 'b': b,
        'a_se': perr[0], 'b_se': perr[1],
        'equation': f'y = {a:.4g} · x^{b:.4g}',
        'model': 'power',
        'x_data': x_clean, 'y_data': y_clean,
        'y_pred': y_pred,
        'x_fit': x_fit, 'y_fit': y_fit,
        **stats
    }


def exponential_fit(x: np.ndarray, y: np.ndarray,
                    with_offset: bool = False) -> Dict[str, Any]:
    """
    Fit exponential model: y = a * exp(b*x) [or y = a * exp(-b*x) + c]

    Args:
        x, y: Input data arrays
        with_offset: If True, fit 3-parameter decay model

    Returns:
        Dictionary with parameters and fit statistics
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 data points")

    if with_offset:
        # 3-parameter: y = a * exp(-b*x) + c
        p0 = [y_clean.max() - y_clean.min(), 0.1, y_clean.min()]
        try:
            popt, pcov = curve_fit(_exponential_decay_func, x_clean, y_clean, p0=p0, maxfev=10000)
        except RuntimeError:
            raise ValueError("Exponential decay fit did not converge")

        a, b, c = popt
        y_pred = _exponential_decay_func(x_clean, a, b, c)
        stats = _compute_fit_stats(x_clean, y_clean, y_pred, 3)
        x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
        y_fit = _exponential_decay_func(x_fit, a, b, c)
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 3

        return {
            'a': a, 'b': b, 'c': c,
            'a_se': perr[0], 'b_se': perr[1], 'c_se': perr[2],
            'equation': f'y = {a:.4g} · exp(-{b:.4g}·x) + {c:.4g}',
            'model': 'exponential_decay',
            'x_data': x_clean, 'y_data': y_clean,
            'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
            **stats
        }
    else:
        # 2-parameter: y = a * exp(b*x)
        # Filter positive y for log transform initial guess
        pos_mask = y_clean > 0
        if pos_mask.sum() >= 2:
            log_y = np.log(y_clean[pos_mask])
            slope, intercept, _, _, _ = scipy_stats.linregress(x_clean[pos_mask], log_y)
            p0 = [np.exp(intercept), slope]
        else:
            p0 = [1.0, 0.01]

        try:
            popt, pcov = curve_fit(_exponential_func, x_clean, y_clean, p0=p0, maxfev=10000)
        except RuntimeError:
            raise ValueError("Exponential fit did not converge")

        a, b = popt
        y_pred = _exponential_func(x_clean, a, b)
        stats = _compute_fit_stats(x_clean, y_clean, y_pred, 2)
        x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
        y_fit = _exponential_func(x_fit, a, b)
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan, np.nan]

        return {
            'a': a, 'b': b,
            'a_se': perr[0], 'b_se': perr[1],
            'equation': f'y = {a:.4g} · exp({b:.4g}·x)',
            'model': 'exponential',
            'x_data': x_clean, 'y_data': y_clean,
            'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
            **stats
        }


def logistic_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit logistic / sigmoid curve: y = L / (1 + exp(-k*(x - x0))) + b

    Args:
        x, y: Input data arrays

    Returns:
        Dictionary with L, k, x0, b parameters and fit statistics
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 4:
        raise ValueError("Need at least 4 data points for logistic fit")

    # Initial guesses
    L0 = y_clean.max() - y_clean.min()
    k0 = 1.0
    x0_0 = x_clean.mean()
    b0 = y_clean.min()

    try:
        popt, pcov = curve_fit(
            _logistic_func, x_clean, y_clean,
            p0=[L0, k0, x0_0, b0],
            maxfev=20000
        )
    except RuntimeError:
        raise ValueError("Logistic fit did not converge")

    L, k, x0, b = popt
    y_pred = _logistic_func(x_clean, L, k, x0, b)
    stats = _compute_fit_stats(x_clean, y_clean, y_pred, 4)
    x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_fit = _logistic_func(x_fit, L, k, x0, b)
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 4

    return {
        'L': L, 'k': k, 'x0': x0, 'b': b,
        'L_se': perr[0], 'k_se': perr[1], 'x0_se': perr[2], 'b_se': perr[3],
        'equation': f'y = {L:.4g} / (1 + exp(-{k:.4g}·(x - {x0:.4g}))) + {b:.4g}',
        'model': 'logistic',
        'x_data': x_clean, 'y_data': y_clean,
        'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
        **stats
    }


def sinusoidal_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit sinusoidal model: y = A * sin(2π*f*x + φ) + offset

    Uses FFT for initial frequency estimate.

    Args:
        x, y: Input data arrays

    Returns:
        Dictionary with A, f, phi, offset parameters and fit statistics
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 4:
        raise ValueError("Need at least 4 data points for sinusoidal fit")

    # Initial guesses via FFT
    offset0 = y_clean.mean()
    A0 = (y_clean.max() - y_clean.min()) / 2

    # Estimate frequency from FFT
    n = len(x_clean)
    dx = np.median(np.diff(np.sort(x_clean)))
    if dx > 0:
        fft_vals = np.fft.rfft(y_clean - offset0)
        fft_freqs = np.fft.rfftfreq(n, d=dx)
        # Skip DC component
        magnitudes = np.abs(fft_vals[1:])
        if len(magnitudes) > 0:
            peak_idx = np.argmax(magnitudes) + 1
            f0 = fft_freqs[peak_idx]
        else:
            f0 = 1.0 / (x_clean.max() - x_clean.min())
    else:
        f0 = 0.1

    phi0 = 0.0

    try:
        popt, pcov = curve_fit(
            _sinusoidal_func, x_clean, y_clean,
            p0=[A0, f0, phi0, offset0],
            maxfev=20000
        )
    except RuntimeError:
        raise ValueError("Sinusoidal fit did not converge")

    A, f, phi, offset = popt
    y_pred = _sinusoidal_func(x_clean, A, f, phi, offset)
    stats = _compute_fit_stats(x_clean, y_clean, y_pred, 4)
    x_fit = np.linspace(x_clean.min(), x_clean.max(), 500)
    y_fit = _sinusoidal_func(x_fit, A, f, phi, offset)
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 4

    # Period
    period = 1.0 / abs(f) if f != 0 else float('inf')

    return {
        'amplitude': abs(A), 'frequency': abs(f), 'phase': phi, 'offset': offset,
        'period': period,
        'A_se': perr[0], 'f_se': perr[1], 'phi_se': perr[2], 'offset_se': perr[3],
        'equation': f'y = {abs(A):.4g} · sin(2π·{abs(f):.4g}·x + {phi:.4g}) + {offset:.4g}',
        'model': 'sinusoidal',
        'x_data': x_clean, 'y_data': y_clean,
        'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
        **stats
    }


def gompertz_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Fit Gompertz growth curve: y = a * exp(-b * exp(-c * x))

    Args:
        x, y: Input data arrays

    Returns:
        Dictionary with a, b, c parameters and fit statistics
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 data points")

    a0 = y_clean.max()
    b0 = 5.0
    c0 = 0.1

    try:
        popt, pcov = curve_fit(
            _gompertz_func, x_clean, y_clean,
            p0=[a0, b0, c0], maxfev=20000
        )
    except RuntimeError:
        raise ValueError("Gompertz fit did not converge")

    a, b, c = popt
    y_pred = _gompertz_func(x_clean, a, b, c)
    stats = _compute_fit_stats(x_clean, y_clean, y_pred, 3)
    x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_fit = _gompertz_func(x_fit, a, b, c)
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 3

    return {
        'a': a, 'b': b, 'c': c,
        'a_se': perr[0], 'b_se': perr[1], 'c_se': perr[2],
        'equation': f'y = {a:.4g} · exp(-{b:.4g} · exp(-{c:.4g}·x))',
        'model': 'gompertz',
        'x_data': x_clean, 'y_data': y_clean,
        'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
        **stats
    }


# ============================================================================
# REDUCED MAJOR AXIS (TYPE II) REGRESSION
# ============================================================================

def rma_regression(x: np.ndarray, y: np.ndarray,
                   confidence: float = 0.95) -> Dict[str, Any]:
    """
    Reduced Major Axis (RMA) / Type II regression.

    Minimizes the product of vertical and horizontal deviations.
    Appropriate when both X and Y have measurement error.

    slope_RMA = sign(r) * (SD_y / SD_x)
    intercept_RMA = mean_y - slope * mean_x

    Args:
        x, y: Input data arrays
        confidence: Confidence level for intervals

    Returns:
        Dictionary with slope, intercept, r, CI, fit statistics
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)

    if n < 3:
        raise ValueError("Need at least 3 data points")

    mean_x, mean_y = x_clean.mean(), y_clean.mean()
    sd_x, sd_y = x_clean.std(ddof=1), y_clean.std(ddof=1)
    r = np.corrcoef(x_clean, y_clean)[0, 1]

    if sd_x == 0:
        raise ValueError("X has zero variance")

    # RMA slope and intercept
    slope = np.sign(r) * sd_y / sd_x
    intercept = mean_y - slope * mean_x

    y_pred = slope * x_clean + intercept
    stats = _compute_fit_stats(x_clean, y_clean, y_pred, 2)

    # Confidence intervals for slope (approximate)
    # Based on Jolicoeur (1990) method
    B = (1 - r ** 2) / (n - 2)
    t_val = scipy_stats.t.ppf(1 - (1 - confidence) / 2, n - 2)

    slope_se = abs(slope) * np.sqrt(B)
    slope_ci = (slope - t_val * slope_se, slope + t_val * slope_se)

    intercept_se = np.sqrt(sd_y ** 2 / n + slope_se ** 2 * mean_x ** 2)
    intercept_ci = (intercept - t_val * intercept_se, intercept + t_val * intercept_se)

    x_fit = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_fit = slope * x_fit + intercept

    return {
        'slope': slope,
        'intercept': intercept,
        'slope_se': slope_se,
        'intercept_se': intercept_se,
        'slope_ci': slope_ci,
        'intercept_ci': intercept_ci,
        'r': r,
        'r_squared': r ** 2,
        'equation': f'y = {slope:.4g}·x + {intercept:.4g}',
        'model': 'rma',
        'x_data': x_clean, 'y_data': y_clean,
        'y_pred': y_pred, 'x_fit': x_fit, 'y_fit': y_fit,
        'confidence': confidence,
        **stats
    }


# ============================================================================
# GENERALIZED LINEAR MODELS (GLM)
# ============================================================================

def glm_fit(X: np.ndarray, y: np.ndarray,
            family: str = 'gaussian',
            link: str = 'identity') -> Dict[str, Any]:
    """
    Generalized Linear Model (GLM) via statsmodels.

    Args:
        X: Predictor matrix (n × p), without intercept
        y: Response vector (length n)
        family: 'gaussian', 'binomial', 'poisson', 'gamma', 'inverse_gaussian'
        link: 'identity', 'log', 'logit', 'inverse', 'probit' (auto-selected if None)

    Returns:
        Dictionary with coefficients, SE, p-values, deviance, AIC, predictions
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for GLM. Install with: pip install statsmodels")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Add constant
    X_with_const = sm.add_constant(X)

    # Select family
    family_map = {
        'gaussian': sm.families.Gaussian(),
        'binomial': sm.families.Binomial(),
        'poisson': sm.families.Poisson(),
        'gamma': sm.families.Gamma(),
        'inverse_gaussian': sm.families.InverseGaussian(),
    }

    if family not in family_map:
        raise ValueError(f"Unknown family: {family}. Choose from {list(family_map.keys())}")

    fam = family_map[family]

    model = sm.GLM(y, X_with_const, family=fam)
    result = model.fit()

    return {
        'coefficients': result.params.tolist(),
        'standard_errors': result.bse.tolist(),
        'z_values': result.tvalues.tolist(),
        'p_values': result.pvalues.tolist(),
        'confidence_intervals': result.conf_int().tolist(),
        'deviance': result.deviance,
        'null_deviance': result.null_deviance,
        'aic': result.aic,
        'bic': result.bic,
        'log_likelihood': result.llf,
        'df_residual': int(result.df_resid),
        'df_model': int(result.df_model),
        'predictions': result.predict(X_with_const).tolist(),
        'residuals_deviance': result.resid_deviance.tolist(),
        'summary': str(result.summary()),
        'family': family,
    }


# ============================================================================
# COMPARE MULTIPLE FITS
# ============================================================================

def compare_fits(x: np.ndarray, y: np.ndarray,
                 models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fit multiple models and compare using AIC/BIC/R².

    Args:
        x, y: Input data
        models: List of model names to try. Default: all available.

    Returns:
        DataFrame comparing fits, sorted by AIC
    """
    if models is None:
        models = ['power', 'exponential', 'logistic', 'sinusoidal', 'gompertz', 'rma']

    fit_funcs = {
        'power': power_fit,
        'exponential': exponential_fit,
        'logistic': logistic_fit,
        'sinusoidal': sinusoidal_fit,
        'gompertz': gompertz_fit,
        'rma': rma_regression,
    }

    results = []
    for name in models:
        if name not in fit_funcs:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fit_funcs[name](x, y)
            results.append({
                'Model': name,
                'Equation': result.get('equation', ''),
                'R²': result.get('r_squared', np.nan),
                'Adj R²': result.get('adj_r_squared', np.nan),
                'RMSE': result.get('rmse', np.nan),
                'AIC': result.get('aic', np.nan),
                'BIC': result.get('bic', np.nan),
            })
        except (ValueError, RuntimeError):
            results.append({
                'Model': name,
                'Equation': 'Failed to converge',
                'R²': np.nan,
                'Adj R²': np.nan,
                'RMSE': np.nan,
                'AIC': np.nan,
                'BIC': np.nan,
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('AIC', ascending=True, na_position='last').reset_index(drop=True)
    return df
