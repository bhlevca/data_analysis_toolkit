"""
Rust Accelerated Functions
==========================

This module provides Python wrappers for Rust-accelerated functions.
If the Rust extension is not installed, it falls back to pure Python implementations.

Features:
- Toggle between Rust and Python implementations at runtime
- Automatic fallback if Rust isn't compiled
- Benchmark utilities to compare performance

Installation (optional, for Rust acceleration):
    cd rust_extensions
    pip install maturin
    maturin develop --release
"""

import warnings
from typing import Optional, Tuple

import numpy as np

# Try to import Rust extension
try:
    import data_toolkit_rust as _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _rust = None

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

class AccelerationSettings:
    """Global settings for Rust/Python acceleration."""

    _use_rust: bool = True  # Default to using Rust if available
    _listeners: list = []   # Callbacks when setting changes

    @classmethod
    def use_rust(cls) -> bool:
        """Check if Rust acceleration should be used."""
        return cls._use_rust and RUST_AVAILABLE

    @classmethod
    def set_use_rust(cls, value: bool) -> None:
        """Enable or disable Rust acceleration."""
        cls._use_rust = value
        # Notify listeners
        for callback in cls._listeners:
            try:
                callback(value)
            except:
                pass

    @classmethod
    def toggle(cls) -> bool:
        """Toggle Rust acceleration on/off. Returns new state."""
        cls.set_use_rust(not cls._use_rust)
        return cls._use_rust

    @classmethod
    def add_listener(cls, callback) -> None:
        """Add a callback to be notified when setting changes."""
        cls._listeners.append(callback)

    @classmethod
    def remove_listener(cls, callback) -> None:
        """Remove a listener callback."""
        if callback in cls._listeners:
            cls._listeners.remove(callback)

    @classmethod
    def get_status(cls) -> dict:
        """Get current acceleration status."""
        return {
            'rust_compiled': RUST_AVAILABLE,
            'rust_enabled': cls._use_rust,
            'active_backend': 'Rust' if cls.use_rust() else 'Python',
        }


def is_rust_available() -> bool:
    """Check if Rust extensions are compiled and available."""
    return RUST_AVAILABLE


def is_rust_enabled() -> bool:
    """Check if Rust acceleration is currently enabled."""
    return AccelerationSettings.use_rust()


def set_rust_enabled(enabled: bool) -> None:
    """Enable or disable Rust acceleration."""
    AccelerationSettings.set_use_rust(enabled)


def get_backend_name() -> str:
    """Get the name of the currently active backend."""
    return 'Rust' if AccelerationSettings.use_rust() else 'Python'


# ============================================================================
# DISTANCE CORRELATION
# ============================================================================

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate distance correlation between two arrays.

    Distance correlation can detect non-linear relationships, unlike Pearson.
    Returns a value between 0 and 1.

    Args:
        x: First array
        y: Second array

    Returns:
        Distance correlation coefficient
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return _rust.distance_correlation(x, y)
    else:
        return _python_distance_correlation(x, y)


def _python_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pure Python implementation of distance correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    # Distance matrices
    a = np.abs(x[:, None] - x)
    b = np.abs(y[:, None] - y)

    # Double centering
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

    # Distance covariances
    dcov_xy = (A * B).sum() / (n * n)
    dcov_xx = (A * A).sum() / (n * n)
    dcov_yy = (B * B).sum() / (n * n)

    denom = np.sqrt(np.sqrt(dcov_xx) * np.sqrt(dcov_yy))
    if denom == 0:
        return 0.0

    return np.sqrt(max(0, dcov_xy)) / denom


def distance_correlation_matrix(
    features: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Calculate distance correlation for multiple features against a target.

    Args:
        features: 2D array (n_samples, n_features)
        target: 1D array (n_samples,)

    Returns:
        1D array of distance correlations
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return np.asarray(_rust.distance_correlation_matrix(features, target))
    else:
        n_features = features.shape[1]
        return np.array([
            _python_distance_correlation(features[:, i], target)
            for i in range(n_features)
        ])


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform bootstrap for linear regression coefficient confidence intervals.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0-1)

    Returns:
        Tuple of (mean_coefficients, ci_lower, ci_upper)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return tuple(np.asarray(arr) for arr in
                    _rust.bootstrap_linear_regression(X, y, n_bootstrap, confidence))
    else:
        return _python_bootstrap_linear_regression(X, y, n_bootstrap, confidence)


def _python_bootstrap_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python bootstrap implementation."""
    from sklearn.linear_model import LinearRegression

    n_samples, n_features = X.shape
    bootstrap_coefs = np.zeros((n_bootstrap, n_features))

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        model = LinearRegression()
        model.fit(X[indices], y[indices])
        bootstrap_coefs[i] = model.coef_

    alpha = (1 - confidence) / 2
    mean_coefs = bootstrap_coefs.mean(axis=0)
    ci_lower = np.percentile(bootstrap_coefs, alpha * 100, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, (1 - alpha) * 100, axis=0)

    return mean_coefs, ci_lower, ci_upper


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_predictions(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    residual_std: float,
    n_simulations: int = 1000,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation for prediction uncertainty.

    Args:
        X: Feature matrix
        coefficients: Model coefficients
        intercept: Model intercept
        residual_std: Standard deviation of residuals
        n_simulations: Number of simulations
        confidence: Confidence level

    Returns:
        Tuple of (mean_predictions, ci_lower, ci_upper)
    """
    X = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return tuple(np.asarray(arr) for arr in
                    _rust.monte_carlo_predictions(
                        X, coefficients, intercept, residual_std,
                        n_simulations, confidence
                    ))
    else:
        return _python_monte_carlo_predictions(
            X, coefficients, intercept, residual_std, n_simulations, confidence
        )


def _python_monte_carlo_predictions(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    residual_std: float,
    n_simulations: int,
    confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python Monte Carlo implementation."""
    n_samples = X.shape[0]
    predictions = np.zeros((n_simulations, n_samples))

    for i in range(n_simulations):
        noisy_coef = coefficients + np.random.randn(len(coefficients)) * residual_std / 10
        noisy_intercept = intercept + np.random.randn() * residual_std
        predictions[i] = X @ noisy_coef + noisy_intercept + np.random.randn(n_samples) * residual_std

    alpha = (1 - confidence) / 2
    mean_pred = predictions.mean(axis=0)
    ci_lower = np.percentile(predictions, alpha * 100, axis=0)
    ci_upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)

    return mean_pred, ci_lower, ci_upper


# ============================================================================
# TRANSFER ENTROPY
# ============================================================================

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
    lag: int = 1
) -> float:
    """
    Calculate transfer entropy from source to target.

    Transfer entropy measures directed information flow.

    Args:
        source: Source time series
        target: Target time series
        n_bins: Number of bins for discretization
        lag: Time lag

    Returns:
        Transfer entropy value (>= 0)
    """
    source = np.asarray(source, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return _rust.transfer_entropy(source, target, n_bins, lag)
    else:
        return _python_transfer_entropy(source, target, n_bins, lag)


def _python_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    lag: int
) -> float:
    """Pure Python transfer entropy implementation."""
    n = len(source)
    if n <= lag:
        return 0.0

    # Bin the data
    def bin_data(data):
        min_val, max_val = data.min(), data.max()
        if max_val == min_val:
            return np.zeros(len(data), dtype=int)
        return np.clip(((data - min_val) / (max_val - min_val) * n_bins).astype(int), 0, n_bins - 1)

    source_binned = bin_data(source)
    target_binned = bin_data(target)

    # Count probabilities using numpy
    from collections import Counter

    count = n - lag
    p_y_ypast_xpast = Counter()
    p_y_ypast = Counter()
    p_ypast_xpast = Counter()
    p_ypast = Counter()

    for i in range(lag, n):
        y = target_binned[i]
        y_past = target_binned[i - lag]
        x_past = source_binned[i - lag]

        p_y_ypast_xpast[(y, y_past, x_past)] += 1
        p_y_ypast[(y, y_past)] += 1
        p_ypast_xpast[(y_past, x_past)] += 1
        p_ypast[y_past] += 1

    # Calculate transfer entropy
    te = 0.0
    for (y, y_past, x_past), joint_count in p_y_ypast_xpast.items():
        p_joint = joint_count / count
        p_yy = p_y_ypast[(y, y_past)] / count
        p_yx = p_ypast_xpast[(y_past, x_past)] / count
        p_y_only = p_ypast[y_past] / count

        if p_joint > 0 and p_yy > 0 and p_yx > 0 and p_y_only > 0:
            ratio = (p_joint * p_y_only) / (p_yy * p_yx)
            if ratio > 0:
                te += p_joint * np.log(ratio)

    return max(0.0, te)


# ============================================================================
# LEAD-LAG CORRELATION
# ============================================================================

def lead_lag_correlations(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correlations at multiple lags.

    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag (positive and negative)

    Returns:
        Tuple of (lags, correlations)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        lags, corrs = _rust.lead_lag_correlations(x, y, max_lag)
        return np.asarray(lags), np.asarray(corrs)
    else:
        return _python_lead_lag_correlations(x, y, max_lag)


def _python_lead_lag_correlations(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python lead-lag implementation."""
    from scipy.stats import pearsonr

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros(len(lags))
    n = min(len(x), len(y))

    for i, lag in enumerate(lags):
        try:
            if lag == 0:
                corr, _ = pearsonr(x[:n], y[:n])
            elif lag > 0:
                corr, _ = pearsonr(x[:n-lag], y[lag:n])
            else:
                corr, _ = pearsonr(x[-lag:n], y[:n+lag])
            correlations[i] = corr
        except:
            correlations[i] = 0.0

    return lags, correlations


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using IQR method for multiple columns.

    Args:
        data: 2D array (n_samples, n_columns)
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Tuple of (counts, percentages) for each column
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if AccelerationSettings.use_rust():
        counts, pcts = _rust.detect_outliers_iqr(data, multiplier)
        return np.asarray(counts), np.asarray(pcts)
    else:
        return _python_detect_outliers_iqr(data, multiplier)


def _python_detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python IQR outlier detection."""
    n_cols = data.shape[1]
    counts = np.zeros(n_cols, dtype=int)
    percentages = np.zeros(n_cols)

    for j in range(n_cols):
        col = data[:, j]
        col = col[~np.isnan(col)]

        if len(col) < 4:
            continue

        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers = (col < lower) | (col > upper)
        counts[j] = outliers.sum()
        percentages[j] = (counts[j] / len(col)) * 100

    return counts, percentages


# ============================================================================
# MUTUAL INFORMATION
# ============================================================================

def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Estimate mutual information using binning.

    Args:
        x: First array
        y: Second array
        n_bins: Number of bins

    Returns:
        Mutual information estimate
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return _rust.mutual_information(x, y, n_bins)
    else:
        return _python_mutual_information(x, y, n_bins)


def _python_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Pure Python mutual information implementation."""
    from collections import Counter

    n = min(len(x), len(y))

    # Bin the data
    def bin_data(data):
        min_val, max_val = data.min(), data.max()
        if max_val == min_val:
            return np.zeros(len(data), dtype=int)
        return np.clip(((data - min_val) / (max_val - min_val) * n_bins).astype(int), 0, n_bins - 1)

    x_binned = bin_data(x[:n])
    y_binned = bin_data(y[:n])

    # Count probabilities
    p_xy = Counter(zip(x_binned, y_binned))
    p_x = Counter(x_binned)
    p_y = Counter(y_binned)

    mi = 0.0
    for (xi, yi), joint_count in p_xy.items():
        p_joint = joint_count / n
        px = p_x[xi] / n
        py = p_y[yi] / n

        if p_joint > 0 and px > 0 and py > 0:
            mi += p_joint * np.log(p_joint / (px * py))

    return max(0.0, mi)


# ============================================================================
# ROLLING STATISTICS
# ============================================================================

def rolling_statistics(
    data: np.ndarray,
    window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rolling mean and standard deviation efficiently.

    Args:
        data: 1D array
        window: Window size

    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        means, stds = _rust.rolling_statistics(data, window)
        return np.asarray(means), np.asarray(stds)
    else:
        return _python_rolling_statistics(data, window)


def _python_rolling_statistics(
    data: np.ndarray,
    window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python rolling statistics using cumsum trick."""
    import pandas as pd
    series = pd.Series(data)
    means = series.rolling(window=window).mean().values
    stds = series.rolling(window=window).std().values
    return means, stds


# ============================================================================
# NEURAL NETWORK PREPROCESSING
# ============================================================================

def create_sequences(
    data: np.ndarray,
    sequence_length: int
) -> np.ndarray:
    """
    Create sliding window sequences for LSTM/RNN from time series data.

    This is a CPU-intensive operation for large datasets.

    Args:
        data: 1D array of time series values
        sequence_length: Number of time steps in each sequence

    Returns:
        2D array of shape (n_samples - sequence_length, sequence_length)
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        return np.asarray(_rust.create_sequences(data, sequence_length))
    else:
        return _python_create_sequences(data, sequence_length)


def _python_create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """Pure Python implementation of sequence creation."""
    n = len(data)
    if sequence_length >= n:
        raise ValueError("Sequence length must be less than data length")

    n_sequences = n - sequence_length
    sequences = np.zeros((n_sequences, sequence_length))

    for i in range(n_sequences):
        sequences[i] = data[i:i + sequence_length]

    return sequences


def create_sequences_with_targets(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with targets for supervised learning (LSTM forecasting).

    Args:
        data: 1D array of time series values
        sequence_length: Number of time steps in each input sequence
        forecast_horizon: Number of time steps to predict

    Returns:
        Tuple of (X, y) where:
        - X has shape (n_samples, sequence_length)
        - y has shape (n_samples, forecast_horizon)
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if AccelerationSettings.use_rust():
        X, y = _rust.create_sequences_with_targets(data, sequence_length, forecast_horizon)
        return np.asarray(X), np.asarray(y)
    else:
        return _python_create_sequences_with_targets(data, sequence_length, forecast_horizon)


def _python_create_sequences_with_targets(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python implementation."""
    n = len(data)
    total_window = sequence_length + forecast_horizon

    if total_window > n:
        raise ValueError("sequence_length + forecast_horizon must be <= data length")

    n_sequences = n - total_window + 1
    X = np.zeros((n_sequences, sequence_length))
    y = np.zeros((n_sequences, forecast_horizon))

    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = data[i + sequence_length:i + total_window]

    return X, y


def batch_minmax_normalize(
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max normalize each column of a 2D array in parallel.

    Args:
        data: 2D array (n_samples, n_features)

    Returns:
        Tuple of (normalized_data, mins, maxs) for inverse transform
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if AccelerationSettings.use_rust():
        norm, mins, maxs = _rust.batch_minmax_normalize(data)
        return np.asarray(norm), np.asarray(mins), np.asarray(maxs)
    else:
        return _python_batch_minmax_normalize(data)


def _python_batch_minmax_normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python min-max normalization."""
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins

    # Avoid division by zero
    ranges[ranges == 0] = 1.0
    normalized = (data - mins) / ranges

    # Center columns with zero range
    zero_range_mask = (maxs == mins)
    if np.any(zero_range_mask):
        normalized[:, zero_range_mask] = 0.5

    return normalized, mins, maxs


def batch_zscore_normalize(
    data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize each column of a 2D array in parallel.

    Args:
        data: 2D array (n_samples, n_features)

    Returns:
        Tuple of (normalized_data, means, stds) for inverse transform
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if AccelerationSettings.use_rust():
        norm, means, stds = _rust.batch_zscore_normalize(data)
        return np.asarray(norm), np.asarray(means), np.asarray(stds)
    else:
        return _python_batch_zscore_normalize(data)


def _python_batch_zscore_normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python z-score normalization."""
    means = data.mean(axis=0)
    stds = data.std(axis=0)

    # Avoid division by zero
    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0
    normalized = (data - means) / stds_safe

    # Zero out columns with zero std
    zero_std_mask = (stds == 0)
    if np.any(zero_std_mask):
        normalized[:, zero_std_mask] = 0.0

    return normalized, means, stds


def batch_mse_errors(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> np.ndarray:
    """
    Calculate per-sample MSE reconstruction errors (for autoencoder).

    Args:
        original: 2D array of original data
        reconstructed: 2D array of reconstructed data (same shape)

    Returns:
        1D array of MSE values for each row
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.ndim == 1:
        original = original.reshape(1, -1)
        reconstructed = reconstructed.reshape(1, -1)

    if AccelerationSettings.use_rust():
        return np.asarray(_rust.batch_mse_errors(original, reconstructed))
    else:
        return np.mean((original - reconstructed) ** 2, axis=1)


def batch_mae_errors(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> np.ndarray:
    """
    Calculate per-sample MAE reconstruction errors.

    Args:
        original: 2D array of original data
        reconstructed: 2D array of reconstructed data (same shape)

    Returns:
        1D array of MAE values for each row
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.ndim == 1:
        original = original.reshape(1, -1)
        reconstructed = reconstructed.reshape(1, -1)

    if AccelerationSettings.use_rust():
        return np.asarray(_rust.batch_mae_errors(original, reconstructed))
    else:
        return np.mean(np.abs(original - reconstructed), axis=1)


# ============================================================================
# BENCHMARK UTILITY
# ============================================================================

def benchmark_rust_vs_python(n_samples: int = 10000) -> dict:
    """
    Benchmark Rust vs Python implementations.

    Args:
        n_samples: Number of samples for benchmark

    Returns:
        Dictionary with timing results
    """
    import time

    # Generate test data
    np.random.seed(42)
    x = np.random.randn(n_samples)
    y = 2 * x + np.random.randn(n_samples) * 0.5
    X = np.random.randn(n_samples, 5)

    results = {}

    # Distance correlation
    start = time.perf_counter()
    _python_distance_correlation(x, y)
    results['distance_correlation_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.distance_correlation(x, y)
        results['distance_correlation_rust'] = time.perf_counter() - start
        results['distance_correlation_speedup'] = (
            results['distance_correlation_python'] /
            results['distance_correlation_rust']
        )

    # Bootstrap (smaller iterations for speed)
    start = time.perf_counter()
    _python_bootstrap_linear_regression(X, y, 100, 0.95)
    results['bootstrap_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.bootstrap_linear_regression(X, y, 100, 0.95)
        results['bootstrap_rust'] = time.perf_counter() - start
        results['bootstrap_speedup'] = (
            results['bootstrap_python'] / results['bootstrap_rust']
        )

    # Lead-lag
    start = time.perf_counter()
    _python_lead_lag_correlations(x, y, 20)
    results['lead_lag_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.lead_lag_correlations(x, y, 20)
        results['lead_lag_rust'] = time.perf_counter() - start
        results['lead_lag_speedup'] = (
            results['lead_lag_python'] / results['lead_lag_rust']
        )

    # Sequence creation (new)
    start = time.perf_counter()
    _python_create_sequences(x, 50)
    results['create_sequences_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.create_sequences(x, 50)
        results['create_sequences_rust'] = time.perf_counter() - start
        results['create_sequences_speedup'] = (
            results['create_sequences_python'] / results['create_sequences_rust']
        )

    # Batch normalization (new)
    start = time.perf_counter()
    _python_batch_minmax_normalize(X)
    results['batch_normalize_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.batch_minmax_normalize(X)
        results['batch_normalize_rust'] = time.perf_counter() - start
        results['batch_normalize_speedup'] = (
            results['batch_normalize_python'] / results['batch_normalize_rust']
        )

    # Batch MSE errors (new)
    X2 = X + np.random.randn(*X.shape) * 0.1
    start = time.perf_counter()
    np.mean((X - X2) ** 2, axis=1)  # Python baseline
    results['batch_mse_python'] = time.perf_counter() - start

    if RUST_AVAILABLE:
        start = time.perf_counter()
        _rust.batch_mse_errors(X, X2)
        results['batch_mse_rust'] = time.perf_counter() - start
        results['batch_mse_speedup'] = (
            results['batch_mse_python'] / results['batch_mse_rust']
        )

    return results


if __name__ == '__main__':
    print(f"Rust available: {RUST_AVAILABLE}")
    print("\nRunning benchmark...")
    results = benchmark_rust_vs_python(5000)

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    for key, value in results.items():
        if 'speedup' in key:
            print(f"{key}: {value:.1f}x faster")
        else:
            print(f"{key}: {value*1000:.2f} ms")
