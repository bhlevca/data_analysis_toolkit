"""
Enhanced Time Series Analysis Module
Contains methods for time series analysis including:
- ACF/PACF analysis
- Stationarity tests
- ARIMA modeling and decomposition
- Fourier Transform Analysis (FFT, Power Spectral Density)
- Wavelet Analysis (CWT, DWT for nonstationary processes)

Version: 2.0
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import chi2

# Try to import pywt for CWT and DWT
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Import accelerated functions - handle both package and direct run
try:
    from .rust_accelerated import (
        rolling_statistics as _accel_rolling_stats,
        AccelerationSettings
    )
except ImportError:
    from rust_accelerated import (
        rolling_statistics as _accel_rolling_stats,
        AccelerationSettings
    )

# Import signal analysis functions
try:
    from . import signal_analysis
except ImportError:
    import signal_analysis


class TimeSeriesAnalysis:
    """Time series analysis methods"""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df

    def acf_analysis(self, column: str, lags: int = 40) -> Dict[str, Any]:
        """
        Calculate autocorrelation function

        Returns:
            Dictionary with ACF values and confidence intervals
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna()

        acf_values = acf(data, nlags=lags, fft=True)

        n = len(data)
        conf_int = 1.96 / np.sqrt(n)

        return {
            'acf': acf_values,
            'lags': np.arange(len(acf_values)),
            'conf_int_upper': conf_int,
            'conf_int_lower': -conf_int,
            'n_obs': n
        }

    def pacf_analysis(self, column: str, lags: int = 40) -> Dict[str, Any]:
        """
        Calculate partial autocorrelation function

        Returns:
            Dictionary with PACF values
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna()
        max_lags = min(lags, len(data) // 2 - 1)

        pacf_values = pacf(data, nlags=max_lags)

        n = len(data)
        conf_int = 1.96 / np.sqrt(n)

        return {
            'pacf': pacf_values,
            'lags': np.arange(len(pacf_values)),
            'conf_int_upper': conf_int,
            'conf_int_lower': -conf_int,
            'n_obs': n
        }

    def stationarity_test(self, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Perform Augmented Dickey-Fuller test for stationarity

        Returns:
            Dictionary with test results for each column
        """
        if self.df is None:
            return {}

        results = {}

        for col in columns:
            data = self.df[col].dropna()

            try:
                adf_result = adfuller(data)

                results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'used_lag': adf_result[2],
                    'n_obs': adf_result[3],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                results[col] = {'error': str(e)}

        return results

    def arima_model(self, column: str, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        Fit ARIMA model

        Args:
            column: Column name for time series
            order: (p, d, q) order for ARIMA

        Returns:
            Dictionary with ARIMA results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna()

        try:
            model = ARIMA(data, order=order)
            results = model.fit()

            fitted_values = results.fittedvalues

            return {
                'order': order,
                'aic': results.aic,
                'bic': results.bic,
                'summary': str(results.summary()),
                'fitted_values': fitted_values.values,
                'original_data': data.values,
                'residuals': results.resid.values,
                'params': dict(results.params)
            }
        except Exception as e:
            return {'error': str(e)}

    def time_decomposition(self, column: str, model: str = 'additive',
                          period: int = None) -> Dict[str, Any]:
        """
        Perform seasonal decomposition

        Args:
            column: Column name
            model: 'additive' or 'multiplicative'
            period: Period for seasonal component

        Returns:
            Dictionary with decomposition components
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna()

        if period is None:
            period = min(12, len(data) // 2)

        try:
            decomposition = seasonal_decompose(data, model=model, period=period)

            return {
                'observed': decomposition.observed.values,
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.values,
                'period': period,
                'model': model
            }
        except Exception as e:
            return {'error': str(e)}

    def rolling_statistics(self, column: str, window: int = 12) -> Dict[str, Any]:
        """
        Calculate rolling mean and standard deviation

        Uses Rust acceleration when enabled for faster computation.

        Returns:
            Dictionary with rolling statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna()

        # Use accelerated rolling statistics (handles Rust/Python switching internally)
        rolling_mean, rolling_std = _accel_rolling_stats(data.values, window)

        return {
            'original': data.values,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'window': window
        }

    def seasonal_strength(self, column: str, period: int = None) -> Dict[str, float]:
        """
        Calculate strength of trend and seasonality

        Returns:
            Dictionary with strength measures
        """
        if self.df is None:
            return {}

        decomp = self.time_decomposition(column, period=period)

        if 'error' in decomp:
            return decomp

        residual = decomp['residual']
        seasonal = decomp['seasonal']
        trend = decomp['trend']

        valid_mask = ~np.isnan(residual) & ~np.isnan(seasonal) & ~np.isnan(trend)

        if valid_mask.sum() < 2:
            return {'error': 'Insufficient valid data'}

        var_resid = np.var(residual[valid_mask])
        var_seasonal_resid = np.var(seasonal[valid_mask] + residual[valid_mask])
        var_trend_resid = np.var(trend[valid_mask] + residual[valid_mask])

        seasonal_strength = max(0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0
        trend_strength = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0

        return {
            'seasonal_strength': seasonal_strength,
            'trend_strength': trend_strength
        }

    def plot_acf_pacf(self, column: str, lags: int = 40) -> plt.Figure:
        """Plot ACF and PACF"""
        if self.df is None:
            return None

        data = self.df[column].dropna()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        plot_acf(data, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')

        max_lags = min(lags, len(data) // 2 - 1)
        plot_pacf(data, lags=max_lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        return fig

    def plot_arima_fit(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot ARIMA model fit"""
        original = results.get('original_data')
        fitted = results.get('fitted_values')

        if original is None or fitted is None:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(original, label='Actual', alpha=0.8)
        ax.plot(fitted, label='ARIMA Fitted', alpha=0.7)
        ax.legend()
        ax.set_title('ARIMA Model Fit')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        plt.tight_layout()

        return fig

    def plot_decomposition(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot time series decomposition"""
        observed = results.get('observed')
        trend = results.get('trend')
        seasonal = results.get('seasonal')
        residual = results.get('residual')

        if observed is None:
            return None

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        axes[0].plot(observed)
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Time Series Decomposition')

        axes[1].plot(trend)
        axes[1].set_ylabel('Trend')

        axes[2].plot(seasonal)
        axes[2].set_ylabel('Seasonal')

        axes[3].plot(residual)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')

        plt.tight_layout()
        return fig

    def plot_rolling_stats(self, results: Dict[str, Any], column: str) -> plt.Figure:
        """Plot rolling statistics"""
        original = results.get('original')
        rolling_mean = results.get('rolling_mean')
        rolling_std = results.get('rolling_std')
        window = results.get('window', 12)

        if original is None:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(original, label='Original', alpha=0.7)
        axes[0].plot(rolling_mean, label=f'Rolling Mean (window={window})', color='red')
        axes[0].legend()
        axes[0].set_title(f'Rolling Mean - {column}')
        axes[0].grid(True)

        axes[2].plot(seasonal)
        axes[2].set_ylabel('Seasonal')

        axes[3].plot(residual)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')

        plt.tight_layout()
        return fig

    def plot_rolling_stats(self, results: Dict[str, Any], column: str) -> plt.Figure:
        """Plot rolling statistics"""
        original = results.get('original')
        rolling_mean = results.get('rolling_mean')
        rolling_std = results.get('rolling_std')
        window = results.get('window', 12)

        if original is None:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(original, label='Original', alpha=0.7)
        axes[0].plot(rolling_mean, label=f'Rolling Mean (window={window})', color='red')
        axes[0].legend()
        axes[0].set_title(f'Rolling Mean - {column}')
        axes[0].grid(True)

        axes[1].plot(rolling_std, label=f'Rolling Std (window={window})', color='orange')
        axes[1].legend()
        axes[1].set_title(f'Rolling Standard Deviation - {column}')
        axes[1].grid(True)

        plt.tight_layout()
        return fig

    # =========================================================================
    # FOURIER ANALYSIS
    # =========================================================================

    def fourier_transform(self, column: str, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Compute Fast Fourier Transform (FFT)

        Decomposes time series into frequency components.

        Args:
            column: Column name for time series
            sampling_rate: Sampling frequency (samples per unit time)

        Returns:
            Dictionary with FFT results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        return signal_analysis.fourier_transform(self.df, column, sampling_rate)

    def power_spectral_density(self, column: str, sampling_rate: float = 1.0,
                              window: str = 'hamming', nperseg: int = None) -> Dict[str, Any]:
        """
        Compute Power Spectral Density (PSD) using Welch's method

        More robust estimate of PSD than FFT, reduces noise.

        Args:
            column: Column name for time series
            sampling_rate: Sampling frequency
            window: Window type ('hamming', 'hann', 'blackman', 'bartlett')
            nperseg: Length of each segment

        Returns:
            Dictionary with PSD results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        return signal_analysis.power_spectral_density(self.df, column, sampling_rate=sampling_rate, window=window, nperseg=nperseg)

    def plot_fft(self, results: Dict[str, Any], column: str, max_freq: float = None) -> plt.Figure:
        """Plot FFT results"""
        return signal_analysis.plot_fft(results, column, max_freq=max_freq)

    def plot_power_spectral_density(self, results: Dict[str, Any], column: str) -> plt.Figure:
        """Plot Power Spectral Density"""
        return signal_analysis.plot_power_spectral_density(results, column)

    # =========================================================================
    # WAVELET ANALYSIS
    # =========================================================================

    def continuous_wavelet_transform(self, column: str, scales: np.ndarray = None,
                                     wavelet: str = 'morl', sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Continuous Wavelet Transform (CWT)

        Analyzes nonstationary time series by decomposing into
        time-frequency representation.

        Ideal for detecting transient features and time-varying frequencies.

        Args:
            column: Column name for time series
            scales: Scales to analyze (if None, auto-generated)
            wavelet: Wavelet type ('morl' for Morlet, 'mexh' for Mexican Hat, etc.)

        Returns:
            Dictionary with CWT results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        return signal_analysis.continuous_wavelet_transform(self.df, column, scales=scales, wavelet=wavelet, sampling_rate=sampling_rate)

    def discrete_wavelet_transform(self, column: str, wavelet: str = 'db4',
                                  level: int = 3) -> Dict[str, Any]:
        """
        Discrete Wavelet Transform (DWT)

        Multi-resolution analysis useful for denoising and feature extraction.

        Args:
            column: Column name
            wavelet: Wavelet type ('db4', 'db8', 'sym5', etc.)
            level: Decomposition level

        Returns:
            Dictionary with DWT coefficients
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        return signal_analysis.discrete_wavelet_transform(self.df, column, wavelet=wavelet, level=level)

    def plot_wavelet_power(self, cwt_results: Dict[str, Any], column: str) -> plt.Figure:
        """Plot Continuous Wavelet Transform power spectrum"""
        return signal_analysis.plot_wavelet_power(cwt_results, column)

    def plot_wavelet_torrence(self, cwt_results: Dict[str, Any], column: str,
                              y_scale: str = 'log', significance_level: float = 0.95, show_coi: bool = True, wavelet: str = None, ax=None) -> plt.Figure:
        """Torrence & Compo style wavelet power plot with COI overlay.

        This function plots log(power) as a filled contour, y-axis as period (increasing upward),
        and overlays the cone of influence (COI)."""
        return signal_analysis.plot_wavelet_torrence(cwt_results, column, y_scale=y_scale, significance_level=significance_level, show_coi=show_coi, wavelet=wavelet, ax=ax)

    def plot_discrete_wavelet(self, dwt_results: Dict[str, Any], column: str) -> plt.Figure:
        """Plot Discrete Wavelet Transform coefficients"""
        return signal_analysis.plot_discrete_wavelet(dwt_results, column)

