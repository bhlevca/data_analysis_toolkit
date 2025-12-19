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
from statsmodels.tsa.stattools import adfuller, acf, pacf, grangercausalitytests, coint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.spatial.distance import cdist
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import chi2

# Try to import VECM (may not be in all statsmodels versions)
try:
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    VECM_AVAILABLE = True
except ImportError:
    VECM_AVAILABLE = False

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

    def auto_arima(self, column: str, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                   seasonal: bool = False, m: int = 12) -> Dict[str, Any]:
        """
        Automatically select best ARIMA/SARIMA parameters using grid search with AIC.
        
        Args:
            column: Column name for time series
            max_p: Maximum AR order to test
            max_d: Maximum differencing order to test
            max_q: Maximum MA order to test
            seasonal: Whether to include seasonal components
            m: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
            
        Returns:
            Dictionary with best model parameters and fit results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        if len(data) < 30:
            return {'error': 'Need at least 30 observations for auto_arima'}
        
        try:
            best_aic = float('inf')
            best_order = (0, 0, 0)
            best_seasonal_order = (0, 0, 0, 0)
            best_model = None
            
            # Determine differencing order using ADF test
            from statsmodels.tsa.stattools import adfuller
            d = 0
            temp_data = data.copy()
            while d <= max_d:
                adf_result = adfuller(temp_data, autolag='AIC')
                if adf_result[1] < 0.05:  # Stationary
                    break
                temp_data = temp_data.diff().dropna()
                d += 1
            
            # Grid search over p and q
            tested_orders = []
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue  # Skip trivial model
                    
                    try:
                        if seasonal:
                            # Test with seasonal component
                            for P in range(2):
                                for Q in range(2):
                                    seasonal_order = (P, 1, Q, m)
                                    model = SARIMAX(data, order=(p, d, q), 
                                                    seasonal_order=seasonal_order,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                    results = model.fit(disp=False)
                                    tested_orders.append({
                                        'order': (p, d, q),
                                        'seasonal_order': seasonal_order,
                                        'aic': results.aic
                                    })
                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = seasonal_order
                                        best_model = results
                        else:
                            model = ARIMA(data, order=(p, d, q))
                            results = model.fit()
                            tested_orders.append({
                                'order': (p, d, q),
                                'aic': results.aic
                            })
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_model = results
                    except Exception:
                        continue
            
            if best_model is None:
                return {'error': 'Could not find suitable ARIMA model'}
            
            return {
                'best_order': best_order,
                'best_seasonal_order': best_seasonal_order if seasonal else None,
                'aic': best_aic,
                'bic': best_model.bic,
                'n_tested': len(tested_orders),
                'fitted_values': best_model.fittedvalues.values,
                'original_data': data.values,
                'residuals': best_model.resid.values,
                'summary': str(best_model.summary()),
                'params': dict(best_model.params),
                'is_seasonal': seasonal
            }
        except Exception as e:
            return {'error': str(e)}

    def sarima_model(self, column: str, order: Tuple[int, int, int] = (1, 1, 1),
                     seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)) -> Dict[str, Any]:
        """
        Fit SARIMA (Seasonal ARIMA) model.
        
        Args:
            column: Column name for time series
            order: (p, d, q) non-seasonal order
            seasonal_order: (P, D, Q, m) seasonal order
            
        Returns:
            Dictionary with SARIMA results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        try:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            
            return {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': results.aic,
                'bic': results.bic,
                'summary': str(results.summary()),
                'fitted_values': results.fittedvalues.values,
                'original_data': data.values,
                'residuals': results.resid.values,
                'params': dict(results.params)
            }
        except Exception as e:
            return {'error': str(e)}

    def arima_forecast(self, column: str, steps: int = 10, 
                       order: Tuple[int, int, int] = None,
                       seasonal_order: Tuple[int, int, int, int] = None,
                       confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Forecast future values using ARIMA/SARIMA with confidence intervals.
        
        Args:
            column: Column name for time series
            steps: Number of steps to forecast
            order: (p, d, q) ARIMA order. If None, uses auto_arima.
            seasonal_order: (P, D, Q, m) seasonal order. If provided, uses SARIMA.
            confidence_level: Confidence level for prediction intervals (0-1)
            
        Returns:
            Dictionary with forecast, confidence intervals, and model info
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        try:
            # Auto-select order if not provided
            if order is None:
                auto_result = self.auto_arima(column, max_p=3, max_d=2, max_q=3, 
                                              seasonal=seasonal_order is not None)
                if 'error' in auto_result:
                    return auto_result
                order = auto_result['best_order']
                if seasonal_order is None and auto_result.get('best_seasonal_order'):
                    seasonal_order = auto_result['best_seasonal_order']
            
            # Fit model
            if seasonal_order:
                model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ARIMA(data, order=order)
            
            results = model.fit(disp=False) if seasonal_order else model.fit()
            
            # Generate forecast with confidence intervals
            alpha = 1 - confidence_level
            forecast_result = results.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=alpha)
            
            # Create time index for forecast
            if hasattr(data.index, 'freq') and data.index.freq:
                forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq=data.index.freq)[1:]
            else:
                forecast_index = np.arange(len(data), len(data) + steps)
            
            return {
                'forecast': forecast_mean.values,
                'forecast_index': list(forecast_index),
                'lower_ci': conf_int.iloc[:, 0].values,
                'upper_ci': conf_int.iloc[:, 1].values,
                'confidence_level': confidence_level,
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': results.aic,
                'bic': results.bic,
                'fitted_values': results.fittedvalues.values,
                'original_data': data.values,
                'original_index': list(data.index),
                'residuals': results.resid.values,
                'summary': str(results.summary())
            }
        except Exception as e:
            return {'error': str(e)}

    def plot_forecast(self, forecast_results: Dict[str, Any], title: str = "ARIMA Forecast") -> plt.Figure:
        """
        Plot forecast results with confidence intervals.
        
        Args:
            forecast_results: Dictionary from arima_forecast()
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if 'error' in forecast_results:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Original data
        original = forecast_results.get('original_data', [])
        fitted = forecast_results.get('fitted_values', [])
        forecast = forecast_results.get('forecast', [])
        lower_ci = forecast_results.get('lower_ci', [])
        upper_ci = forecast_results.get('upper_ci', [])
        
        n_original = len(original)
        n_forecast = len(forecast)
        
        # Plot original data
        ax.plot(range(n_original), original, 'b-', label='Original Data', linewidth=1.5)
        
        # Plot fitted values
        ax.plot(range(n_original), fitted, 'g--', label='Fitted', alpha=0.7)
        
        # Plot forecast
        forecast_x = range(n_original, n_original + n_forecast)
        ax.plot(forecast_x, forecast, 'r-', label='Forecast', linewidth=2)
        
        # Plot confidence interval
        conf_level = forecast_results.get('confidence_level', 0.95)
        ax.fill_between(forecast_x, lower_ci, upper_ci, color='red', alpha=0.2,
                       label=f'{conf_level*100:.0f}% Confidence Interval')
        
        # Add vertical line at forecast start
        ax.axvline(x=n_original - 1, color='gray', linestyle='--', alpha=0.5)
        
        order = forecast_results.get('order', (0, 0, 0))
        seasonal = forecast_results.get('seasonal_order')
        model_str = f"ARIMA{order}"
        if seasonal:
            model_str = f"SARIMA{order}x{seasonal}"
        
        ax.set_title(f'{title}\n{model_str} - AIC: {forecast_results.get("aic", 0):.2f}')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

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

    # =========================================================================
    # EXTENDED SPECTRAL ANALYSIS
    # =========================================================================

    def coherence_analysis(self, column1: str, column2: str, sampling_rate: float = 1.0,
                           nperseg: int = None) -> Dict[str, Any]:
        """
        Compute magnitude-squared coherence between two signals.
        
        Coherence measures linear correlation as a function of frequency.
        Values range from 0 (no correlation) to 1 (perfect correlation).
        
        Args:
            column1: First signal column
            column2: Second signal column
            sampling_rate: Sampling frequency
            nperseg: Segment length for Welch method
            
        Returns:
            Dictionary with coherence, phase, and cross-spectral density
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        return signal_analysis.coherence_analysis(self.df, column1, column2, 
                                                   sampling_rate=sampling_rate, nperseg=nperseg)

    def cross_wavelet_transform(self, column1: str, column2: str, scales: np.ndarray = None,
                                 wavelet: str = 'morl', sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Compute Cross-Wavelet Transform (XWT) between two time series.
        
        Reveals common power and relative phase in time-frequency space.
        
        Args:
            column1: First signal column
            column2: Second signal column
            scales: Wavelet scales (auto-generated if None)
            wavelet: Wavelet type (default: 'morl')
            sampling_rate: Sampling frequency
            
        Returns:
            Dictionary with XWT power, phase difference, and COI
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        return signal_analysis.cross_wavelet_transform(self.df, column1, column2,
                                                        scales=scales, wavelet=wavelet,
                                                        sampling_rate=sampling_rate)

    def wavelet_coherence(self, column1: str, column2: str, scales: np.ndarray = None,
                          wavelet: str = 'morl', sampling_rate: float = 1.0,
                          smooth_factor: int = 5) -> Dict[str, Any]:
        """
        Compute Wavelet Coherence (WTC) between two time series.
        
        Measures intensity of covariance in time-frequency space,
        normalized to [0, 1] like regular coherence.
        
        Args:
            column1: First signal column
            column2: Second signal column
            scales: Wavelet scales
            wavelet: Wavelet type
            sampling_rate: Sampling frequency
            smooth_factor: Smoothing window size
            
        Returns:
            Dictionary with coherence, phase, and significance info
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        return signal_analysis.wavelet_coherence(self.df, column1, column2,
                                                  scales=scales, wavelet=wavelet,
                                                  sampling_rate=sampling_rate,
                                                  smooth_factor=smooth_factor)

    def harmonic_analysis(self, column: str, n_harmonics: int = 5,
                          sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Perform harmonic analysis using least-squares fitting.
        
        Fits sinusoids to extract dominant periodic components.
        Useful for tidal analysis, seasonal patterns, etc.
        
        Args:
            column: Column to analyze
            n_harmonics: Number of harmonics to fit
            sampling_rate: Sampling frequency
            
        Returns:
            Dictionary with harmonics, amplitudes, phases, R-squared
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        return signal_analysis.harmonic_analysis(self.df, column,
                                                  n_harmonics=n_harmonics,
                                                  sampling_rate=sampling_rate)

    def plot_coherence(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot coherence analysis results"""
        return signal_analysis.plot_coherence(results)

    def plot_cross_wavelet(self, results: Dict[str, Any], show_phase_arrows: bool = True) -> plt.Figure:
        """Plot Cross-Wavelet Transform with phase arrows"""
        return signal_analysis.plot_cross_wavelet(results, show_phase_arrows=show_phase_arrows)

    def plot_wavelet_coherence(self, results: Dict[str, Any], show_phase_arrows: bool = True) -> plt.Figure:
        """Plot Wavelet Coherence with phase arrows"""
        return signal_analysis.plot_wavelet_coherence(results, show_phase_arrows=show_phase_arrows)

    def plot_harmonic_analysis(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot harmonic analysis results"""
        return signal_analysis.plot_harmonic_analysis(results)

    # ===== MULTIVARIATE TIME SERIES METHODS =====
    
    def var_model(self, columns: List[str], maxlags: int = None, ic: str = 'aic') -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model for multivariate time series.
        
        VAR captures linear interdependencies among multiple time series.
        Each variable is modeled as a linear function of its own past values
        and past values of other variables.
        
        Args:
            columns: List of column names to include in VAR
            maxlags: Maximum number of lags to consider (auto-select if None)
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
            
        Returns:
            Dictionary with VAR results, coefficients, and diagnostics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if len(columns) < 2:
            return {'error': 'VAR requires at least 2 variables'}
        
        try:
            # Prepare data matrix
            data = self.df[columns].dropna()
            
            if len(data) < 30:
                return {'error': 'Need at least 30 observations for VAR'}
            
            # Fit VAR model
            model = VAR(data)
            
            # Select optimal lag order if not specified
            if maxlags is None:
                maxlags = min(15, len(data) // 5)
            
            lag_order = model.select_order(maxlags=maxlags)
            optimal_lag = getattr(lag_order, ic, 1)
            if optimal_lag is None or optimal_lag == 0:
                optimal_lag = 1
            
            results = model.fit(maxlags=optimal_lag)
            
            # Extract coefficients for each equation
            coef_dict = {}
            for i, col in enumerate(columns):
                coef_dict[col] = dict(results.params.iloc[:, i])
            
            # Granger causality tests
            granger_results = {}
            for caused in columns:
                for causing in columns:
                    if caused != causing:
                        try:
                            gc_test = grangercausalitytests(data[[caused, causing]], maxlag=optimal_lag, verbose=False)
                            # Get p-value from F-test at optimal lag
                            p_value = gc_test[optimal_lag][0]['ssr_ftest'][1]
                            granger_results[f'{causing} -> {caused}'] = {
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception:
                            pass
            
            return {
                'columns': columns,
                'optimal_lag': optimal_lag,
                'lag_order_selection': {
                    'aic': lag_order.aic,
                    'bic': lag_order.bic,
                    'hqic': lag_order.hqic,
                    'fpe': lag_order.fpe
                },
                'aic': results.aic,
                'bic': results.bic,
                'coefficients': coef_dict,
                'fitted_values': {col: results.fittedvalues[col].values for col in columns},
                'residuals': {col: results.resid[col].values for col in columns},
                'granger_causality': granger_results,
                'summary': str(results.summary())
            }
        except Exception as e:
            return {'error': str(e)}

    def var_forecast(self, var_results: Dict[str, Any], steps: int = 10) -> Dict[str, Any]:
        """
        Generate forecasts from a fitted VAR model.
        
        Args:
            var_results: Results dictionary from var_model()
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecasts for each variable
        """
        if 'error' in var_results:
            return var_results
        
        columns = var_results.get('columns', [])
        optimal_lag = var_results.get('optimal_lag', 1)
        
        try:
            data = self.df[columns].dropna()
            model = VAR(data)
            results = model.fit(maxlags=optimal_lag)
            
            # Generate forecast
            forecast = results.forecast(data.values[-optimal_lag:], steps=steps)
            
            forecast_dict = {}
            for i, col in enumerate(columns):
                forecast_dict[col] = forecast[:, i].tolist()
            
            return {
                'forecast': forecast_dict,
                'steps': steps,
                'columns': columns
            }
        except Exception as e:
            return {'error': str(e)}

    def vecm_model(self, columns: List[str], deterministic: str = 'co', k_ar_diff: int = 1) -> Dict[str, Any]:
        """
        Fit Vector Error Correction Model (VECM) for cointegrated time series.
        
        VECM is appropriate when variables are non-stationary but cointegrated,
        meaning they share a common long-run equilibrium relationship.
        
        Args:
            columns: List of column names to include
            deterministic: Deterministic terms: 'n'=none, 'co'=constant in cointegration,
                          'ci'=constant inside, 'lo'=linear outside, 'li'=linear inside
            k_ar_diff: Number of lagged differences
            
        Returns:
            Dictionary with VECM results, cointegrating vectors, and diagnostics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if not VECM_AVAILABLE:
            return {'error': 'VECM not available in this statsmodels version'}
        
        if len(columns) < 2:
            return {'error': 'VECM requires at least 2 variables'}
        
        try:
            data = self.df[columns].dropna()
            
            if len(data) < 50:
                return {'error': 'Need at least 50 observations for VECM'}
            
            # Map string deterministic to det_order for Johansen test
            det_order_map = {'n': -1, 'co': 0, 'ci': 0, 'lo': 1, 'li': 1}
            det_order = det_order_map.get(deterministic, 0)
            
            # Johansen cointegration test
            johansen_result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Determine cointegration rank
            trace_stats = johansen_result.lr1  # Trace statistics
            crit_values = johansen_result.cvt[:, 1]  # 5% critical values
            
            coint_rank = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > crit_values[i]:
                    coint_rank = i + 1
            
            if coint_rank == 0:
                return {
                    'warning': 'No cointegration detected at 5% significance level',
                    'trace_statistics': trace_stats.tolist(),
                    'critical_values_5pct': crit_values.tolist(),
                    'cointegration_rank': 0,
                    'columns': columns
                }
            
            # Fit VECM with string deterministic parameter
            model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
            results = model.fit()
            
            return {
                'columns': columns,
                'cointegration_rank': coint_rank,
                'trace_statistics': trace_stats.tolist(),
                'critical_values_5pct': crit_values.tolist(),
                'cointegrating_vectors': johansen_result.evec[:, :coint_rank].tolist(),
                'alpha': results.alpha.tolist(),  # Loading matrix
                'beta': results.beta.tolist(),    # Cointegrating vectors
                'fitted_values': {col: results.fittedvalues[:, i] for i, col in enumerate(columns)},
                'residuals': {col: results.resid[:, i] for i, col in enumerate(columns)},
                'summary': str(results.summary())
            }
        except Exception as e:
            return {'error': str(e)}

    def dtw_distance(self, column1: str, column2: str, window: int = None) -> Dict[str, Any]:
        """
        Compute Dynamic Time Warping (DTW) distance between two time series.
        
        DTW measures similarity between temporal sequences that may vary in
        speed or timing. Useful for comparing time series of different lengths
        or with phase shifts.
        
        Args:
            column1: First time series column
            column2: Second time series column
            window: Sakoe-Chiba band width (None for no constraint)
            
        Returns:
            Dictionary with DTW distance, path, and alignment info
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        try:
            x = self.df[column1].dropna().values
            y = self.df[column2].dropna().values
            
            n, m = len(x), len(y)
            
            if n < 2 or m < 2:
                return {'error': 'Time series must have at least 2 points'}
            
            # Initialize DTW matrix with infinity
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # Fill DTW matrix
            for i in range(1, n + 1):
                if window is None:
                    j_start, j_end = 1, m + 1
                else:
                    j_start = max(1, i - window)
                    j_end = min(m + 1, i + window + 1)
                
                for j in range(j_start, j_end):
                    cost = abs(x[i-1] - y[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],     # insertion
                        dtw_matrix[i, j-1],     # deletion
                        dtw_matrix[i-1, j-1]    # match
                    )
            
            dtw_distance = dtw_matrix[n, m]
            
            # Backtrack to find optimal path
            path = []
            i, j = n, m
            while i > 0 or j > 0:
                path.append((i-1, j-1))
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    options = [
                        (dtw_matrix[i-1, j-1], i-1, j-1),
                        (dtw_matrix[i-1, j], i-1, j),
                        (dtw_matrix[i, j-1], i, j-1)
                    ]
                    _, i, j = min(options, key=lambda x: x[0])
            
            path = path[::-1]  # Reverse to get forward path
            
            # Normalized distance
            normalized_distance = dtw_distance / (n + m)
            
            # Euclidean distance for comparison (if same length)
            if n == m:
                euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
            else:
                euclidean_dist = None
            
            return {
                'dtw_distance': dtw_distance,
                'normalized_distance': normalized_distance,
                'euclidean_distance': euclidean_dist,
                'path': path,
                'path_length': len(path),
                'series1_length': n,
                'series2_length': m,
                'column1': column1,
                'column2': column2
            }
        except Exception as e:
            return {'error': str(e)}

    def dtw_matrix(self, columns: List[str], window: int = None) -> Dict[str, Any]:
        """
        Compute DTW distance matrix for multiple time series.
        
        Useful for clustering or similarity analysis across many variables.
        
        Args:
            columns: List of column names
            window: Sakoe-Chiba band width
            
        Returns:
            Dictionary with distance matrix and column mapping
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        n_cols = len(columns)
        if n_cols < 2:
            return {'error': 'Need at least 2 columns for distance matrix'}
        
        try:
            distance_matrix = np.zeros((n_cols, n_cols))
            
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    result = self.dtw_distance(columns[i], columns[j], window=window)
                    if 'error' in result:
                        return result
                    dist = result['normalized_distance']
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            return {
                'distance_matrix': distance_matrix.tolist(),
                'columns': columns,
                'n_columns': n_cols
            }
        except Exception as e:
            return {'error': str(e)}

    def plot_var_results(self, var_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot VAR model results: fitted vs actual and residuals.
        
        Args:
            var_results: Results from var_model()
            
        Returns:
            Matplotlib figure
        """
        if 'error' in var_results:
            return None
        
        columns = var_results.get('columns', [])
        fitted = var_results.get('fitted_values', {})
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4 * n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        data = self.df[columns].dropna()
        
        for i, col in enumerate(columns):
            # Fitted vs Actual
            ax = axes[i, 0]
            actual = data[col].values
            fitted_vals = fitted.get(col, [])
            
            # Align lengths
            min_len = min(len(actual), len(fitted_vals))
            offset = len(actual) - min_len
            
            ax.plot(range(offset, len(actual)), actual[offset:], 'b-', label='Actual', alpha=0.7)
            ax.plot(range(offset, len(actual)), fitted_vals[:min_len], 'r--', label='Fitted', alpha=0.7)
            ax.set_title(f'{col}: Actual vs Fitted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Residuals
            ax = axes[i, 1]
            residuals = var_results.get('residuals', {}).get(col, [])
            ax.plot(residuals, 'g-', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_title(f'{col}: Residuals')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"VAR({var_results.get('optimal_lag', 1)}) Model Results", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_dtw_alignment(self, dtw_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot DTW alignment between two time series.
        
        Args:
            dtw_results: Results from dtw_distance()
            
        Returns:
            Matplotlib figure with alignment visualization
        """
        if 'error' in dtw_results:
            return None
        
        col1 = dtw_results.get('column1', 'Series 1')
        col2 = dtw_results.get('column2', 'Series 2')
        path = dtw_results.get('path', [])
        
        x = self.df[col1].dropna().values
        y = self.df[col2].dropna().values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top-left: Both series
        ax = axes[0, 0]
        ax.plot(x, 'b-', label=col1, linewidth=1.5)
        ax.plot(y, 'r-', label=col2, linewidth=1.5)
        ax.set_title('Time Series Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top-right: Alignment lines
        ax = axes[0, 1]
        ax.plot(range(len(x)), x, 'b-', label=col1)
        ax.plot(range(len(y)), y + np.max(x) - np.min(y) + np.std(x), 'r-', label=col2)
        
        # Draw alignment lines (sample every nth point for clarity)
        n_lines = min(50, len(path))
        step = max(1, len(path) // n_lines)
        for k in range(0, len(path), step):
            i, j = path[k]
            if i < len(x) and j < len(y):
                ax.plot([i, j], [x[i], y[j] + np.max(x) - np.min(y) + np.std(x)],
                       'g-', alpha=0.3, linewidth=0.5)
        
        ax.set_title('DTW Alignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Warping path
        ax = axes[1, 0]
        path_i = [p[0] for p in path]
        path_j = [p[1] for p in path]
        ax.plot(path_j, path_i, 'k-', linewidth=2)
        ax.plot([0, max(len(y)-1, 1)], [0, max(len(x)-1, 1)], 'r--', alpha=0.5, label='Diagonal')
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title('Warping Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Info text
        ax = axes[1, 1]
        ax.axis('off')
        info_text = f"""
DTW Analysis Results
====================

DTW Distance: {dtw_results.get('dtw_distance', 0):.4f}
Normalized Distance: {dtw_results.get('normalized_distance', 0):.4f}
"""
        if dtw_results.get('euclidean_distance') is not None:
            info_text += f"Euclidean Distance: {dtw_results.get('euclidean_distance', 0):.4f}\n"
        info_text += f"""
Path Length: {dtw_results.get('path_length', 0)}
Series 1 ({col1}): {dtw_results.get('series1_length', 0)} points
Series 2 ({col2}): {dtw_results.get('series2_length', 0)} points
        """
        ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
               verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        return fig


