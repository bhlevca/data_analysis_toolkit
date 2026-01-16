"""
Advanced Time Series Module
============================
Advanced time series analysis methods:
- Prophet integration for forecasting
- Changepoint detection (PELT, Binary Segmentation)
- Dynamic Time Warping (DTW)
- State Space Models
- Vector Autoregression (VAR)
- Granger Causality
- Cointegration testing

Version: 1.0

Requirements:
    pip install prophet ruptures dtaidistance
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Optional imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests, coint, adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class AdvancedTimeSeries:
    """Advanced time series analysis methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.prophet_model = None
        self.var_model = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    # =========================================================================
    # PROPHET FORECASTING
    # =========================================================================
    
    def prophet_forecast(self, date_col: str, value_col: str,
                         periods: int = 30, freq: str = 'D',
                         yearly_seasonality: bool = True,
                         weekly_seasonality: bool = True,
                         daily_seasonality: bool = False,
                         holidays: pd.DataFrame = None,
                         changepoint_prior_scale: float = 0.05,
                         seasonality_prior_scale: float = 10.0) -> Dict[str, Any]:
        """
        Facebook Prophet forecasting
        
        Args:
            date_col: Date column name
            value_col: Value column name
            periods: Number of periods to forecast
            freq: Frequency ('D'=daily, 'H'=hourly, 'W'=weekly, 'M'=monthly)
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            holidays: DataFrame with holiday dates
            changepoint_prior_scale: Flexibility of trend changepoints
            seasonality_prior_scale: Strength of seasonality
            
        Returns:
            Dictionary with forecast and components
        """
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet not installed. Install with: pip install prophet'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data for Prophet
        df_prophet = self.df[[date_col, value_col]].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet.dropna()
        
        # Initialize model
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        if holidays is not None:
            self.prophet_model.add_country_holidays(country_name='US')
        
        # Fit model
        self.prophet_model.fit(df_prophet)
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
        
        # Forecast
        forecast = self.prophet_model.predict(future)
        
        # Extract results
        results = {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
            'full_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
            'components': {
                'trend': forecast['trend'].tolist(),
            },
            'changepoints': self.prophet_model.changepoints.tolist() if hasattr(self.prophet_model, 'changepoints') else [],
            'n_changepoints': len(self.prophet_model.changepoints) if hasattr(self.prophet_model, 'changepoints') else 0,
            'periods_forecast': periods,
            'frequency': freq
        }
        
        # Add seasonality components if present
        if yearly_seasonality and 'yearly' in forecast.columns:
            results['components']['yearly'] = forecast['yearly'].tolist()
        if weekly_seasonality and 'weekly' in forecast.columns:
            results['components']['weekly'] = forecast['weekly'].tolist()
        if daily_seasonality and 'daily' in forecast.columns:
            results['components']['daily'] = forecast['daily'].tolist()
        
        return results
    
    def prophet_cross_validation(self, date_col: str, value_col: str,
                                  horizon: str = '30 days',
                                  initial: str = '365 days',
                                  period: str = '30 days') -> Dict[str, Any]:
        """
        Cross-validation for Prophet model
        
        Args:
            date_col: Date column
            value_col: Value column
            horizon: Forecast horizon
            initial: Initial training period
            period: Period between cutoffs
            
        Returns:
            Dictionary with cross-validation metrics
        """
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet not installed'}
        
        from prophet.diagnostics import cross_validation, performance_metrics
        
        # Fit model if not already
        if self.prophet_model is None:
            self.prophet_forecast(date_col, value_col)
        
        # Cross-validation
        df_cv = cross_validation(self.prophet_model, horizon=horizon,
                                 initial=initial, period=period)
        
        # Performance metrics
        metrics = performance_metrics(df_cv)
        
        return {
            'mse': float(metrics['mse'].mean()),
            'rmse': float(metrics['rmse'].mean()),
            'mae': float(metrics['mae'].mean()),
            'mape': float(metrics['mape'].mean()) if 'mape' in metrics else None,
            'coverage': float(metrics['coverage'].mean()) if 'coverage' in metrics else None,
            'metrics_by_horizon': metrics.to_dict('records')
        }
    
    def plot_prophet_forecast(self, forecast_results: Dict[str, Any] = None) -> plt.Figure:
        """Plot Prophet forecast"""
        if not PROPHET_AVAILABLE or self.prophet_model is None:
            return None
        
        fig = self.prophet_model.plot(
            self.prophet_model.predict(
                self.prophet_model.make_future_dataframe(
                    periods=forecast_results.get('periods_forecast', 30) if forecast_results else 30
                )
            )
        )
        plt.title('Prophet Forecast')
        return fig
    
    def plot_prophet_components(self) -> plt.Figure:
        """Plot Prophet forecast components"""
        if not PROPHET_AVAILABLE or self.prophet_model is None:
            return None
        
        future = self.prophet_model.make_future_dataframe(periods=30)
        forecast = self.prophet_model.predict(future)
        fig = self.prophet_model.plot_components(forecast)
        return fig
    
    # =========================================================================
    # CHANGEPOINT DETECTION
    # =========================================================================
    
    def detect_changepoints(self, column: str, method: str = 'pelt',
                            n_bkps: int = None, penalty: float = None,
                            min_size: int = 2) -> Dict[str, Any]:
        """
        Detect changepoints in time series
        
        Args:
            column: Column to analyze
            method: Detection method
                   - 'pelt': Pruned Exact Linear Time
                   - 'binseg': Binary Segmentation
                   - 'bottomup': Bottom-Up segmentation
                   - 'window': Window-based
            n_bkps: Number of breakpoints (for binseg, bottomup)
            penalty: Penalty value (for pelt)
            min_size: Minimum segment size
            
        Returns:
            Dictionary with changepoint locations and segments
        """
        if not RUPTURES_AVAILABLE:
            return {'error': 'ruptures not installed. Install with: pip install ruptures'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        signal = self.df[column].dropna().values
        
        # Select algorithm
        if method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=min_size)
            if penalty is None:
                penalty = np.log(len(signal)) * signal.var()
            result = algo.fit_predict(signal, pen=penalty)
            
        elif method == 'binseg':
            algo = rpt.Binseg(model='rbf', min_size=min_size)
            algo.fit(signal)
            if n_bkps is None:
                n_bkps = 5
            result = algo.predict(n_bkps=n_bkps)
            
        elif method == 'bottomup':
            algo = rpt.BottomUp(model='rbf', min_size=min_size)
            algo.fit(signal)
            if n_bkps is None:
                n_bkps = 5
            result = algo.predict(n_bkps=n_bkps)
            
        elif method == 'window':
            algo = rpt.Window(model='rbf', min_size=min_size, width=50)
            algo.fit(signal)
            if n_bkps is None:
                n_bkps = 5
            result = algo.predict(n_bkps=n_bkps)
            
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Remove the last element (end of signal)
        changepoints = result[:-1] if result else []
        
        # Segment statistics
        segments = []
        prev = 0
        for cp in changepoints + [len(signal)]:
            segment_data = signal[prev:cp]
            segments.append({
                'start': int(prev),
                'end': int(cp),
                'mean': float(np.mean(segment_data)),
                'std': float(np.std(segment_data)),
                'length': int(cp - prev)
            })
            prev = cp
        
        return {
            'changepoints': changepoints,
            'n_changepoints': len(changepoints),
            'segments': segments,
            'method': method
        }
    
    def plot_changepoints(self, column: str, changepoint_results: Dict[str, Any]) -> plt.Figure:
        """Plot time series with detected changepoints"""
        if self.df is None:
            return None
        
        signal = self.df[column].dropna().values
        changepoints = changepoint_results.get('changepoints', [])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(signal, 'b-', alpha=0.7, label='Signal')
        
        # Mark changepoints
        for cp in changepoints:
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
        
        # Color segments
        prev = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(changepoints) + 1))
        for i, cp in enumerate(changepoints + [len(signal)]):
            ax.axvspan(prev, cp, alpha=0.1, color=colors[i])
            prev = cp
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel(column)
        ax.set_title(f'Changepoint Detection ({len(changepoints)} changepoints found)')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # DYNAMIC TIME WARPING
    # =========================================================================
    
    def dtw_distance(self, series1: Union[str, np.ndarray],
                     series2: Union[str, np.ndarray],
                     window: int = None) -> Dict[str, Any]:
        """
        Calculate Dynamic Time Warping distance
        
        Args:
            series1: First series (column name or array)
            series2: Second series (column name or array)
            window: Warping window constraint
            
        Returns:
            Dictionary with DTW distance and path
        """
        if not DTW_AVAILABLE:
            return {'error': 'dtaidistance not installed. Install with: pip install dtaidistance'}
        
        # Get series
        if isinstance(series1, str):
            s1 = self.df[series1].dropna().values.astype(np.float64)
        else:
            s1 = np.array(series1, dtype=np.float64)
        
        if isinstance(series2, str):
            s2 = self.df[series2].dropna().values.astype(np.float64)
        else:
            s2 = np.array(series2, dtype=np.float64)
        
        # Calculate DTW distance
        if window:
            distance = dtw.distance(s1, s2, window=window)
        else:
            distance = dtw.distance(s1, s2)
        
        # Get warping path
        try:
            path = dtw.warping_path(s1, s2)
        except:
            path = []
        
        # Normalized distance
        normalized_distance = distance / (len(s1) + len(s2))
        
        return {
            'dtw_distance': float(distance),
            'normalized_distance': float(normalized_distance),
            'warping_path': path,
            'series1_length': len(s1),
            'series2_length': len(s2)
        }
    
    def dtw_distance_matrix(self, columns: List[str]) -> Dict[str, Any]:
        """
        Calculate DTW distance matrix for multiple series
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary with distance matrix
        """
        if not DTW_AVAILABLE:
            return {'error': 'dtaidistance not installed'}
        
        n = len(columns)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                result = self.dtw_distance(columns[i], columns[j])
                if 'error' not in result:
                    dist = result['dtw_distance']
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        return {
            'distance_matrix': distance_matrix.tolist(),
            'columns': columns
        }
    
    def plot_dtw_alignment(self, series1: str, series2: str) -> plt.Figure:
        """Plot DTW alignment between two series"""
        if self.df is None:
            return None
        
        s1 = self.df[series1].dropna().values
        s2 = self.df[series2].dropna().values
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top: Both series
        axes[0].plot(s1, 'b-', label=series1)
        axes[0].plot(s2, 'r-', label=series2)
        axes[0].legend()
        axes[0].set_title('Time Series Comparison')
        
        # Bottom: DTW alignment visualization
        if DTW_AVAILABLE:
            try:
                path = dtw.warping_path(s1.astype(np.float64), s2.astype(np.float64))
                axes[1].plot(s1, 'b-', label=series1)
                axes[1].plot(s2, 'r-', label=series2)
                
                # Draw alignment lines
                for (i, j) in path[::5]:  # Every 5th point
                    axes[1].plot([i, j], [s1[i], s2[j]], 'g-', alpha=0.3)
                
                axes[1].legend()
                axes[1].set_title('DTW Alignment')
            except:
                axes[1].text(0.5, 0.5, 'Could not compute DTW path',
                           ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # VECTOR AUTOREGRESSION (VAR)
    # =========================================================================
    
    def var_analysis(self, columns: List[str], max_lags: int = 10,
                     ic: str = 'aic') -> Dict[str, Any]:
        """
        Vector Autoregression analysis
        
        Args:
            columns: Columns for VAR model
            max_lags: Maximum lags to test
            ic: Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            Dictionary with VAR results
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna()
        
        # Fit VAR model
        model = VAR(data)
        
        # Select optimal lag
        lag_order = model.select_order(maxlags=max_lags)
        optimal_lag = getattr(lag_order, ic)
        
        # Fit with optimal lag
        self.var_model = model.fit(optimal_lag)
        
        # Results
        results = {
            'optimal_lag': int(optimal_lag),
            'lag_criteria': {
                'aic': int(lag_order.aic),
                'bic': int(lag_order.bic),
                'hqic': int(lag_order.hqic)
            },
            'coefficients': {},
            'summary_statistics': {
                'aic': float(self.var_model.aic),
                'bic': float(self.var_model.bic),
                'fpe': float(self.var_model.fpe)
            }
        }
        
        # Extract coefficients
        for i, col in enumerate(columns):
            coefs = self.var_model.coefs[:, i, :].flatten().tolist()
            results['coefficients'][col] = coefs
        
        return results
    
    def var_forecast(self, steps: int = 10) -> Dict[str, Any]:
        """
        Forecast using fitted VAR model
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecasts
        """
        if self.var_model is None:
            return {'error': 'VAR model not fitted. Run var_analysis first.'}
        
        forecast = self.var_model.forecast(self.var_model.endog[-self.var_model.k_ar:], steps)
        
        columns = self.var_model.names
        
        return {
            'forecast': {col: forecast[:, i].tolist() for i, col in enumerate(columns)},
            'steps': steps
        }
    
    def granger_causality(self, columns: List[str], max_lag: int = 10,
                          test: str = 'ssr_ftest') -> Dict[str, Any]:
        """
        Granger causality test
        
        Args:
            columns: Columns to test (pairwise)
            max_lag: Maximum lag for testing
            test: Test type ('ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest')
            
        Returns:
            Dictionary with Granger causality results
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna()
        
        results = {}
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    try:
                        test_result = grangercausalitytests(
                            data[[col2, col1]],  # [effect, cause]
                            maxlag=max_lag,
                            verbose=False
                        )
                        
                        # Get minimum p-value across lags
                        p_values = [test_result[lag][0][test][1] for lag in range(1, max_lag + 1)]
                        min_p = min(p_values)
                        best_lag = p_values.index(min_p) + 1
                        
                        results[f'{col1} -> {col2}'] = {
                            'p_value': float(min_p),
                            'best_lag': int(best_lag),
                            'causes': min_p < 0.05
                        }
                    except Exception as e:
                        results[f'{col1} -> {col2}'] = {'error': str(e)}
        
        return {
            'pairwise_results': results,
            'max_lag': max_lag,
            'test': test
        }
    
    # =========================================================================
    # COINTEGRATION
    # =========================================================================
    
    def cointegration_test(self, columns: List[str],
                           method: str = 'johansen') -> Dict[str, Any]:
        """
        Test for cointegration between time series
        
        Args:
            columns: Columns to test
            method: 'johansen' or 'engle-granger'
            
        Returns:
            Dictionary with cointegration test results
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna()
        
        if method == 'johansen':
            # Johansen test
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            trace_stat = result.lr1.tolist()
            trace_crit = result.cvt.tolist()
            max_eig_stat = result.lr2.tolist()
            max_eig_crit = result.cvm.tolist()
            
            # Count cointegrating relations
            n_coint = sum(1 for i, (stat, crit) in enumerate(zip(trace_stat, trace_crit))
                         if stat > crit[1])  # 95% level
            
            return {
                'method': 'johansen',
                'trace_statistics': trace_stat,
                'trace_critical_values': trace_crit,
                'max_eigenvalue_statistics': max_eig_stat,
                'max_eigenvalue_critical_values': max_eig_crit,
                'n_cointegrating_relations': n_coint,
                'is_cointegrated': n_coint > 0
            }
            
        elif method == 'engle-granger':
            if len(columns) != 2:
                return {'error': 'Engle-Granger test requires exactly 2 columns'}
            
            stat, p_value, crit_values = coint(data[columns[0]], data[columns[1]])
            
            return {
                'method': 'engle-granger',
                'test_statistic': float(stat),
                'p_value': float(p_value),
                'critical_values': {
                    '1%': float(crit_values[0]),
                    '5%': float(crit_values[1]),
                    '10%': float(crit_values[2])
                },
                'is_cointegrated': p_value < 0.05
            }
        
        return {'error': f'Unknown method: {method}'}
    
    def stationarity_test(self, column: str, test: str = 'adf') -> Dict[str, Any]:
        """
        Test for stationarity
        
        Args:
            column: Column to test
            test: 'adf' (Augmented Dickey-Fuller)
            
        Returns:
            Dictionary with stationarity test results
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        data = self.df[column].dropna().values
        
        if test == 'adf':
            result = adfuller(data)
            
            return {
                'test': 'Augmented Dickey-Fuller',
                'test_statistic': float(result[0]),
                'p_value': float(result[1]),
                'lags_used': int(result[2]),
                'n_observations': int(result[3]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'is_stationary': result[1] < 0.05,
                'interpretation': 'Stationary (reject H0)' if result[1] < 0.05 else 'Non-stationary (fail to reject H0)'
            }
        
        return {'error': f'Unknown test: {test}'}
