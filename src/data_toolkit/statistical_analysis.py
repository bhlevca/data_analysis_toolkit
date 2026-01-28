"""
Enhanced Statistical Analysis Module
Contains comprehensive statistical methods including:
- Descriptive statistics and distributions
- Statistical tests (parametric and non-parametric)
- Probability density functions
- Correlation and association analysis
- Outlier detection
- Hypothesis testing

Version: 2.0
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from scipy.stats import (beta, chi2_contingency, expon, f_oneway, fligner,
                         gamma, kendalltau, kruskal, levene)
from scipy.stats import lognorm
from scipy.stats import lognorm as scipy_lognorm
from scipy.stats import (mannwhitneyu, norm, normaltest, pearsonr, shapiro,
                         spearmanr, ttest_ind, ttest_rel, weibull_min,
                         wilcoxon)

warnings.filterwarnings('ignore')

# Import accelerated functions
try:
    from .rust_accelerated import AccelerationSettings
    from .rust_accelerated import detect_outliers_iqr as _accel_outliers
except ImportError:
    from rust_accelerated import AccelerationSettings
    from rust_accelerated import detect_outliers_iqr as _accel_outliers


class StatisticalAnalysis:

    def plot_outlier_line(self, column: str, outlier_info: dict) -> plt.Figure:
        """
        Plot a line plot of the data with outlier thresholds and outlier markers.
        Args:
            column: Column name to plot
            outlier_info: Dict from outlier_detection() for this column
        Returns:
            Matplotlib Figure
        """
        data = self.df[column].dropna()
        outlier_indices = outlier_info.get('outlier_indices', []) if outlier_info else []
        is_outlier = data.index.isin(outlier_indices)
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot all data points (non-outliers in blue, outliers in red)
        ax.scatter(data.index[~is_outlier], data.values[~is_outlier], color='steelblue', s=20, label='Normal')
        if outlier_indices:
            ax.scatter(data.index[is_outlier], data.values[is_outlier], color='red', marker='x', s=80, label='Outliers', zorder=5)
        # Plot thresholds
        if outlier_info:
            if 'lower_bound' in outlier_info:
                ax.axhline(outlier_info['lower_bound'], color='orange', linestyle='--', label='Lower Bound')
            if 'upper_bound' in outlier_info:
                ax.axhline(outlier_info['upper_bound'], color='orange', linestyle='--', label='Upper Bound')
        ax.set_title(f"Outlier Detection: {column}")
        ax.set_xlabel('Index')
        ax.set_ylabel(column)
        ax.legend()
        plt.tight_layout()
        return fig

    def outlier_table(self, column: str, outlier_info: dict) -> pd.DataFrame:
        """
        Return a DataFrame of outlier values and their indices for a given column.
        """
        outlier_indices = outlier_info.get('outlier_indices', [])
        if not outlier_indices:
            return pd.DataFrame(columns=['Index', 'Value'])
        data = self.df[column].loc[outlier_indices]
        return pd.DataFrame({'Index': data.index, 'Value': data.values})

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.last_test_results = {}

    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df

    # =========================================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================================

    def descriptive_stats(self, columns: List[str]) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics

        Returns:
            DataFrame with statistics including skewness and kurtosis
        """
        if self.df is None or not columns:
            return pd.DataFrame()

        stats_dict = {}
        for col in columns:
            data = self.df[col].dropna()
            stats_dict[col] = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                '25%': data.quantile(0.25),
                '50%': data.median(),
                '75%': data.quantile(0.75),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'variance': data.var(),
                'range': data.max() - data.min(),
                'iqr': data.quantile(0.75) - data.quantile(0.25)
            }

        # Return DataFrame with statistics as index and features as columns
        return pd.DataFrame(stats_dict)

    def correlation_matrix(self, columns: List[str], method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix

        Args:
            columns: List of column names
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix as DataFrame
        """
        if self.df is None or not columns:
            return pd.DataFrame()

        return self.df[columns].corr(method=method)

    def plot_correlation_matrices(self, columns: List[str]) -> plt.Figure:
        """
        Plot correlation matrices with multiple methods
        """
        if self.df is None or not columns:
            return None

        corr_pearson = self.df[columns].corr(method='pearson')
        corr_spearman = self.df[columns].corr(method='spearman')
        corr_kendall = self.df[columns].corr(method='kendall')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Pearson Correlation\n(Linear relationships)')

        sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Spearman Correlation\n(Monotonic relationships)')

        sns.heatmap(corr_kendall, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=axes[2], cbar_kws={'label': 'Correlation'})
        axes[2].set_title('Kendall Correlation\n(Rank-based)')

        plt.tight_layout()
        return fig

    # =========================================================================
    # DISTRIBUTION ANALYSIS & PROBABILITY DENSITY FUNCTIONS
    # =========================================================================

    def fit_distributions(self, column: str) -> Dict[str, Any]:
        """
        Fit multiple probability distributions to data

        Fits: Normal, Gamma, Exponential, Lognormal, Weibull, Beta

        Returns:
            Dictionary with fitted parameters and goodness-of-fit metrics
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data = self.df[column].dropna().values

        # Ensure data is positive for certain distributions
        if np.any(data <= 0):
            data_positive = data[data > 0]
            if len(data_positive) < len(data) * 0.5:
                data_shifted = data - data.min() + 1  # Shift to ensure positive
            else:
                data_positive = data_positive
        else:
            data_positive = data
            data_shifted = data

        results = {}

        # Normal Distribution
        try:
            mu, sigma = norm.fit(data)
            results['Normal'] = {
                'parameters': {'mu': float(mu), 'sigma': float(sigma)},
                'ks_stat': float(stats.kstest(data, 'norm', args=(mu, sigma))[0]),
                'shapiro_p': float(shapiro(data)[1])
            }
        except:
            results['Normal'] = {'error': 'Failed to fit'}

        # Gamma Distribution
        try:
            shape, loc, scale = gamma.fit(data_positive)
            results['Gamma'] = {
                'parameters': {'shape': float(shape), 'loc': float(loc), 'scale': float(scale)},
                'ks_stat': float(stats.kstest(data_positive, 'gamma', args=(shape, loc, scale))[0])
            }
        except:
            results['Gamma'] = {'error': 'Failed to fit'}

        # Exponential Distribution
        try:
            loc, scale = expon.fit(data_positive)
            results['Exponential'] = {
                'parameters': {'loc': float(loc), 'scale': float(scale)},
                'ks_stat': float(stats.kstest(data_positive, 'expon', args=(loc, scale))[0])
            }
        except:
            results['Exponential'] = {'error': 'Failed to fit'}

        # Lognormal Distribution
        try:
            shape, loc, scale = lognorm.fit(data_positive)
            results['Lognormal'] = {
                'parameters': {'shape': float(shape), 'loc': float(loc), 'scale': float(scale)},
                'ks_stat': float(stats.kstest(data_positive, 'lognorm', args=(shape, loc, scale))[0])
            }
        except:
            results['Lognormal'] = {'error': 'Failed to fit'}

        # Weibull Distribution
        try:
            shape, loc, scale = weibull_min.fit(data_positive)
            results['Weibull'] = {
                'parameters': {'shape': float(shape), 'loc': float(loc), 'scale': float(scale)},
                'ks_stat': float(stats.kstest(data_positive, 'weibull_min', args=(shape, loc, scale))[0])
            }
        except:
            results['Weibull'] = {'error': 'Failed to fit'}

        # Beta Distribution
        try:
            # Beta requires data in (0, 1)
            data_normalized = (data - data.min()) / (data.max() - data.min())
            a, b, loc, scale = beta.fit(data_normalized)
            results['Beta'] = {
                'parameters': {'a': float(a), 'b': float(b), 'loc': float(loc), 'scale': float(scale)},
                'ks_stat': float(stats.kstest(data_normalized, 'beta', args=(a, b, loc, scale))[0])
            }
        except:
            results['Beta'] = {'error': 'Failed to fit'}

        return results

    def plot_distribution_fit(self, column: str, distributions: List[str] = None) -> plt.Figure:
        """
        Plot data histogram with fitted distribution curves
        """
        if self.df is None:
            return None

        data = self.df[column].dropna().values

        if distributions is None:
            distributions = ['Normal', 'Gamma', 'Lognormal']

        fit_results = self.fit_distributions(column)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram
        counts, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black', label='Data')

        # Plot fitted distributions
        x = np.linspace(data.min(), data.max(), 200)

        colors = ['red', 'green', 'orange', 'purple', 'brown']
        color_idx = 0

        for dist_name in distributions:
            if dist_name not in fit_results or 'error' in fit_results[dist_name]:
                continue

            try:
                if dist_name == 'Normal':
                    params = fit_results['Normal']['parameters']
                    y = norm.pdf(x, params['mu'], params['sigma'])
                elif dist_name == 'Gamma':
                    params = fit_results['Gamma']['parameters']
                    y = gamma.pdf(x, params['shape'], params['loc'], params['scale'])
                elif dist_name == 'Exponential':
                    params = fit_results['Exponential']['parameters']
                    y = expon.pdf(x, params['loc'], params['scale'])
                elif dist_name == 'Lognormal':
                    params = fit_results['Lognormal']['parameters']
                    y = lognorm.pdf(x, params['shape'], params['loc'], params['scale'])
                elif dist_name == 'Weibull':
                    params = fit_results['Weibull']['parameters']
                    y = weibull_min.pdf(x, params['shape'], params['loc'], params['scale'])
                elif dist_name == 'Beta':
                    params = fit_results['Beta']['parameters']
                    x_norm = (x - x.min()) / (x.max() - x.min())
                    y = beta.pdf(x_norm, params['a'], params['b'], params['loc'], params['scale'])
                else:
                    continue

                ax.plot(x, y, color=colors[color_idx % len(colors)], linewidth=2, label=dist_name)
                color_idx += 1
            except:
                pass

        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Distribution Fit: {column}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def distribution_analysis(self, columns: List[str]) -> Dict[str, Dict]:
        """
        Comprehensive distribution analysis
        """
        if self.df is None:
            return {}

        results = {}

        for col in columns:
            data = self.df[col].dropna()

            # Normality tests
            shapiro_stat, shapiro_p = shapiro(data[:5000]) if len(data) >= 3 else (np.nan, np.nan)
            kstest_stat, kstest_p = stats.kstest(data, 'norm')

            results[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'shapiro_stat': float(shapiro_stat),
                'shapiro_p': float(shapiro_p),
                'ks_stat': float(kstest_stat),
                'ks_p': float(kstest_p),
                'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
            }

        return results

    def plot_distributions(self, columns: List[str]) -> plt.Figure:
        """Plot histograms, Q-Q plots, and box plots"""
        if self.df is None or not columns:
            return None

        n_features = len(columns)
        fig, axes = plt.subplots(n_features, 3, figsize=(15, 4*n_features))

        if n_features == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(columns):
            data = self.df[col].dropna()

            # Histogram
            axes[i, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            axes[i, 0].set_title(f'{col} - Histogram')
            axes[i, 0].set_xlabel('Value')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].grid(True, alpha=0.3)

            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{col} - Q-Q Plot')
            axes[i, 1].grid(True, alpha=0.3)

            # Box plot
            axes[i, 2].boxplot(data, vert=True)
            axes[i, 2].set_title(f'{col} - Box Plot')
            axes[i, 2].set_ylabel('Value')
            axes[i, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # =========================================================================
    # ENHANCED PROBABILITY DISTRIBUTION ANALYSIS
    # =========================================================================

    def fit_extended_distributions(self, column: str, distributions: List[str] = None) -> Dict[str, Any]:
        """
        Fit extended set of probability distributions with comprehensive goodness-of-fit tests.
        
        Args:
            column: Column name to analyze
            distributions: List of distributions to fit. Options:
                          ['normal', 't', 'chi2', 'uniform', 'poisson', 'binomial',
                           'gamma', 'exponential', 'lognormal', 'weibull', 'beta',
                           'pareto', 'cauchy', 'laplace', 'logistic']
                          If None, fits common continuous distributions.
        
        Returns:
            Dictionary with fitted parameters, GOF tests, and best fit
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        n = len(data)
        
        if n < 10:
            return {'error': 'Need at least 10 data points'}
        
        if distributions is None:
            distributions = ['normal', 't', 'gamma', 'exponential', 'lognormal', 
                           'weibull', 'uniform', 'laplace', 'logistic']
        
        results = {'data_summary': {
            'n': n,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }}
        
        # Ensure positive data for certain distributions
        data_positive = data[data > 0] if np.any(data <= 0) else data
        
        dist_results = {}
        
        for dist_name in distributions:
            try:
                if dist_name == 'normal':
                    params = norm.fit(data)
                    dist = norm(*params)
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=params)
                    ad_stat = stats.anderson(data, 'norm')
                    dist_results['normal'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'anderson_statistic': float(ad_stat.statistic),
                        'anderson_critical_5pct': float(ad_stat.critical_values[2]),
                        'aic': self._aic(data, dist, 2),
                        'bic': self._bic(data, dist, 2)
                    }
                
                elif dist_name == 't':
                    params = stats.t.fit(data)
                    dist = stats.t(*params)
                    ks_stat, ks_p = stats.kstest(data, 't', args=params)
                    dist_results['t'] = {
                        'parameters': {'df': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data, dist, 3),
                        'bic': self._bic(data, dist, 3)
                    }
                
                elif dist_name == 'chi2' and np.all(data >= 0):
                    params = stats.chi2.fit(data_positive)
                    dist = stats.chi2(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'chi2', args=params)
                    dist_results['chi2'] = {
                        'parameters': {'df': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data_positive, dist, 3),
                        'bic': self._bic(data_positive, dist, 3)
                    }
                
                elif dist_name == 'uniform':
                    params = stats.uniform.fit(data)
                    dist = stats.uniform(*params)
                    ks_stat, ks_p = stats.kstest(data, 'uniform', args=params)
                    dist_results['uniform'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data, dist, 2),
                        'bic': self._bic(data, dist, 2)
                    }
                
                elif dist_name == 'gamma' and len(data_positive) > 0:
                    params = gamma.fit(data_positive)
                    dist = gamma(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'gamma', args=params)
                    dist_results['gamma'] = {
                        'parameters': {'a': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data_positive, dist, 3),
                        'bic': self._bic(data_positive, dist, 3)
                    }
                
                elif dist_name == 'exponential' and len(data_positive) > 0:
                    params = expon.fit(data_positive)
                    dist = expon(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'expon', args=params)
                    ad_stat = stats.anderson(data_positive, 'expon')
                    dist_results['exponential'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'anderson_statistic': float(ad_stat.statistic),
                        'aic': self._aic(data_positive, dist, 2),
                        'bic': self._bic(data_positive, dist, 2)
                    }
                
                elif dist_name == 'lognormal' and len(data_positive) > 0:
                    params = lognorm.fit(data_positive)
                    dist = lognorm(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'lognorm', args=params)
                    dist_results['lognormal'] = {
                        'parameters': {'s': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data_positive, dist, 3),
                        'bic': self._bic(data_positive, dist, 3)
                    }
                
                elif dist_name == 'weibull' and len(data_positive) > 0:
                    params = weibull_min.fit(data_positive)
                    dist = weibull_min(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'weibull_min', args=params)
                    dist_results['weibull'] = {
                        'parameters': {'c': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data_positive, dist, 3),
                        'bic': self._bic(data_positive, dist, 3)
                    }
                
                elif dist_name == 'laplace':
                    params = stats.laplace.fit(data)
                    dist = stats.laplace(*params)
                    ks_stat, ks_p = stats.kstest(data, 'laplace', args=params)
                    dist_results['laplace'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data, dist, 2),
                        'bic': self._bic(data, dist, 2)
                    }
                
                elif dist_name == 'logistic':
                    params = stats.logistic.fit(data)
                    dist = stats.logistic(*params)
                    ks_stat, ks_p = stats.kstest(data, 'logistic', args=params)
                    dist_results['logistic'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data, dist, 2),
                        'bic': self._bic(data, dist, 2)
                    }
                
                elif dist_name == 'pareto' and len(data_positive) > 0:
                    params = stats.pareto.fit(data_positive)
                    dist = stats.pareto(*params)
                    ks_stat, ks_p = stats.kstest(data_positive, 'pareto', args=params)
                    dist_results['pareto'] = {
                        'parameters': {'b': params[0], 'loc': params[1], 'scale': params[2]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data_positive, dist, 3),
                        'bic': self._bic(data_positive, dist, 3)
                    }
                
                elif dist_name == 'cauchy':
                    params = stats.cauchy.fit(data)
                    dist = stats.cauchy(*params)
                    ks_stat, ks_p = stats.kstest(data, 'cauchy', args=params)
                    dist_results['cauchy'] = {
                        'parameters': {'loc': params[0], 'scale': params[1]},
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_p),
                        'aic': self._aic(data, dist, 2),
                        'bic': self._bic(data, dist, 2)
                    }
            
            except Exception as e:
                dist_results[dist_name] = {'error': str(e)}
        
        results['distributions'] = dist_results
        
        # Find best fit by AIC
        valid_fits = {k: v for k, v in dist_results.items() if 'aic' in v}
        if valid_fits:
            best_by_aic = min(valid_fits.items(), key=lambda x: x[1]['aic'])
            best_by_ks = max(valid_fits.items(), key=lambda x: x[1].get('ks_pvalue', 0))
            results['best_fit_aic'] = best_by_aic[0]
            results['best_fit_ks'] = best_by_ks[0]
        
        return results

    def _aic(self, data: np.ndarray, dist, k: int) -> float:
        """Calculate Akaike Information Criterion"""
        try:
            log_likelihood = np.sum(dist.logpdf(data))
            return 2 * k - 2 * log_likelihood
        except:
            return float('inf')

    def _bic(self, data: np.ndarray, dist, k: int) -> float:
        """Calculate Bayesian Information Criterion"""
        try:
            n = len(data)
            log_likelihood = np.sum(dist.logpdf(data))
            return k * np.log(n) - 2 * log_likelihood
        except:
            return float('inf')

    def random_variable_analysis(self, column: str) -> Dict[str, Any]:
        """
        Comprehensive random variable analysis including moments and quantiles.
        
        Args:
            column: Column name to analyze
            
        Returns:
            Dictionary with complete probability analysis
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        n = len(data)
        
        if n < 3:
            return {'error': 'Need at least 3 data points'}
        
        # Moments
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        std = np.std(data, ddof=1)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)  # Excess kurtosis
        
        # Quantiles
        quantiles = {
            '1%': float(np.percentile(data, 1)),
            '5%': float(np.percentile(data, 5)),
            '10%': float(np.percentile(data, 10)),
            '25%': float(np.percentile(data, 25)),
            '50%': float(np.percentile(data, 50)),
            '75%': float(np.percentile(data, 75)),
            '90%': float(np.percentile(data, 90)),
            '95%': float(np.percentile(data, 95)),
            '99%': float(np.percentile(data, 99))
        }
        
        # IQR and outlier bounds
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Coefficient of variation
        cv = (std / mean * 100) if mean != 0 else float('inf')
        
        # Standard error of mean
        sem = std / np.sqrt(n)
        
        # 95% CI for mean
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_lower = mean - t_crit * sem
        ci_upper = mean + t_crit * sem
        
        return {
            'n': n,
            'moments': {
                'mean': float(mean),
                'variance': float(var),
                'std': float(std),
                'skewness': float(skewness),
                'excess_kurtosis': float(kurtosis)
            },
            'quantiles': quantiles,
            'iqr': float(iqr),
            'outlier_bounds': {
                'lower_fence': float(lower_fence),
                'upper_fence': float(upper_fence)
            },
            'coefficient_of_variation': float(cv),
            'standard_error': float(sem),
            'confidence_interval_95': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'range': {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data))
            }
        }

    def empirical_cdf(self, column: str, n_points: int = 100) -> Dict[str, Any]:
        """
        Compute empirical cumulative distribution function.
        
        Args:
            column: Column name
            n_points: Number of points for smooth CDF
            
        Returns:
            Dictionary with x values and CDF values
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        # Empirical CDF
        ecdf_y = np.arange(1, n + 1) / n
        
        # Smooth CDF for plotting
        x_smooth = np.linspace(data.min(), data.max(), n_points)
        cdf_smooth = np.array([np.mean(data <= x) for x in x_smooth])
        
        return {
            'x': data_sorted.tolist(),
            'cdf': ecdf_y.tolist(),
            'x_smooth': x_smooth.tolist(),
            'cdf_smooth': cdf_smooth.tolist(),
            'n': n
        }

    def qq_analysis(self, column: str, distribution: str = 'normal') -> Dict[str, Any]:
        """
        Quantile-Quantile analysis against a theoretical distribution.
        
        Args:
            column: Column name
            distribution: 'normal', 't', 'exponential', 'uniform'
            
        Returns:
            Dictionary with QQ data and correlation
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        # Theoretical quantiles
        probs = (np.arange(1, n + 1) - 0.5) / n
        
        if distribution == 'normal':
            theoretical = norm.ppf(probs)
        elif distribution == 't':
            theoretical = stats.t.ppf(probs, df=n-1)
        elif distribution == 'exponential':
            theoretical = expon.ppf(probs)
        elif distribution == 'uniform':
            theoretical = stats.uniform.ppf(probs)
        else:
            return {'error': f'Unknown distribution: {distribution}'}
        
        # Correlation coefficient (deviation from line indicates departure from distribution)
        correlation = np.corrcoef(theoretical, data_sorted)[0, 1]
        
        # Fit line
        slope, intercept = np.polyfit(theoretical, data_sorted, 1)
        
        return {
            'theoretical_quantiles': theoretical.tolist(),
            'sample_quantiles': data_sorted.tolist(),
            'correlation': float(correlation),
            'line_slope': float(slope),
            'line_intercept': float(intercept),
            'distribution': distribution
        }

    def plot_probability_analysis(self, column: str) -> plt.Figure:
        """
        Comprehensive probability distribution plot panel.
        
        Args:
            column: Column to analyze
            
        Returns:
            Figure with histogram, CDF, QQ-plot, and box plot
        """
        if self.df is None:
            return None
        
        data = self.df[column].dropna().values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram with fitted normal
        ax = axes[0, 0]
        ax.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        x = np.linspace(data.min(), data.max(), 100)
        mu, sigma = norm.fit(data)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{column} - Histogram with Normal Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Empirical CDF
        ax = axes[0, 1]
        data_sorted = np.sort(data)
        ecdf = np.arange(1, len(data) + 1) / len(data)
        ax.step(data_sorted, ecdf, where='post', color='steelblue', linewidth=2, label='Empirical CDF')
        ax.plot(x, norm.cdf(x, mu, sigma), 'r--', linewidth=2, label='Normal CDF')
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{column} - Empirical CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'{column} - Q-Q Plot (Normal)')
        ax.grid(True, alpha=0.3)
        
        # 4. Box plot with violin
        ax = axes[1, 1]
        parts = ax.violinplot(data, positions=[1], showmeans=True, showextrema=True)
        parts['bodies'][0].set_facecolor('steelblue')
        parts['bodies'][0].set_alpha(0.7)
        ax.boxplot(data, positions=[2])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Violin', 'Box'])
        ax.set_ylabel('Value')
        ax.set_title(f'{column} - Distribution Shape')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    # =========================================================================
    # STATISTICAL HYPOTHESIS TESTS
    # =========================================================================

    def ttest_independent(self, column1: str, column2: str,
                         equal_var: bool = True) -> Dict[str, Any]:
        """
        Independent Samples t-test

        Tests if two independent groups have different means.
        Assumes normal distributions.

        Returns:
            Dictionary with test statistic, p-value, and interpretation
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data1 = self.df[column1].dropna().values
        data2 = self.df[column2].dropna().values

        t_stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)

        results = {
            'test': 'Independent t-test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'n1': len(data1),
            'n2': len(data2),
            'mean1': float(data1.mean()),
            'mean2': float(data2.mean()),
            'mean_diff': float(data1.mean() - data2.mean()),
            'significant': p_value < 0.05,
            'equal_var_assumed': equal_var,
            'interpretation': 'Means are significantly different' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def ttest_paired(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Paired Samples t-test

        Tests if paired observations have different means.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data1 = self.df[column1].dropna()
        data2 = self.df[column2].dropna()

        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx].values
        data2 = data2.loc[common_idx].values

        t_stat, p_value = ttest_rel(data1, data2)

        results = {
            'test': 'Paired t-test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'n': len(data1),
            'mean_diff': float((data1 - data2).mean()),
            'std_diff': float((data1 - data2).std()),
            'significant': p_value < 0.05,
            'interpretation': 'Means are significantly different' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def mann_whitney_u(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Mann-Whitney U Test (non-parametric)

        Tests if two independent groups have different distributions.
        Does NOT assume normality.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data1 = self.df[column1].dropna().values
        data2 = self.df[column2].dropna().values

        u_stat, p_value = mannwhitneyu(data1, data2)

        results = {
            'test': 'Mann-Whitney U',
            'statistic': float(u_stat),
            'p_value': float(p_value),
            'n1': len(data1),
            'n2': len(data2),
            'median1': float(np.median(data1)),
            'median2': float(np.median(data2)),
            'significant': p_value < 0.05,
            'interpretation': 'Distributions are significantly different' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def wilcoxon_signed_rank(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Wilcoxon Signed-Rank Test (non-parametric)

        Non-parametric alternative to paired t-test.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data1 = self.df[column1].dropna()
        data2 = self.df[column2].dropna()

        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx].values
        data2 = data2.loc[common_idx].values

        w_stat, p_value = wilcoxon(data1, data2)

        results = {
            'test': 'Wilcoxon Signed-Rank',
            'statistic': float(w_stat),
            'p_value': float(p_value),
            'n': len(data1),
            'significant': p_value < 0.05,
            'interpretation': 'Distributions are significantly different' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def kruskal_wallis(self, column_names: List[str]) -> Dict[str, Any]:
        """
        Kruskal-Wallis Test

        Non-parametric test for comparing 3+ independent groups.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data_list = [self.df[col].dropna().values for col in column_names]

        h_stat, p_value = kruskal(*data_list)

        results = {
            'test': 'Kruskal-Wallis',
            'statistic': float(h_stat),
            'p_value': float(p_value),
            'n_groups': len(column_names),
            'group_names': column_names,
            'significant': p_value < 0.05,
            'interpretation': 'Groups have significantly different distributions' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def anova_oneway(self, column_names: List[str]) -> Dict[str, Any]:
        """
        One-Way ANOVA (parametric)

        Tests if 3+ independent groups have different means.
        Assumes normal distributions and equal variances.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data_list = [self.df[col].dropna().values for col in column_names]

        f_stat, p_value = f_oneway(*data_list)

        results = {
            'test': 'One-Way ANOVA',
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'n_groups': len(column_names),
            'group_names': column_names,
            'significant': p_value < 0.05,
            'interpretation': 'Groups have significantly different means' if p_value < 0.05 else 'No significant difference'
        }

        self.last_test_results = results
        return results

    def anova_twoway(self, data_column: str, factor1: str, factor2: str) -> Dict[str, Any]:
        """
        Two-Way ANOVA with interaction effects.
        
        Tests main effects of two factors and their interaction on a continuous outcome.
        
        Args:
            data_column: Column with continuous dependent variable
            factor1: First categorical factor (grouping variable)
            factor2: Second categorical factor (grouping variable)
            
        Returns:
            Dictionary with F-statistics, p-values for main effects and interaction
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            from statsmodels.stats.anova import anova_lm
            
            # Prepare data
            df_clean = self.df[[data_column, factor1, factor2]].dropna()
            
            if len(df_clean) < 10:
                return {'error': 'Need at least 10 observations for two-way ANOVA'}
            
            # Fit model with interaction
            formula = f'{data_column} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'
            model = ols(formula, data=df_clean).fit()
            anova_table = anova_lm(model, typ=2)
            
            # Extract results
            results = {
                'test': 'Two-Way ANOVA',
                'data_column': data_column,
                'factor1': factor1,
                'factor2': factor2,
                'n_observations': len(df_clean),
                'effects': {}
            }
            
            # Main effect of factor1
            f1_key = f'C({factor1})'
            if f1_key in anova_table.index:
                results['effects']['factor1'] = {
                    'name': factor1,
                    'sum_sq': float(anova_table.loc[f1_key, 'sum_sq']),
                    'df': float(anova_table.loc[f1_key, 'df']),
                    'F': float(anova_table.loc[f1_key, 'F']),
                    'p_value': float(anova_table.loc[f1_key, 'PR(>F)']),
                    'significant': anova_table.loc[f1_key, 'PR(>F)'] < 0.05
                }
            
            # Main effect of factor2
            f2_key = f'C({factor2})'
            if f2_key in anova_table.index:
                results['effects']['factor2'] = {
                    'name': factor2,
                    'sum_sq': float(anova_table.loc[f2_key, 'sum_sq']),
                    'df': float(anova_table.loc[f2_key, 'df']),
                    'F': float(anova_table.loc[f2_key, 'F']),
                    'p_value': float(anova_table.loc[f2_key, 'PR(>F)']),
                    'significant': anova_table.loc[f2_key, 'PR(>F)'] < 0.05
                }
            
            # Interaction effect
            int_key = f'C({factor1}):C({factor2})'
            if int_key in anova_table.index:
                results['effects']['interaction'] = {
                    'name': f'{factor1} x {factor2}',
                    'sum_sq': float(anova_table.loc[int_key, 'sum_sq']),
                    'df': float(anova_table.loc[int_key, 'df']),
                    'F': float(anova_table.loc[int_key, 'F']),
                    'p_value': float(anova_table.loc[int_key, 'PR(>F)']),
                    'significant': anova_table.loc[int_key, 'PR(>F)'] < 0.05
                }
            
            # Residual
            if 'Residual' in anova_table.index:
                results['residual'] = {
                    'sum_sq': float(anova_table.loc['Residual', 'sum_sq']),
                    'df': float(anova_table.loc['Residual', 'df'])
                }
            
            # R-squared
            results['r_squared'] = float(model.rsquared)
            results['adj_r_squared'] = float(model.rsquared_adj)
            
            # Summary interpretation
            sig_effects = [name for name, info in results['effects'].items() if info.get('significant')]
            if sig_effects:
                results['interpretation'] = f"Significant effects: {', '.join(sig_effects)}"
            else:
                results['interpretation'] = "No significant effects detected"
            
            self.last_test_results = results
            return results
            
        except Exception as e:
            return {'error': str(e)}

    def anova_repeated_measures(self, data_column: str, subject_column: str, 
                                 within_factor: str) -> Dict[str, Any]:
        """
        Repeated-Measures ANOVA (within-subjects).
        
        Tests differences when the same subjects are measured under different conditions.
        
        Args:
            data_column: Column with continuous dependent variable
            subject_column: Column identifying subjects
            within_factor: Column with within-subjects factor (conditions)
            
        Returns:
            Dictionary with F-statistic, p-value, and sphericity test
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        try:
            from scipy import stats as scipy_stats
            
            # Prepare data
            df_clean = self.df[[data_column, subject_column, within_factor]].dropna()
            
            # Get unique conditions and subjects
            conditions = df_clean[within_factor].unique()
            subjects = df_clean[subject_column].unique()
            n_conditions = len(conditions)
            n_subjects = len(subjects)
            
            if n_conditions < 2:
                return {'error': 'Need at least 2 conditions for repeated-measures ANOVA'}
            if n_subjects < 3:
                return {'error': 'Need at least 3 subjects for repeated-measures ANOVA'}
            
            # Create data matrix (subjects x conditions)
            data_matrix = []
            for subj in subjects:
                subj_data = []
                for cond in conditions:
                    vals = df_clean[(df_clean[subject_column] == subj) & 
                                   (df_clean[within_factor] == cond)][data_column].values
                    if len(vals) > 0:
                        subj_data.append(vals.mean())  # Average if multiple measurements
                    else:
                        subj_data.append(np.nan)
                data_matrix.append(subj_data)
            
            data_matrix = np.array(data_matrix)
            
            # Remove subjects with missing data
            valid_mask = ~np.any(np.isnan(data_matrix), axis=1)
            data_matrix = data_matrix[valid_mask]
            n_subjects_valid = data_matrix.shape[0]
            
            if n_subjects_valid < 3:
                return {'error': 'Not enough complete subjects for analysis'}
            
            # Calculate repeated-measures ANOVA manually
            # Grand mean
            grand_mean = np.mean(data_matrix)
            
            # Between-subjects sum of squares
            subject_means = np.mean(data_matrix, axis=1)
            ss_between_subjects = n_conditions * np.sum((subject_means - grand_mean) ** 2)
            
            # Within-subjects (conditions) sum of squares
            condition_means = np.mean(data_matrix, axis=0)
            ss_within = n_subjects_valid * np.sum((condition_means - grand_mean) ** 2)
            
            # Error sum of squares (interaction)
            ss_error = 0
            for i in range(n_subjects_valid):
                for j in range(n_conditions):
                    ss_error += (data_matrix[i, j] - subject_means[i] - condition_means[j] + grand_mean) ** 2
            
            # Degrees of freedom
            df_conditions = n_conditions - 1
            df_subjects = n_subjects_valid - 1
            df_error = df_conditions * df_subjects
            
            # Mean squares
            ms_conditions = ss_within / df_conditions
            ms_error = ss_error / df_error
            
            # F-statistic
            f_stat = ms_conditions / ms_error if ms_error > 0 else float('inf')
            p_value = 1 - scipy_stats.f.cdf(f_stat, df_conditions, df_error)
            
            # Effect size (partial eta-squared)
            eta_squared = ss_within / (ss_within + ss_error)
            
            # Mauchly's sphericity test (simplified)
            # This is an approximation
            variances = np.var(data_matrix, axis=0, ddof=1)
            covariances = np.cov(data_matrix.T)
            sphericity_approx = np.std(variances) / np.mean(variances) if np.mean(variances) > 0 else 0
            
            results = {
                'test': 'Repeated-Measures ANOVA',
                'data_column': data_column,
                'within_factor': within_factor,
                'n_subjects': n_subjects_valid,
                'n_conditions': n_conditions,
                'conditions': conditions.tolist(),
                'ss_between_subjects': float(ss_between_subjects),
                'ss_within_conditions': float(ss_within),
                'ss_error': float(ss_error),
                'df_conditions': df_conditions,
                'df_error': df_error,
                'ms_conditions': float(ms_conditions),
                'ms_error': float(ms_error),
                'F': float(f_stat),
                'p_value': float(p_value),
                'partial_eta_squared': float(eta_squared),
                'sphericity_concern': sphericity_approx > 0.3,
                'significant': p_value < 0.05,
                'interpretation': f"Conditions differ significantly (F={f_stat:.2f}, p={p_value:.4f})" if p_value < 0.05 else "No significant difference between conditions"
            }
            
            self.last_test_results = results
            return results
            
        except Exception as e:
            return {'error': str(e)}

    def posthoc_tukey(self, data_column: str, group_column: str) -> Dict[str, Any]:
        """
        Tukey's Honest Significant Difference (HSD) post-hoc test.
        
        Performs pairwise comparisons after significant ANOVA.
        Controls family-wise error rate.
        
        Args:
            data_column: Column with continuous dependent variable
            group_column: Column with group labels
            
        Returns:
            Dictionary with pairwise comparison results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        try:
            from scipy import stats as scipy_stats
            from itertools import combinations
            
            df_clean = self.df[[data_column, group_column]].dropna()
            groups = df_clean[group_column].unique()
            n_groups = len(groups)
            
            if n_groups < 2:
                return {'error': 'Need at least 2 groups for post-hoc tests'}
            
            # Get group data
            group_data = {g: df_clean[df_clean[group_column] == g][data_column].values 
                         for g in groups}
            group_means = {g: np.mean(v) for g, v in group_data.items()}
            group_ns = {g: len(v) for g, v in group_data.items()}
            
            # Calculate MSE (pooled variance)
            n_total = sum(group_ns.values())
            grand_mean = np.mean(df_clean[data_column])
            
            ss_within = sum(np.sum((group_data[g] - group_means[g]) ** 2) for g in groups)
            df_within = n_total - n_groups
            mse = ss_within / df_within if df_within > 0 else 1
            
            # Tukey HSD comparisons
            comparisons = []
            for g1, g2 in combinations(groups, 2):
                mean_diff = group_means[g1] - group_means[g2]
                
                # Standard error
                se = np.sqrt(mse * (1/group_ns[g1] + 1/group_ns[g2]) / 2)
                
                # Q statistic (studentized range)
                q_stat = abs(mean_diff) / se if se > 0 else 0
                
                # Use studentized range distribution approximation
                # For exact, would need special tables or scipy.stats.studentized_range
                # Approximate p-value using t-distribution with Bonferroni correction
                n_comparisons = n_groups * (n_groups - 1) / 2
                t_stat = q_stat / np.sqrt(2)
                p_raw = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df_within))
                p_adj = min(1.0, p_raw * n_comparisons)  # Bonferroni approximation
                
                comparisons.append({
                    'group1': str(g1),
                    'group2': str(g2),
                    'mean_diff': float(mean_diff),
                    'std_error': float(se),
                    'q_statistic': float(q_stat),
                    'p_value': float(p_adj),
                    'significant': p_adj < 0.05,
                    'ci_lower': float(mean_diff - 1.96 * se),
                    'ci_upper': float(mean_diff + 1.96 * se)
                })
            
            results = {
                'test': "Tukey's HSD",
                'data_column': data_column,
                'group_column': group_column,
                'n_groups': n_groups,
                'n_comparisons': len(comparisons),
                'mse': float(mse),
                'df_within': df_within,
                'comparisons': comparisons,
                'group_means': {str(k): float(v) for k, v in group_means.items()},
                'group_sizes': {str(k): int(v) for k, v in group_ns.items()}
            }
            
            # Summary
            sig_pairs = [f"{c['group1']} vs {c['group2']}" for c in comparisons if c['significant']]
            if sig_pairs:
                results['interpretation'] = f"Significant differences: {', '.join(sig_pairs)}"
            else:
                results['interpretation'] = "No significant pairwise differences"
            
            self.last_test_results = results
            return results
            
        except Exception as e:
            return {'error': str(e)}

    def posthoc_bonferroni(self, data_column: str, group_column: str, 
                           alpha: float = 0.05) -> Dict[str, Any]:
        """
        Bonferroni-corrected pairwise t-tests.
        
        Conservative multiple comparison correction.
        
        Args:
            data_column: Column with continuous dependent variable
            group_column: Column with group labels
            alpha: Family-wise significance level
            
        Returns:
            Dictionary with pairwise comparison results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        try:
            from scipy import stats as scipy_stats
            from itertools import combinations
            
            df_clean = self.df[[data_column, group_column]].dropna()
            groups = df_clean[group_column].unique()
            n_groups = len(groups)
            
            if n_groups < 2:
                return {'error': 'Need at least 2 groups for post-hoc tests'}
            
            # Get group data
            group_data = {g: df_clean[df_clean[group_column] == g][data_column].values 
                         for g in groups}
            group_means = {g: np.mean(v) for g, v in group_data.items()}
            group_stds = {g: np.std(v, ddof=1) for g, v in group_data.items()}
            group_ns = {g: len(v) for g, v in group_data.items()}
            
            # Number of comparisons for Bonferroni correction
            n_comparisons = n_groups * (n_groups - 1) // 2
            alpha_corrected = alpha / n_comparisons
            
            # Pairwise t-tests
            comparisons = []
            for g1, g2 in combinations(groups, 2):
                # Independent samples t-test
                t_stat, p_raw = scipy_stats.ttest_ind(group_data[g1], group_data[g2])
                p_adj = min(1.0, p_raw * n_comparisons)
                
                mean_diff = group_means[g1] - group_means[g2]
                
                # Pooled standard error
                n1, n2 = group_ns[g1], group_ns[g2]
                s1, s2 = group_stds[g1], group_stds[g2]
                se = np.sqrt(s1**2/n1 + s2**2/n2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                comparisons.append({
                    'group1': str(g1),
                    'group2': str(g2),
                    'mean_diff': float(mean_diff),
                    'std_error': float(se),
                    't_statistic': float(t_stat),
                    'p_raw': float(p_raw),
                    'p_adjusted': float(p_adj),
                    'significant': p_adj < alpha,
                    'cohens_d': float(cohens_d),
                    'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
                })
            
            results = {
                'test': 'Bonferroni Pairwise t-tests',
                'data_column': data_column,
                'group_column': group_column,
                'alpha': alpha,
                'alpha_corrected': float(alpha_corrected),
                'n_groups': n_groups,
                'n_comparisons': n_comparisons,
                'comparisons': comparisons,
                'group_means': {str(k): float(v) for k, v in group_means.items()},
                'group_stds': {str(k): float(v) for k, v in group_stds.items()},
                'group_sizes': {str(k): int(v) for k, v in group_ns.items()}
            }
            
            # Summary
            sig_pairs = [f"{c['group1']} vs {c['group2']}" for c in comparisons if c['significant']]
            if sig_pairs:
                results['interpretation'] = f"Significant differences (Bonferroni α={alpha_corrected:.4f}): {', '.join(sig_pairs)}"
            else:
                results['interpretation'] = "No significant pairwise differences after Bonferroni correction"
            
            self.last_test_results = results
            return results
            
        except Exception as e:
            return {'error': str(e)}

    def levene_test(self, column_names: List[str]) -> Dict[str, Any]:
        """
        Levene's Test for Equality of Variances

        Tests if multiple groups have equal variances.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        data_list = [self.df[col].dropna().values for col in column_names]

        stat, p_value = levene(*data_list)

        results = {
            'test': "Levene's Test",
            'statistic': float(stat),
            'p_value': float(p_value),
            'n_groups': len(column_names),
            'equal_variances': p_value > 0.05,
            'interpretation': 'Variances are equal' if p_value > 0.05 else 'Variances are not equal'
        }

        self.last_test_results = results
        return results

    def chi_square_test(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Chi-Square Test of Independence

        Tests association between two categorical variables.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        # Create contingency table
        contingency = pd.crosstab(self.df[column1], self.df[column2])

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        results = {
            'test': 'Chi-Square',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'significant': p_value < 0.05,
            'contingency_table': contingency.values.tolist(),
            'interpretation': 'Variables are associated' if p_value < 0.05 else 'No significant association'
        }

        self.last_test_results = results
        return results

    # =========================================================================
    # OUTLIER DETECTION
    # =========================================================================

    def outlier_detection(self, columns: List[str], method: str = 'iqr',
                          iqr_multiplier: float = 1.5, zscore_threshold: float = 3.0) -> Dict[str, Dict]:
        """
        Detect outliers using IQR or Z-score method

        Args:
            columns: List of column names to analyze
            method: 'iqr' or 'zscore'
            iqr_multiplier: Multiplier for IQR bounds (default 1.5, use 3.0 for "far" outliers)
            zscore_threshold: Z-score threshold (default 3.0, use 2.5 for stricter detection)

        Returns:
            Dict with outlier info per column including indices, bounds, counts
        """
        if self.df is None:
            return {}

        results = {}

        if method == 'iqr':
            data_matrix = self.df[columns].values
            counts, percentages = _accel_outliers(data_matrix, multiplier=iqr_multiplier)

            for i, col in enumerate(columns):
                data = self.df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                outliers = (data < lower_bound) | (data > upper_bound)

                results[col] = {
                    'n_outliers': int(counts[i]),
                    'percentage': float(percentages[i]),
                    'outlier_indices': data[outliers].index.tolist(),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': f'IQR (×{iqr_multiplier})'
                }
        else:  # zscore
            for col in columns:
                data = self.df[col].dropna()
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > zscore_threshold
                n_outliers = outliers.sum()

                # Calculate bounds: values at z=threshold from mean
                mean_val = data.mean()
                std_val = data.std()
                lower_bound = mean_val - zscore_threshold * std_val
                upper_bound = mean_val + zscore_threshold * std_val

                results[col] = {
                    'n_outliers': int(n_outliers),
                    'percentage': float((n_outliers / len(data)) * 100),
                    'outlier_indices': data[outliers].index.tolist(),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': f'Z-score (>{zscore_threshold}σ)'
                }

        return results

    def plot_boxplots(self, columns: List[str], outlier_results: Dict = None) -> plt.Figure:
        """Plot box plots with actual detected outliers highlighted

        Args:
            columns: List of column names to plot
            outlier_results: Dict from outlier_detection() with outlier_indices for each column
        """
        if self.df is None or not columns:
            return None

        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i]
            data = self.df[col].dropna()

            # Plot box without outlier points (we'll add our own)
            bp = ax.boxplot(data, showfliers=False, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)

            # Get outlier info if provided
            if outlier_results and col in outlier_results:
                info = outlier_results[col]
                outlier_indices = info.get('outlier_indices', [])
                is_outlier = self.df.index.isin(outlier_indices)

                # Plot normal points (small blue)
                normal_data = self.df.loc[~is_outlier, col].dropna()
                ax.scatter([1]*len(normal_data), normal_data,
                          c='steelblue', s=15, alpha=0.4, label='Normal')

                # Plot outliers (large red X)
                if len(outlier_indices) > 0:
                    outlier_data = self.df.loc[is_outlier, col].dropna()
                    ax.scatter([1]*len(outlier_data), outlier_data,
                              c='red', s=80, marker='x', linewidths=2,
                              label=f'Outliers ({len(outlier_indices)})', zorder=5)

                # Add IQR bounds as horizontal lines
                if 'lower_bound' in info:
                    ax.axhline(info['lower_bound'], color='orange', linestyle='--',
                              alpha=0.8, label='IQR bounds')
                if 'upper_bound' in info:
                    ax.axhline(info['upper_bound'], color='orange', linestyle='--', alpha=0.8)
            else:
                # No outlier info, just scatter all points
                ax.scatter([1]*len(data), data, c='steelblue', s=15, alpha=0.4)

            ax.set_title(col)
            ax.set_xticks([])
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        fig.suptitle('🔍 Outlier Detection: Red X = Detected Outliers, Orange = IQR Bounds', fontsize=12)
        plt.tight_layout()

        return fig

    # =========================================================================
    # CORRELATION & ASSOCIATION
    # =========================================================================

    def pairwise_correlations(self, columns: List[str]) -> pd.DataFrame:
        """
        Calculate all pairwise correlations with p-values
        """
        if self.df is None or len(columns) < 2:
            return pd.DataFrame()

        results = []

        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                data = self.df[[col1, col2]].dropna()

                pearson_r, pearson_p = pearsonr(data[col1], data[col2])
                spearman_r, spearman_p = spearmanr(data[col1], data[col2])
                kendall_r, kendall_p = kendalltau(data[col1], data[col2])

                results.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Pearson r': float(pearson_r),
                    'Pearson p': float(pearson_p),
                    'Spearman r': float(spearman_r),
                    'Spearman p': float(spearman_p),
                    'Kendall tau': float(kendall_r),
                    'Kendall p': float(kendall_p)
                })

        return pd.DataFrame(results)

    # =========================================================================
    # MULTIPLE TESTING CORRECTION
    # =========================================================================
    
    def multiple_testing_correction(self, p_values: List[float],
                                     method: str = 'bonferroni',
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Apply multiple testing correction to p-values
        
        Args:
            p_values: List of p-values to correct
            method: Correction method
                   - 'bonferroni': Bonferroni correction (conservative)
                   - 'holm': Holm-Bonferroni (step-down)
                   - 'sidak': Šidák correction
                   - 'fdr_bh': Benjamini-Hochberg FDR
                   - 'fdr_by': Benjamini-Yekutieli FDR
            alpha: Significance level
            
        Returns:
            Dictionary with corrected p-values and decisions
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            corrected = np.minimum(p_values * n, 1.0)
            threshold = alpha / n
            
        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_idx = np.argsort(p_values)
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)
            # Ensure monotonicity
            for i in range(1, n):
                corrected[sorted_idx[i]] = max(corrected[sorted_idx[i]], 
                                                corrected[sorted_idx[i-1]])
            threshold = alpha
            
        elif method == 'sidak':
            corrected = 1 - (1 - p_values) ** n
            threshold = 1 - (1 - alpha) ** (1/n)
            
        elif method == 'fdr_bh':
            # Benjamini-Hochberg
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            corrected = np.zeros(n)
            
            for i in range(n):
                corrected[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
            
            # Ensure monotonicity (from largest to smallest)
            for i in range(n - 2, -1, -1):
                corrected[sorted_idx[i]] = min(corrected[sorted_idx[i]], 
                                                corrected[sorted_idx[i+1]])
            corrected = np.minimum(corrected, 1.0)
            threshold = alpha
            
        elif method == 'fdr_by':
            # Benjamini-Yekutieli (more conservative FDR)
            c_m = np.sum(1 / np.arange(1, n + 1))
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            corrected = np.zeros(n)
            
            for i in range(n):
                corrected[sorted_idx[i]] = sorted_p[i] * n * c_m / (i + 1)
            
            for i in range(n - 2, -1, -1):
                corrected[sorted_idx[i]] = min(corrected[sorted_idx[i]], 
                                                corrected[sorted_idx[i+1]])
            corrected = np.minimum(corrected, 1.0)
            threshold = alpha
            
        else:
            return {'error': f'Unknown method: {method}'}
        
        reject = corrected < alpha
        
        return {
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected.tolist(),
            'reject_null': reject.tolist(),
            'n_rejected': int(np.sum(reject)),
            'method': method,
            'alpha': alpha,
            'threshold': float(threshold) if method in ['bonferroni', 'sidak'] else alpha
        }
    
    # =========================================================================
    # VARIANCE INFLATION FACTOR (VIF)
    # =========================================================================
    
    def variance_inflation_factor(self, columns: List[str]) -> Dict[str, Any]:
        """
        Calculate Variance Inflation Factor for multicollinearity detection
        
        VIF > 5 suggests moderate multicollinearity
        VIF > 10 suggests high multicollinearity
        
        Args:
            columns: List of predictor columns
            
        Returns:
            Dictionary with VIF for each column
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna()
        
        if len(data) < len(columns) + 1:
            return {'error': 'Insufficient data for VIF calculation'}
        
        vif_data = {}
        
        for i, col in enumerate(columns):
            # Regress column i on all other columns
            y = data[col].values
            X_cols = [c for c in columns if c != col]
            X = data[X_cols].values
            
            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            try:
                # Calculate R-squared
                coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                y_pred = X_with_const @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # VIF = 1 / (1 - R²)
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                
                vif_data[col] = {
                    'vif': float(vif),
                    'r_squared': float(r_squared),
                    'multicollinearity': 'high' if vif > 10 else 'moderate' if vif > 5 else 'low'
                }
            except:
                vif_data[col] = {'vif': float('nan'), 'error': 'Calculation failed'}
        
        # Overall assessment
        max_vif = max([v['vif'] for v in vif_data.values() if not np.isnan(v['vif'])], default=0)
        
        return {
            'vif_by_column': vif_data,
            'max_vif': float(max_vif),
            'has_multicollinearity': max_vif > 5,
            'severe_multicollinearity': max_vif > 10,
            'recommendation': 'Consider removing highly correlated predictors' if max_vif > 5 else 'No action needed'
        }
    
    # =========================================================================
    # ROBUST STATISTICS
    # =========================================================================
    
    def robust_statistics(self, column: str) -> Dict[str, Any]:
        """
        Calculate robust statistics resistant to outliers
        
        Args:
            column: Column to analyze
            
        Returns:
            Dictionary with robust statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Robust measures of central tendency
        median = float(np.median(data))
        
        # Trimmed mean (10% from each tail)
        trimmed_mean = float(stats.trim_mean(data, 0.1))
        
        # Winsorized mean
        sorted_data = np.sort(data)
        n = len(sorted_data)
        k = int(0.1 * n)
        winsorized = sorted_data.copy()
        if k > 0:
            winsorized[:k] = sorted_data[k]
            winsorized[-k:] = sorted_data[-k-1]
        winsorized_mean = float(np.mean(winsorized))
        
        # Robust measures of dispersion
        # MAD (Median Absolute Deviation)
        mad = float(np.median(np.abs(data - median)))
        # Scaled MAD (consistent with std for normal)
        scaled_mad = mad * 1.4826
        
        # IQR
        q1, q3 = np.percentile(data, [25, 75])
        iqr = float(q3 - q1)
        
        # Qn estimator (robust scale)
        try:
            diffs = []
            for i in range(min(len(data), 500)):  # Limit for performance
                for j in range(i + 1, min(len(data), 500)):
                    diffs.append(abs(data[i] - data[j]))
            qn = 2.2219 * np.percentile(diffs, 25) if diffs else scaled_mad
        except:
            qn = scaled_mad
        
        # Robust skewness (quartile skewness)
        quartile_skewness = ((q3 - median) - (median - q1)) / iqr if iqr > 0 else 0
        
        # Comparison with standard statistics
        mean = float(np.mean(data))
        std = float(np.std(data))
        
        return {
            'n': len(data),
            'robust_center': {
                'median': median,
                'trimmed_mean_10pct': trimmed_mean,
                'winsorized_mean_10pct': winsorized_mean
            },
            'robust_scale': {
                'mad': mad,
                'scaled_mad': float(scaled_mad),
                'iqr': iqr,
                'qn_estimator': float(qn)
            },
            'robust_skewness': {
                'quartile_skewness': float(quartile_skewness),
                'interpretation': 'right-skewed' if quartile_skewness > 0.2 else 'left-skewed' if quartile_skewness < -0.2 else 'symmetric'
            },
            'standard_statistics': {
                'mean': mean,
                'std': std
            },
            'outlier_impact': {
                'mean_median_diff': float(abs(mean - median)),
                'std_mad_ratio': float(std / scaled_mad) if scaled_mad > 0 else float('nan'),
                'likely_outliers': abs(mean - median) > 0.2 * std
            }
        }
    
    def robust_regression(self, x_col: str, y_col: str,
                          method: str = 'huber') -> Dict[str, Any]:
        """
        Robust regression resistant to outliers
        
        Args:
            x_col: Predictor column
            y_col: Response column
            method: 'huber', 'tukey', or 'lad' (least absolute deviation)
            
        Returns:
            Dictionary with regression results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[[x_col, y_col]].dropna()
        x = data[x_col].values
        y = data[y_col].values
        
        # Standard OLS for comparison
        X = np.column_stack([np.ones(len(x)), x])
        ols_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Robust regression (iteratively reweighted least squares)
        if method == 'huber':
            # Huber's M-estimator
            c = 1.345  # tuning constant
        elif method == 'tukey':
            c = 4.685
        elif method == 'lad':
            # LAD is equivalent to L1 regression
            c = float('inf')
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Initial estimate
        coeffs = ols_coeffs.copy()
        
        for iteration in range(50):
            # Residuals
            residuals = y - X @ coeffs
            
            # Scale estimate
            scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
            if scale < 1e-10:
                scale = 1e-10
            
            # Standardized residuals
            u = residuals / scale
            
            # Weights
            if method == 'huber':
                weights = np.where(np.abs(u) <= c, 1, c / np.abs(u))
            elif method == 'tukey':
                weights = np.where(np.abs(u) <= c, (1 - (u/c)**2)**2, 0)
            else:  # LAD
                weights = 1 / (np.abs(residuals) + 1e-10)
            
            # Weighted least squares
            W = np.diag(weights)
            try:
                coeffs_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            except:
                break
            
            # Check convergence
            if np.max(np.abs(coeffs_new - coeffs)) < 1e-6:
                break
            coeffs = coeffs_new
        
        return {
            'robust_intercept': float(coeffs[0]),
            'robust_slope': float(coeffs[1]),
            'ols_intercept': float(ols_coeffs[0]),
            'ols_slope': float(ols_coeffs[1]),
            'method': method,
            'iterations': iteration + 1,
            'slope_difference': float(abs(coeffs[1] - ols_coeffs[1])),
            'interpretation': 'Outliers may be affecting OLS' if abs(coeffs[1] - ols_coeffs[1]) > 0.1 * abs(ols_coeffs[1]) else 'OLS and robust estimates agree'
        }

    def plot_cross_correlation(self, col1: str, col2: str) -> plt.Figure:
        """Plot cross-correlation between two columns"""
        if self.df is None:
            return None

        sig1 = self.df[col1].dropna().values
        sig2 = self.df[col2].dropna().values

        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[:min_len]
        sig2 = sig2[:min_len]

        correlation = correlate(sig1, sig2, mode='full')
        lags = np.arange(-len(sig1)+1, len(sig1))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(lags, correlation, linewidth=2)
        ax.axvline(0, color='r', linestyle='--', label='Zero lag')
        ax.set_title(f'Cross-Correlation: {col1} vs {col2}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def lag_analysis(self, features: List[str], target: str, max_lag: int = 10) -> Dict[str, List[Dict]]:
        """
        Perform lag correlation analysis
        """
        if self.df is None:
            return {}

        results = {}

        for feature in features:
            feature_results = []
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    data = self.df[[feature, target]].dropna()
                    if len(data) > 2:
                        corr, p_val = pearsonr(data[feature], data[target])
                        feature_results.append({
                            'lag': lag,
                            'correlation': float(corr),
                            'p_value': float(p_val)
                        })
                else:
                    if lag < len(self.df):
                        feat_data = self.df[feature][:-lag]
                        targ_data = self.df[target][lag:]
                        common_idx = feat_data.dropna().index.intersection(targ_data.dropna().index)
                        if len(common_idx) > 2:
                            corr, p_val = pearsonr(feat_data.loc[common_idx], targ_data.loc[common_idx])
                            feature_results.append({
                                'lag': lag,
                                'correlation': float(corr),
                                'p_value': float(p_val)
                            })

            results[feature] = feature_results

        return results
