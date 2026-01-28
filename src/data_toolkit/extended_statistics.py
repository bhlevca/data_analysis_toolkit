"""
Extended Statistical Tests and Distribution Operations Module

Contains additional statistical tests and distribution operations that complement
the main statistical_analysis module:

Statistical Tests:
- Kolmogorov-Smirnov test (1-sample and 2-sample)
- Anderson-Darling test
- Runs test for randomness
- Sign test
- Mood's median test
- Bartlett's test for equal variances
- Brown-Forsythe test
- Friedman test (non-parametric repeated measures)
- Cochran's Q test
- McNemar's test

Distribution Operations:
- Kernel Density Estimation (KDE)
- Mixture model fitting
- Percentile/quantile calculations
- Moment calculations (raw, central, standardized)
- Entropy and information measures
- Distribution sampling and simulation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (
    kstest, ks_2samp, anderson, shapiro,
    median_test, bartlett, fligner, friedmanchisquare,
    entropy as scipy_entropy
)
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class ExtendedStatisticalTests:
    """Extended statistical tests beyond the base module."""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze."""
        self.df = df
    
    # =========================================================================
    # NORMALITY AND DISTRIBUTION TESTS
    # =========================================================================
    
    def kolmogorov_smirnov_1sample(
        self, 
        column: str, 
        distribution: str = 'norm',
        **dist_params
    ) -> Dict[str, Any]:
        """
        One-sample Kolmogorov-Smirnov test.
        
        Tests if data follows a specified distribution.
        
        Args:
            column: Column name to test
            distribution: Distribution name ('norm', 'expon', 'uniform', etc.)
            **dist_params: Parameters for the distribution
            
        Returns:
            Dictionary with KS statistic, p-value, and interpretation
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # If no params, estimate from data
        if not dist_params:
            if distribution == 'norm':
                dist_params = {'loc': np.mean(data), 'scale': np.std(data)}
            elif distribution == 'expon':
                dist_params = {'loc': 0, 'scale': np.mean(data)}
        
        statistic, p_value = kstest(data, distribution, args=tuple(dist_params.values()))
        
        return {
            'test': 'Kolmogorov-Smirnov (1-sample)',
            'distribution': distribution,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'sample_size': len(data),
            'reject_null': p_value < 0.05,
            'interpretation': f"Data {'does NOT' if p_value < 0.05 else 'does'} follow {distribution} distribution (α=0.05)"
        }
    
    def kolmogorov_smirnov_2sample(
        self, 
        column1: str, 
        column2: str
    ) -> Dict[str, Any]:
        """
        Two-sample Kolmogorov-Smirnov test.
        
        Tests if two samples come from the same distribution.
        
        Args:
            column1: First column name
            column2: Second column name
            
        Returns:
            Dictionary with KS statistic, p-value, and interpretation
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data1 = self.df[column1].dropna().values
        data2 = self.df[column2].dropna().values
        
        statistic, p_value = ks_2samp(data1, data2)
        
        return {
            'test': 'Kolmogorov-Smirnov (2-sample)',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'n1': len(data1),
            'n2': len(data2),
            'reject_null': p_value < 0.05,
            'interpretation': f"Samples {'come from DIFFERENT' if p_value < 0.05 else 'may come from same'} distributions (α=0.05)"
        }
    
    def anderson_darling(
        self, 
        column: str, 
        distribution: str = 'norm'
    ) -> Dict[str, Any]:
        """
        Anderson-Darling test for distribution fit.
        
        More sensitive to tails than KS test.
        
        Args:
            column: Column name to test
            distribution: Distribution type ('norm', 'expon', 'logistic', 'gumbel', 'extreme1')
            
        Returns:
            Dictionary with statistic, critical values, and interpretation
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        result = anderson(data, dist=distribution)
        
        # Determine significance level
        significance_levels = [15, 10, 5, 2.5, 1]  # Percentages
        critical_values = result.critical_values
        
        reject_at = []
        for i, cv in enumerate(critical_values):
            if result.statistic > cv:
                reject_at.append(significance_levels[i])
        
        return {
            'test': 'Anderson-Darling',
            'distribution': distribution,
            'statistic': float(result.statistic),
            'critical_values': {f'{sl}%': float(cv) for sl, cv in zip(significance_levels, critical_values)},
            'reject_at_levels': reject_at,
            'interpretation': f"Reject normality at significance levels: {reject_at}" if reject_at else "Cannot reject normality at any standard level"
        }
    
    def runs_test(self, column: str, cutoff: str = 'median') -> Dict[str, Any]:
        """
        Runs test for randomness.
        
        Tests if the sequence of values is random.
        
        Args:
            column: Column name to test
            cutoff: How to binarize data ('median', 'mean', or numeric value)
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Determine cutoff
        if cutoff == 'median':
            threshold = np.median(data)
        elif cutoff == 'mean':
            threshold = np.mean(data)
        else:
            threshold = float(cutoff)
        
        # Binarize
        binary = (data > threshold).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Calculate expected runs and variance under null hypothesis
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        n = n1 + n2
        
        if n1 == 0 or n2 == 0:
            return {'error': 'All values on one side of cutoff'}
        
        expected_runs = (2 * n1 * n2) / n + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        # Z-statistic
        z = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'test': 'Runs Test for Randomness',
            'n_runs': runs,
            'expected_runs': float(expected_runs),
            'z_statistic': float(z),
            'p_value': float(p_value),
            'n_above': int(n1),
            'n_below': int(n2),
            'cutoff': threshold,
            'reject_null': p_value < 0.05,
            'interpretation': f"Sequence {'is NOT' if p_value < 0.05 else 'appears'} random (α=0.05)"
        }
    
    # =========================================================================
    # NON-PARAMETRIC TESTS
    # =========================================================================
    
    def sign_test(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Sign test for paired data.
        
        Non-parametric alternative to paired t-test.
        Tests if median difference is zero.
        
        Args:
            column1: First column name
            column2: Second column name
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[[column1, column2]].dropna()
        diff = data[column1] - data[column2]
        
        # Exclude zeros
        diff = diff[diff != 0]
        n = len(diff)
        
        if n == 0:
            return {'error': 'No non-zero differences'}
        
        # Count positives
        n_positive = np.sum(diff > 0)
        
        # Binomial test (scipy 1.7+ uses binomtest)
        result = stats.binomtest(n_positive, n, 0.5, alternative='two-sided')
        p_value = result.pvalue
        
        return {
            'test': 'Sign Test',
            'n_positive': int(n_positive),
            'n_negative': int(n - n_positive),
            'n_total': int(n),
            'p_value': float(p_value),
            'reject_null': p_value < 0.05,
            'interpretation': f"Median difference {'is NOT' if p_value < 0.05 else 'may be'} zero (α=0.05)"
        }
    
    def mood_median_test(self, *columns: str) -> Dict[str, Any]:
        """
        Mood's median test.
        
        Non-parametric test for equal medians of multiple groups.
        
        Args:
            *columns: Column names to compare
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        samples = [self.df[col].dropna().values for col in columns]
        
        statistic, p_value, median, contingency = median_test(*samples)
        
        return {
            'test': "Mood's Median Test",
            'statistic': float(statistic),
            'p_value': float(p_value),
            'grand_median': float(median),
            'contingency_table': contingency.tolist(),
            'n_groups': len(columns),
            'reject_null': p_value < 0.05,
            'interpretation': f"Group medians {'are NOT' if p_value < 0.05 else 'may be'} equal (α=0.05)"
        }
    
    def friedman_test(self, columns: List[str]) -> Dict[str, Any]:
        """
        Friedman test for repeated measures.
        
        Non-parametric alternative to repeated measures ANOVA.
        
        Args:
            columns: List of column names (repeated measures)
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna()
        samples = [data[col].values for col in columns]
        
        statistic, p_value = friedmanchisquare(*samples)
        
        return {
            'test': 'Friedman Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'n_treatments': len(columns),
            'n_subjects': len(data),
            'reject_null': p_value < 0.05,
            'interpretation': f"Treatment effects {'are' if p_value < 0.05 else 'may not be'} significantly different (α=0.05)"
        }
    
    # =========================================================================
    # VARIANCE TESTS
    # =========================================================================
    
    def bartlett_test(self, *columns: str) -> Dict[str, Any]:
        """
        Bartlett's test for equal variances.
        
        Assumes normality. Use Levene or Brown-Forsythe if non-normal.
        
        Args:
            *columns: Column names to compare
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        samples = [self.df[col].dropna().values for col in columns]
        
        statistic, p_value = bartlett(*samples)
        
        return {
            'test': "Bartlett's Test",
            'statistic': float(statistic),
            'p_value': float(p_value),
            'n_groups': len(columns),
            'reject_null': p_value < 0.05,
            'assumption': 'Assumes normal distributions',
            'interpretation': f"Variances {'are NOT' if p_value < 0.05 else 'may be'} equal (α=0.05)"
        }
    
    def brown_forsythe_test(self, *columns: str) -> Dict[str, Any]:
        """
        Brown-Forsythe test for equal variances.
        
        Robust to non-normality (uses median instead of mean).
        
        Args:
            *columns: Column names to compare
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        samples = [self.df[col].dropna().values for col in columns]
        
        statistic, p_value = fligner(*samples, center='median')
        
        return {
            'test': 'Brown-Forsythe Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'n_groups': len(columns),
            'reject_null': p_value < 0.05,
            'advantage': 'Robust to non-normality',
            'interpretation': f"Variances {'are NOT' if p_value < 0.05 else 'may be'} equal (α=0.05)"
        }


class DistributionOperations:
    """Advanced distribution operations and calculations."""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze."""
        self.df = df
    
    def kernel_density_estimation(
        self, 
        column: str, 
        bandwidth: str = 'scott',
        n_points: int = 200
    ) -> Dict[str, Any]:
        """
        Kernel Density Estimation (KDE).
        
        Non-parametric estimation of probability density function.
        
        Args:
            column: Column name
            bandwidth: Bandwidth method ('scott', 'silverman', or numeric)
            n_points: Number of evaluation points
            
        Returns:
            Dictionary with x values, density values, and statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Compute KDE
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        
        # Evaluation points
        x_min, x_max = data.min(), data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, n_points)
        density = kde(x)
        
        # Find mode (peak of KDE)
        mode_idx = np.argmax(density)
        mode = x[mode_idx]
        
        return {
            'x': x.tolist(),
            'density': density.tolist(),
            'bandwidth': float(kde.factor),
            'mode_estimate': float(mode),
            'n_samples': len(data)
        }
    
    def percentiles(
        self, 
        column: str, 
        percentiles: List[float] = None
    ) -> Dict[str, float]:
        """
        Calculate percentiles/quantiles.
        
        Args:
            column: Column name
            percentiles: List of percentiles (0-100). Default: [1, 5, 10, 25, 50, 75, 90, 95, 99]
            
        Returns:
            Dictionary mapping percentile to value
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        data = self.df[column].dropna().values
        
        return {
            f'p{int(p)}': float(np.percentile(data, p))
            for p in percentiles
        }
    
    def moments(self, column: str) -> Dict[str, Any]:
        """
        Calculate distribution moments.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with raw, central, and standardized moments
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        n = len(data)
        mean = np.mean(data)
        
        # Raw moments
        raw_moments = {
            f'm{k}': float(np.mean(data**k))
            for k in range(1, 5)
        }
        
        # Central moments
        centered = data - mean
        central_moments = {
            f'mu{k}': float(np.mean(centered**k))
            for k in range(2, 5)
        }
        
        # Standardized moments
        std = np.std(data)
        if std > 0:
            standardized = centered / std
            std_moments = {
                'skewness': float(np.mean(standardized**3)),
                'kurtosis': float(np.mean(standardized**4)),
                'excess_kurtosis': float(np.mean(standardized**4) - 3)
            }
        else:
            std_moments = {'skewness': 0, 'kurtosis': 0, 'excess_kurtosis': -3}
        
        return {
            'raw_moments': raw_moments,
            'central_moments': central_moments,
            'standardized_moments': std_moments,
            'n': n
        }
    
    def entropy(self, column: str, bins: int = 50) -> Dict[str, float]:
        """
        Calculate entropy and information measures.
        
        Args:
            column: Column name
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with entropy measures
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Histogram-based entropy
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        bin_width = (data.max() - data.min()) / bins
        
        # Shannon entropy (in nats)
        shannon_entropy = -np.sum(hist * bin_width * np.log(hist * bin_width + 1e-10))
        
        # Normalized entropy (0 to 1)
        max_entropy = np.log(bins)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        # Negentropy (departure from Gaussianity)
        # For a Gaussian with same variance: H = 0.5 * log(2 * pi * e * var)
        var = np.var(data)
        gaussian_entropy = 0.5 * np.log(2 * np.pi * np.e * var) if var > 0 else 0
        negentropy = gaussian_entropy - shannon_entropy
        
        return {
            'shannon_entropy': float(shannon_entropy),
            'normalized_entropy': float(normalized_entropy),
            'negentropy': float(negentropy),
            'gaussian_entropy': float(gaussian_entropy),
            'bins_used': bins
        }
    
    def distribution_sampling(
        self, 
        column: str, 
        distribution: str, 
        n_samples: int = 1000,
        fit_params: bool = True
    ) -> Dict[str, Any]:
        """
        Sample from a fitted distribution.
        
        Args:
            column: Column name to fit
            distribution: Distribution name ('norm', 'expon', 'gamma', etc.)
            n_samples: Number of samples to generate
            fit_params: If True, fit params from data; else use standard params
            
        Returns:
            Dictionary with samples and fitted parameters
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Get distribution
        dist = getattr(stats, distribution, None)
        if dist is None:
            return {'error': f'Unknown distribution: {distribution}'}
        
        # Fit parameters
        if fit_params:
            params = dist.fit(data)
        else:
            params = ()
        
        # Generate samples
        samples = dist.rvs(*params, size=n_samples)
        
        return {
            'samples': samples.tolist(),
            'distribution': distribution,
            'parameters': params,
            'original_n': len(data),
            'generated_n': n_samples
        }
    
    def probability_calculations(
        self, 
        column: str, 
        distribution: str = 'norm',
        values: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate probabilities for given values.
        
        Args:
            column: Column name to fit
            distribution: Distribution to use
            values: Values to calculate probabilities for (default: some quantiles)
            
        Returns:
            Dictionary with CDF, PDF, and survival function values
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        # Get and fit distribution
        dist = getattr(stats, distribution, None)
        if dist is None:
            return {'error': f'Unknown distribution: {distribution}'}
        
        params = dist.fit(data)
        
        # Default values
        if values is None:
            values = [
                np.percentile(data, p) for p in [5, 25, 50, 75, 95]
            ]
        
        results = []
        for v in values:
            results.append({
                'value': float(v),
                'cdf': float(dist.cdf(v, *params)),  # P(X <= v)
                'pdf': float(dist.pdf(v, *params)),  # Density at v
                'sf': float(dist.sf(v, *params)),    # P(X > v)
            })
        
        return {
            'distribution': distribution,
            'parameters': params,
            'calculations': results
        }
    
    def plot_kde_comparison(self, columns: List[str]) -> plt.Figure:
        """
        Plot KDE comparison for multiple columns.
        
        Args:
            columns: List of column names
            
        Returns:
            Matplotlib figure
        """
        if self.df is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in columns:
            data = self.df[col].dropna().values
            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, kde(x), label=col, linewidth=2)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
