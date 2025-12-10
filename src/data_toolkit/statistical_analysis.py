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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import (
    pearsonr, spearmanr, kendalltau, shapiro, normaltest,
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal,
    f_oneway, levene, fligner, chi2_contingency,
    norm, gamma, expon, lognorm, weibull_min, beta, lognorm as scipy_lognorm
)
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import accelerated functions
try:
    from .rust_accelerated import (
        detect_outliers_iqr as _accel_outliers,
        AccelerationSettings
    )
except ImportError:
    from rust_accelerated import (
        detect_outliers_iqr as _accel_outliers,
        AccelerationSettings
    )


class StatisticalAnalysis:
    """Enhanced statistical analysis methods"""
    
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
    
    def outlier_detection(self, columns: List[str], method: str = 'iqr') -> Dict[str, Dict]:
        """
        Detect outliers using IQR or Z-score method
        """
        if self.df is None:
            return {}
        
        results = {}
        
        if method == 'iqr':
            data_matrix = self.df[columns].values
            counts, percentages = _accel_outliers(data_matrix, multiplier=1.5)
            
            for i, col in enumerate(columns):
                data = self.df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                
                results[col] = {
                    'n_outliers': int(counts[i]),
                    'percentage': float(percentages[i]),
                    'outlier_indices': data[outliers].index.tolist(),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        else:  # zscore
            for col in columns:
                data = self.df[col].dropna()
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > 3
                n_outliers = outliers.sum()
                
                results[col] = {
                    'n_outliers': int(n_outliers),
                    'percentage': float((n_outliers / len(data)) * 100),
                    'outlier_indices': data[outliers].index.tolist()
                }
        
        return results
    
    def plot_boxplots(self, columns: List[str]) -> plt.Figure:
        """Plot box plots for outlier visualization"""
        if self.df is None or not columns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df[columns].boxplot(ax=ax)
        ax.set_title('Box Plots - Outlier Detection')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
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
