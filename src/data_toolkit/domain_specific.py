"""
Domain-Specific Analysis Module
================================
Specialized analysis methods for specific scientific domains:

Ecological/Environmental Analysis:
- Mann-Kendall trend test
- Sen's slope estimator
- Moran's I spatial autocorrelation
- Shannon diversity index
- Simpson's diversity index
- Species abundance models

Climate Analysis:
- Standardized Precipitation Index (SPI)
- Palmer Drought Severity Index concepts
- Extreme value analysis
- Return period calculation

Biostatistics:
- Bland-Altman analysis
- Method comparison studies
- ICC (Intraclass Correlation)
- Kappa statistics

Version: 1.0

Requirements:
    pip install pymannkendall esda libpysal
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma

warnings.filterwarnings('ignore')

# Optional imports
try:
    import pymannkendall as mk
    MK_AVAILABLE = True
except ImportError:
    MK_AVAILABLE = False

try:
    from esda.moran import Moran
    import libpysal
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False


class DomainSpecificAnalysis:
    """Domain-specific analysis methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    # =========================================================================
    # ECOLOGICAL/ENVIRONMENTAL ANALYSIS
    # =========================================================================
    
    def mann_kendall_test(self, column: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Mann-Kendall trend test
        
        Non-parametric test for monotonic trend in time series.
        Commonly used in hydrology and climate studies.
        
        Args:
            column: Column to analyze
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        if MK_AVAILABLE:
            # Use pymannkendall library
            result = mk.original_test(data, alpha=alpha)
            
            return {
                'trend': result.trend,
                'h': result.h,  # True if significant trend
                'p_value': float(result.p),
                'z_statistic': float(result.z),
                's_statistic': float(result.s),
                'tau': float(result.Tau),  # Kendall's tau
                'sens_slope': float(result.slope),
                'interpretation': f"{'Significant' if result.h else 'No significant'} {result.trend} trend"
            }
        else:
            # Manual implementation
            n = len(data)
            s = 0
            
            for k in range(n - 1):
                for j in range(k + 1, n):
                    s += np.sign(data[j] - data[k])
            
            # Variance
            unique, counts = np.unique(data, return_counts=True)
            g = len(unique)
            
            if n == g:  # No ties
                var_s = (n * (n - 1) * (2 * n + 5)) / 18
            else:
                tp = np.sum(counts * (counts - 1) * (2 * counts + 5))
                var_s = (n * (n - 1) * (2 * n + 5) - tp) / 18
            
            # Z-statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Determine trend
            if p_value < alpha:
                if z > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'no trend'
            
            # Calculate Kendall's tau
            tau = s / (n * (n - 1) / 2)
            
            return {
                'trend': trend,
                'h': p_value < alpha,
                'p_value': float(p_value),
                'z_statistic': float(z),
                's_statistic': float(s),
                'tau': float(tau),  # Kendall's tau
                'interpretation': f"{'Significant' if p_value < alpha else 'No significant'} {trend}"
            }
    
    def sens_slope(self, column: str, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Sen's slope estimator
        
        Robust non-parametric estimator of linear trend.
        
        Args:
            column: Column to analyze
            confidence: Confidence level for interval
            
        Returns:
            Dictionary with slope estimate and confidence interval
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        n = len(data)
        
        # Calculate all pairwise slopes
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                slopes.append((data[j] - data[i]) / (j - i))
        
        slopes = np.array(slopes)
        
        # Median slope
        sens_slope = np.median(slopes)
        
        # Confidence interval
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        n_slopes = len(slopes)
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        
        c_alpha = z * np.sqrt(var_s)
        m1 = int((n_slopes - c_alpha) / 2)
        m2 = int((n_slopes + c_alpha) / 2) + 1
        
        slopes_sorted = np.sort(slopes)
        ci_lower = slopes_sorted[max(0, m1)]
        ci_upper = slopes_sorted[min(n_slopes - 1, m2)]
        
        # Intercept (median of y - slope * x)
        intercepts = data - sens_slope * np.arange(n)
        intercept = np.median(intercepts)
        
        return {
            'sens_slope': float(sens_slope),
            'slope': float(sens_slope),  # Alias for compatibility
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'intercept': float(intercept),
            'n_slopes': len(slopes),
            'confidence': confidence
        }
    
    def seasonal_mann_kendall(self, column: str, period: int = 12,
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        Seasonal Mann-Kendall test
        
        For data with seasonal patterns.
        
        Args:
            column: Column to analyze
            period: Season length (e.g., 12 for monthly data)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if not MK_AVAILABLE:
            return {'error': 'pymannkendall not installed. Install with: pip install pymannkendall'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        result = mk.seasonal_test(data, period=period, alpha=alpha)
        
        return {
            'trend': result.trend,
            'h': result.h,
            'p_value': float(result.p),
            'z_statistic': float(result.z),
            's_statistic': float(result.s),
            'sens_slope': float(result.slope),
            'period': period,
            'interpretation': f"{'Significant' if result.h else 'No significant'} {result.trend} trend"
        }
    
    # =========================================================================
    # BIODIVERSITY INDICES
    # =========================================================================
    
    def shannon_diversity(self, abundance_col: Union[str, List[str]] = None,
                          abundance_data: np.ndarray = None) -> Dict[str, Any]:
        """
        Shannon Diversity Index (H')
        
        Measures species diversity considering richness and evenness.
        
        Args:
            abundance_col: Column name, list of column names, or single column with abundance counts
            abundance_data: Or provide array directly
            
        Returns:
            Dictionary with diversity metrics
        """
        if abundance_data is None:
            if self.df is None:
                return {'error': 'No data loaded'}
            if abundance_col:
                if isinstance(abundance_col, list):
                    # List of species columns - sum each column
                    abundance_data = self.df[abundance_col].sum().values
                else:
                    abundance_data = self.df[abundance_col].dropna().values
            else:
                # Assume each column is a species
                abundance_data = self.df.select_dtypes(include=[np.number]).sum().values
        
        # Convert to numpy array if needed
        abundance_data = np.asarray(abundance_data)
        
        # Remove zeros
        abundance_data = abundance_data[abundance_data > 0]
        
        # Proportions
        total = np.sum(abundance_data)
        p = abundance_data / total
        
        # Shannon index
        H = -np.sum(p * np.log(p))
        
        # Maximum possible diversity (all species equal)
        H_max = np.log(len(abundance_data))
        
        # Pielou's evenness
        J = H / H_max if H_max > 0 else 0
        
        return {
            'shannon_index': float(H),
            'diversity_index': float(H),  # Alias for compatibility
            'max_diversity': float(H_max),
            'pielou_evenness': float(J),
            'evenness': float(J),  # Alias for compatibility
            'species_richness': int(len(abundance_data)),
            'richness': int(len(abundance_data)),  # Alias for compatibility
            'total_individuals': float(total),
            'interpretation': self._interpret_shannon(H)
        }
    
    def _interpret_shannon(self, H: float) -> str:
        """Interpret Shannon diversity value"""
        if H < 1:
            return "Low diversity"
        elif H < 2:
            return "Moderate diversity"
        elif H < 3:
            return "Moderately high diversity"
        else:
            return "High diversity"
    
    def simpson_diversity(self, abundance_data: np.ndarray = None) -> Dict[str, Any]:
        """
        Simpson's Diversity Index
        
        Probability that two randomly selected individuals are different species.
        
        Args:
            abundance_data: Species abundance counts
            
        Returns:
            Dictionary with Simpson's indices
        """
        if abundance_data is None:
            if self.df is None:
                return {'error': 'No data loaded'}
            abundance_data = self.df.select_dtypes(include=[np.number]).sum().values
        
        # Remove zeros
        abundance_data = abundance_data[abundance_data > 0]
        
        n = np.sum(abundance_data)
        
        # Simpson's D
        D = np.sum(abundance_data * (abundance_data - 1)) / (n * (n - 1))
        
        # Simpson's 1-D (diversity)
        diversity = 1 - D
        
        # Simpson's reciprocal (effective number of species)
        reciprocal = 1 / D if D > 0 else float('inf')
        
        return {
            'simpson_D': float(D),
            'simpson_diversity': float(diversity),  # 1 - D
            'simpson_reciprocal': float(reciprocal),  # 1/D
            'species_richness': int(len(abundance_data)),
            'interpretation': f"{'High' if diversity > 0.7 else 'Moderate' if diversity > 0.4 else 'Low'} diversity"
        }
    
    # =========================================================================
    # SPATIAL ANALYSIS
    # =========================================================================
    
    def morans_i(self, value_col: str, x_col: str, y_col: str,
                 method: str = 'knn', k: int = 8) -> Dict[str, Any]:
        """
        Moran's I spatial autocorrelation
        
        Measures spatial clustering of values.
        
        Args:
            value_col: Column with values
            x_col: X coordinate column
            y_col: Y coordinate column
            method: Weight matrix method ('knn', 'distance')
            k: Number of neighbors for KNN
            
        Returns:
            Dictionary with Moran's I results
        """
        if not SPATIAL_AVAILABLE:
            return {'error': 'esda/libpysal not installed. Install with: pip install esda libpysal'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        values = self.df[value_col].values
        coords = np.column_stack([self.df[x_col].values, self.df[y_col].values])
        
        # Create spatial weights
        if method == 'knn':
            w = libpysal.weights.KNN.from_array(coords, k=k)
        else:
            w = libpysal.weights.DistanceBand.from_array(coords, threshold=1)
        
        w.transform = 'r'  # Row standardize
        
        # Calculate Moran's I
        mi = Moran(values, w)
        
        return {
            'morans_i': float(mi.I),
            'expected_i': float(mi.EI),
            'variance': float(mi.VI_norm),
            'z_score': float(mi.z_norm),
            'p_value': float(mi.p_norm),
            'is_significant': mi.p_norm < 0.05,
            'interpretation': self._interpret_morans_i(mi.I, mi.p_norm)
        }
    
    def _interpret_morans_i(self, I: float, p: float) -> str:
        """Interpret Moran's I value"""
        if p >= 0.05:
            return "No significant spatial autocorrelation"
        elif I > 0:
            return "Significant positive spatial autocorrelation (clustering)"
        else:
            return "Significant negative spatial autocorrelation (dispersion)"
    
    # =========================================================================
    # CLIMATE ANALYSIS
    # =========================================================================
    
    def standardized_precipitation_index(self, precip_col: str,
                                          scale: int = 3) -> Dict[str, Any]:
        """
        Standardized Precipitation Index (SPI)
        
        Drought index based on precipitation probability.
        
        Args:
            precip_col: Precipitation column
            scale: Time scale in months (1, 3, 6, 12, etc.)
            
        Returns:
            Dictionary with SPI values
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        precip = self.df[precip_col].values
        
        # Rolling sum for scale
        if scale > 1:
            precip_scaled = pd.Series(precip).rolling(window=scale).sum().values
        else:
            precip_scaled = precip
        
        # Fit gamma distribution (for non-zero values)
        precip_nonzero = precip_scaled[precip_scaled > 0]
        precip_nonzero = precip_nonzero[~np.isnan(precip_nonzero)]
        
        if len(precip_nonzero) < 30:
            return {'error': 'Insufficient data for SPI calculation'}
        
        # Gamma parameters
        alpha, loc, beta = stats.gamma.fit(precip_nonzero, floc=0)
        
        # Calculate SPI
        spi = np.full(len(precip_scaled), np.nan)
        
        for i, p in enumerate(precip_scaled):
            if np.isnan(p):
                continue
            if p == 0:
                # Use probability of zero
                q = len(precip_scaled[precip_scaled == 0]) / len(precip_scaled)
            else:
                q = stats.gamma.cdf(p, alpha, loc, beta)
            
            # Transform to standard normal
            spi[i] = stats.norm.ppf(q) if 0 < q < 1 else np.nan
        
        # Classify drought
        drought_classes = self._classify_spi(spi)
        
        return {
            'spi_values': spi.tolist(),
            'scale': scale,
            'gamma_alpha': float(alpha),
            'gamma_beta': float(beta),
            'drought_classification': drought_classes,
            'current_spi': float(spi[-1]) if not np.isnan(spi[-1]) else None
        }
    
    def _classify_spi(self, spi: np.ndarray) -> Dict[str, int]:
        """Classify SPI values into drought categories"""
        classes = {
            'extremely_wet': int(np.sum(spi >= 2.0)),
            'very_wet': int(np.sum((spi >= 1.5) & (spi < 2.0))),
            'moderately_wet': int(np.sum((spi >= 1.0) & (spi < 1.5))),
            'near_normal': int(np.sum((spi > -1.0) & (spi < 1.0))),
            'moderately_dry': int(np.sum((spi <= -1.0) & (spi > -1.5))),
            'severely_dry': int(np.sum((spi <= -1.5) & (spi > -2.0))),
            'extremely_dry': int(np.sum(spi <= -2.0))
        }
        return classes
    
    def extreme_value_analysis(self, column: str, block_size: int = None,
                                return_periods: List[float] = None) -> Dict[str, Any]:
        """
        Extreme value analysis using GEV distribution
        
        Args:
            column: Column with data
            block_size: Size for block maxima (None = annual if datetime index)
            return_periods: Return periods to calculate
            
        Returns:
            Dictionary with GEV parameters and return levels
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna().values
        
        if return_periods is None:
            return_periods = [2, 5, 10, 25, 50, 100]
        
        # Get block maxima
        if block_size:
            n_blocks = len(data) // block_size
            maxima = np.array([data[i*block_size:(i+1)*block_size].max() 
                              for i in range(n_blocks)])
        else:
            maxima = data  # Use all data
        
        # Fit GEV distribution
        c, loc, scale = stats.genextreme.fit(maxima)
        
        # Calculate return levels
        return_levels = {}
        for T in return_periods:
            p = 1 - 1/T
            return_level = stats.genextreme.ppf(p, c, loc, scale)
            return_levels[T] = float(return_level)
        
        return {
            'gev_shape': float(c),
            'gev_location': float(loc),
            'gev_scale': float(scale),
            'return_levels': return_levels,
            'n_maxima': len(maxima),
            'interpretation': self._interpret_gev_shape(c)
        }
    
    def _interpret_gev_shape(self, c: float) -> str:
        """Interpret GEV shape parameter"""
        if c > 0:
            return "Frechet type (heavy upper tail, unbounded)"
        elif c < 0:
            return "Weibull type (bounded upper tail)"
        else:
            return "Gumbel type (light tail)"
    
    # =========================================================================
    # BIOSTATISTICS
    # =========================================================================
    
    def bland_altman(self, method1_col: str, method2_col: str,
                     confidence: float = 0.95) -> Dict[str, Any]:
        """
        Bland-Altman analysis for method comparison
        
        Args:
            method1_col: First measurement method column
            method2_col: Second measurement method column
            confidence: Confidence level for limits
            
        Returns:
            Dictionary with Bland-Altman statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        m1 = self.df[method1_col].dropna()
        m2 = self.df[method2_col].loc[m1.index].dropna()
        m1 = m1.loc[m2.index]
        
        # Calculate differences and means
        diff = m1 - m2
        mean = (m1 + m2) / 2
        
        # Statistics
        mean_diff = diff.mean()
        std_diff = diff.std()
        
        # Limits of agreement
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        loa_lower = mean_diff - z * std_diff
        loa_upper = mean_diff + z * std_diff
        
        # Confidence intervals for limits
        se_loa = np.sqrt(3 * std_diff**2 / len(diff))
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'loa_lower': float(loa_lower),
            'loa_upper': float(loa_upper),
            'lower_loa': float(loa_lower),  # Alias for compatibility
            'upper_loa': float(loa_upper),  # Alias for compatibility
            'loa_lower_ci': (float(loa_lower - 1.96*se_loa), float(loa_lower + 1.96*se_loa)),
            'loa_upper_ci': (float(loa_upper - 1.96*se_loa), float(loa_upper + 1.96*se_loa)),
            'n_pairs': len(diff),
            'bias': float(mean_diff),
            'interpretation': f"Mean difference (bias): {mean_diff:.3f}, 95% limits: [{loa_lower:.3f}, {loa_upper:.3f}]"
        }
    
    def plot_bland_altman(self, method1_col: str, method2_col: str) -> plt.Figure:
        """Create Bland-Altman plot"""
        if self.df is None:
            return None
        
        result = self.bland_altman(method1_col, method2_col)
        
        m1 = self.df[method1_col].dropna()
        m2 = self.df[method2_col].loc[m1.index].dropna()
        m1 = m1.loc[m2.index]
        
        diff = m1 - m2
        mean = (m1 + m2) / 2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(mean, diff, alpha=0.6)
        
        # Mean difference line
        ax.axhline(y=result['mean_difference'], color='red', linestyle='-',
                  label=f"Mean: {result['mean_difference']:.3f}")
        
        # Limits of agreement
        ax.axhline(y=result['loa_upper'], color='red', linestyle='--',
                  label=f"+1.96 SD: {result['loa_upper']:.3f}")
        ax.axhline(y=result['loa_lower'], color='red', linestyle='--',
                  label=f"-1.96 SD: {result['loa_lower']:.3f}")
        
        ax.set_xlabel(f'Mean of {method1_col} and {method2_col}')
        ax.set_ylabel(f'Difference ({method1_col} - {method2_col})')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def intraclass_correlation(self, columns: List[str],
                                model: str = 'two-way',
                                type: str = 'agreement') -> Dict[str, Any]:
        """
        Intraclass Correlation Coefficient (ICC)
        
        Args:
            columns: Rater/measurement columns
            model: 'one-way', 'two-way'
            type: 'agreement' or 'consistency'
            
        Returns:
            Dictionary with ICC values
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[columns].dropna().values
        n, k = data.shape  # subjects, raters
        
        # Grand mean
        grand_mean = data.mean()
        
        # Subject means
        subject_means = data.mean(axis=1)
        
        # Rater means
        rater_means = data.mean(axis=0)
        
        # Sum of squares
        ss_total = np.sum((data - grand_mean)**2)
        ss_between = k * np.sum((subject_means - grand_mean)**2)
        ss_within = ss_total - ss_between
        
        if model == 'two-way':
            ss_raters = n * np.sum((rater_means - grand_mean)**2)
            ss_error = ss_within - ss_raters
            
            # Mean squares
            ms_between = ss_between / (n - 1)
            ms_raters = ss_raters / (k - 1)
            ms_error = ss_error / ((n - 1) * (k - 1))
            
            if type == 'agreement':
                # ICC(2,1) - two-way random, absolute agreement
                icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + 
                       k * (ms_raters - ms_error) / n)
            else:
                # ICC(3,1) - two-way mixed, consistency
                icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)
        else:
            # One-way
            ms_between = ss_between / (n - 1)
            ms_within = ss_within / (n * (k - 1))
            
            icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        
        # Interpretation
        if icc < 0.5:
            interpretation = "Poor reliability"
        elif icc < 0.75:
            interpretation = "Moderate reliability"
        elif icc < 0.9:
            interpretation = "Good reliability"
        else:
            interpretation = "Excellent reliability"
        
        return {
            'icc': float(icc),
            'model': model,
            'type': type,
            'n_subjects': n,
            'n_raters': k,
            'interpretation': interpretation
        }
    
    def cohens_kappa(self, rater1_col: str, rater2_col: str,
                     weighted: bool = False) -> Dict[str, Any]:
        """
        Cohen's Kappa coefficient for inter-rater agreement
        
        Args:
            rater1_col: First rater column
            rater2_col: Second rater column
            weighted: Use weighted kappa for ordinal data
            
        Returns:
            Dictionary with kappa statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        r1 = self.df[rater1_col].dropna()
        r2 = self.df[rater2_col].loc[r1.index].dropna()
        r1 = r1.loc[r2.index]
        
        # Confusion matrix
        categories = sorted(set(r1) | set(r2))
        n = len(r1)
        k = len(categories)
        
        # Observed agreement matrix
        confusion = pd.crosstab(r1, r2)
        
        # Observed agreement
        po = np.sum(np.diag(confusion)) / n
        
        # Expected agreement
        row_sums = confusion.sum(axis=1) / n
        col_sums = confusion.sum(axis=0) / n
        pe = np.sum(row_sums * col_sums)
        
        # Kappa
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0
        
        # Standard error
        se = np.sqrt((po * (1 - po)) / (n * (1 - pe)**2))
        
        # Confidence interval
        ci_lower = kappa - 1.96 * se
        ci_upper = kappa + 1.96 * se
        
        # Interpretation
        if kappa < 0:
            interpretation = "Less than chance agreement"
        elif kappa < 0.21:
            interpretation = "Slight agreement"
        elif kappa < 0.41:
            interpretation = "Fair agreement"
        elif kappa < 0.61:
            interpretation = "Moderate agreement"
        elif kappa < 0.81:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        
        return {
            'kappa': float(kappa),
            'observed_agreement': float(po),
            'expected_agreement': float(pe),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_observations': n,
            'n_categories': k,
            'interpretation': interpretation
        }
