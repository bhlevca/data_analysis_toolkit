"""
Data Quality Module
====================
Comprehensive data quality assessment and handling:
- Missing data analysis (MCAR, MAR, MNAR patterns)
- Multiple imputation methods
- Data validation and cleaning
- Outlier detection and handling
- Data transformation (Box-Cox, Yeo-Johnson, etc.)
- Data quality reporting

Version: 1.0
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
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False


class DataQuality:
    """Comprehensive data quality assessment and handling"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.quality_report = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    # =========================================================================
    # MISSING DATA ANALYSIS
    # =========================================================================
    
    def missing_data_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive missing data summary
        
        Returns:
            Dictionary with missing data statistics for each column
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        summary = {}
        n_rows = len(self.df)
        
        for col in self.df.columns:
            n_missing = self.df[col].isna().sum()
            summary[col] = {
                'n_missing': int(n_missing),
                'pct_missing': float(n_missing / n_rows * 100),
                'n_present': int(n_rows - n_missing),
                'dtype': str(self.df[col].dtype)
            }
        
        # Overall statistics
        total_cells = n_rows * len(self.df.columns)
        total_missing = self.df.isna().sum().sum()
        
        return {
            'by_column': summary,
            'total_missing': int(total_missing),
            'total_cells': int(total_cells),
            'pct_missing_overall': float(total_missing / total_cells * 100),
            'complete_rows': int(self.df.dropna().shape[0]),
            'rows_with_missing': int(n_rows - self.df.dropna().shape[0])
        }
    
    def missing_pattern_analysis(self) -> Dict[str, Any]:
        """
        Analyze missing data patterns
        
        Returns:
            Dictionary with pattern analysis results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Create missingness indicator matrix
        missing_matrix = self.df.isna().astype(int)
        
        # Pattern frequencies
        patterns = missing_matrix.apply(lambda x: ''.join(x.astype(str)), axis=1)
        pattern_counts = patterns.value_counts()
        
        # Convert to readable format
        pattern_dict = {}
        for pattern, count in pattern_counts.head(20).items():
            cols_missing = [col for col, val in zip(self.df.columns, pattern) if val == '1']
            pattern_dict[pattern] = {
                'count': int(count),
                'pct': float(count / len(self.df) * 100),
                'columns_missing': cols_missing
            }
        
        # Correlation between missingness
        if missing_matrix.sum().sum() > 0:
            # Only calculate for columns with missing values
            cols_with_missing = missing_matrix.columns[missing_matrix.sum() > 0]
            if len(cols_with_missing) > 1:
                miss_corr = missing_matrix[cols_with_missing].corr()
            else:
                miss_corr = pd.DataFrame()
        else:
            miss_corr = pd.DataFrame()
        
        return {
            'n_patterns': len(pattern_counts),
            'patterns': pattern_dict,
            'complete_pattern_count': int(pattern_counts.get('0' * len(self.df.columns), 0)),
            'missingness_correlation': miss_corr.to_dict() if not miss_corr.empty else {}
        }
    
    def little_mcar_test(self, columns: List[str] = None) -> Dict[str, Any]:
        """
        Little's MCAR test
        
        Tests whether data is Missing Completely At Random.
        
        Args:
            columns: Columns to include (numeric only)
            
        Returns:
            Dictionary with test results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_test = self.df[columns].copy()
        
        # Only include rows with at least one missing and one non-missing
        has_missing = df_test.isna().any(axis=1)
        has_complete = df_test.notna().any(axis=1)
        df_test = df_test[has_missing & has_complete]
        
        if len(df_test) == 0:
            return {'error': 'No rows with partial missing data for MCAR test'}
        
        # Create pattern groups
        patterns = df_test.isna().apply(lambda x: tuple(x), axis=1)
        unique_patterns = patterns.unique()
        
        if len(unique_patterns) < 2:
            return {'error': 'Need at least 2 missing patterns for MCAR test'}
        
        # Calculate test statistic (simplified approach)
        # Full Little's test requires EM algorithm
        overall_means = df_test.mean()
        overall_cov = df_test.cov()
        
        chi2_stat = 0
        df = 0
        
        for pattern in unique_patterns:
            mask = patterns == pattern
            pattern_data = df_test[mask]
            observed_cols = [col for col, is_missing in zip(columns, pattern) if not is_missing]
            
            if len(observed_cols) < 1:
                continue
            
            n_pattern = len(pattern_data)
            pattern_means = pattern_data[observed_cols].mean()
            
            # Contribution to chi-square
            for col in observed_cols:
                diff = pattern_means[col] - overall_means[col]
                if overall_cov.loc[col, col] > 0:
                    chi2_stat += n_pattern * (diff ** 2) / overall_cov.loc[col, col]
                    df += 1
        
        if df > 0:
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        else:
            p_value = 1.0
        
        interpretation = "Data appears MCAR" if p_value > 0.05 else "Data is likely NOT MCAR (may be MAR or MNAR)"
        
        return {
            'test_statistic': float(chi2_stat),
            'degrees_of_freedom': int(df),
            'p_value': float(p_value),
            'is_mcar': p_value > 0.05,
            'interpretation': interpretation,
            'n_patterns': len(unique_patterns)
        }
    
    def plot_missing_matrix(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create missing data visualization matrix
        """
        if self.df is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        missing = self.df.isna()
        
        # Create heatmap
        cmap = plt.cm.colors.ListedColormap(['#4287f5', '#FFFFFF'])
        ax.imshow(missing.values.T, aspect='auto', cmap=cmap, interpolation='nearest')
        
        ax.set_yticks(range(len(self.df.columns)))
        ax.set_yticklabels(self.df.columns)
        ax.set_xlabel('Row Index')
        ax.set_title('Missing Data Matrix (white = missing)')
        
        # Add colorbar
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#4287f5', label='Present'),
                          Patch(facecolor='#FFFFFF', edgecolor='black', label='Missing')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_missing_correlations(self) -> plt.Figure:
        """Plot correlation heatmap of missingness indicators"""
        if self.df is None:
            return None
        
        missing_matrix = self.df.isna()
        cols_with_missing = missing_matrix.columns[missing_matrix.sum() > 0]
        
        if len(cols_with_missing) < 2:
            return None
        
        miss_corr = missing_matrix[cols_with_missing].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(miss_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(cols_with_missing)))
        ax.set_yticks(range(len(cols_with_missing)))
        ax.set_xticklabels(cols_with_missing, rotation=45, ha='right')
        ax.set_yticklabels(cols_with_missing)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Missingness Correlation Matrix')
        
        # Add correlation values
        for i in range(len(cols_with_missing)):
            for j in range(len(cols_with_missing)):
                ax.text(j, i, f'{miss_corr.iloc[i, j]:.2f}',
                       ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # DATA IMPUTATION
    # =========================================================================
    
    def impute_missing(self, method: str = 'mean', columns: List[str] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Impute missing values
        
        Args:
            method: Imputation method
                   - 'mean', 'median', 'mode': Simple imputation
                   - 'knn': K-Nearest Neighbors imputation
                   - 'mice': Multiple Imputation by Chained Equations
                   - 'interpolate': Time-series interpolation
                   - 'ffill', 'bfill': Forward/backward fill
            columns: Columns to impute (None = all numeric)
            **kwargs: Additional arguments for imputers
            
        Returns:
            DataFrame with imputed values
        """
        if self.df is None:
            return None
        
        df_imputed = self.df.copy()
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method in ['mean', 'median']:
            imputer = SimpleImputer(strategy=method)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
            
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
            
        elif method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
            
        elif method == 'mice':
            max_iter = kwargs.get('max_iter', 10)
            imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
            
        elif method == 'interpolate':
            interp_method = kwargs.get('interp_method', 'linear')
            df_imputed[columns] = df_imputed[columns].interpolate(method=interp_method)
            
        elif method == 'ffill':
            df_imputed[columns] = df_imputed[columns].fillna(method='ffill')
            
        elif method == 'bfill':
            df_imputed[columns] = df_imputed[columns].fillna(method='bfill')
        
        return df_imputed
    
    def multiple_imputation(self, columns: List[str] = None, n_imputations: int = 5,
                            max_iter: int = 10) -> List[pd.DataFrame]:
        """
        Generate multiple imputed datasets
        
        Args:
            columns: Columns to impute
            n_imputations: Number of imputed datasets
            max_iter: Maximum iterations for MICE
            
        Returns:
            List of imputed DataFrames
        """
        if not SKLEARN_AVAILABLE:
            return []
        
        if self.df is None:
            return []
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        imputed_datasets = []
        
        for i in range(n_imputations):
            imputer = IterativeImputer(max_iter=max_iter, random_state=42 + i)
            df_imputed = self.df.copy()
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
            imputed_datasets.append(df_imputed)
        
        return imputed_datasets
    
    def pool_imputed_results(self, estimates: List[float], variances: List[float]) -> Dict[str, float]:
        """
        Pool results from multiple imputation using Rubin's rules
        
        Args:
            estimates: List of estimates from each imputed dataset
            variances: List of variances from each imputed dataset
            
        Returns:
            Dictionary with pooled estimate, variance, and confidence interval
        """
        m = len(estimates)
        
        # Pooled estimate (mean of estimates)
        pooled_estimate = np.mean(estimates)
        
        # Within-imputation variance
        within_var = np.mean(variances)
        
        # Between-imputation variance
        between_var = np.var(estimates, ddof=1)
        
        # Total variance
        total_var = within_var + (1 + 1/m) * between_var
        
        # Degrees of freedom (Barnard-Rubin adjustment)
        if between_var > 0:
            lambda_ratio = (1 + 1/m) * between_var / total_var
            df_old = (m - 1) / (lambda_ratio ** 2)
        else:
            df_old = float('inf')
        
        # Confidence interval
        se = np.sqrt(total_var)
        ci_lower = pooled_estimate - 1.96 * se
        ci_upper = pooled_estimate + 1.96 * se
        
        return {
            'pooled_estimate': float(pooled_estimate),
            'pooled_variance': float(total_var),
            'pooled_se': float(se),
            'within_variance': float(within_var),
            'between_variance': float(between_var),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_imputations': m
        }
    
    # =========================================================================
    # DATA TRANSFORMATION
    # =========================================================================
    
    def detect_distribution(self, column: str) -> Dict[str, Any]:
        """
        Detect the likely distribution of a column
        
        Returns:
            Dictionary with distribution fit results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        if len(data) < 20:
            return {'error': 'Insufficient data for distribution detection'}
        
        # Test common distributions
        distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min']
        results = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # K-S test
                ks_stat, ks_p = stats.kstest(data, dist_name, params)
                
                # AIC approximation
                log_likelihood = np.sum(dist.logpdf(data, *params))
                n_params = len(params)
                aic = 2 * n_params - 2 * log_likelihood
                
                results[dist_name] = {
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'aic': float(aic),
                    'parameters': params
                }
            except:
                continue
        
        # Find best fit
        if results:
            best_fit = min(results.keys(), key=lambda x: results[x]['aic'])
            results['best_fit'] = best_fit
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for Shapiro
        dagostino_stat, dagostino_p = stats.normaltest(data)
        
        results['normality_tests'] = {
            'shapiro_wilk': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
            'dagostino_pearson': {'statistic': float(dagostino_stat), 'p_value': float(dagostino_p)},
            'is_normal': shapiro_p > 0.05 and dagostino_p > 0.05
        }
        
        return results
    
    def transform_data(self, column: str, method: str = 'box-cox') -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Transform data to approximate normality
        
        Args:
            column: Column to transform
            method: Transformation method
                   - 'box-cox': Box-Cox transformation (requires positive values)
                   - 'yeo-johnson': Yeo-Johnson transformation (handles negative values)
                   - 'log': Log transformation
                   - 'sqrt': Square root transformation
                   - 'quantile': Quantile transformation to normal
                   
        Returns:
            Tuple of (transformed series, transformation parameters)
        """
        if self.df is None:
            return None, {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        if method == 'box-cox':
            if (data <= 0).any():
                return None, {'error': 'Box-Cox requires positive values'}
            transformed, lambda_param = stats.boxcox(data)
            params = {'lambda': float(lambda_param)}
            
        elif method == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
            transformed = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
            params = {'lambda': float(pt.lambdas_[0])}
            
        elif method == 'log':
            if (data <= 0).any():
                transformed = np.log1p(data - data.min() + 1)
                params = {'shift': float(data.min() - 1)}
            else:
                transformed = np.log(data)
                params = {}
                
        elif method == 'sqrt':
            if (data < 0).any():
                return None, {'error': 'Sqrt requires non-negative values'}
            transformed = np.sqrt(data)
            params = {}
            
        elif method == 'quantile':
            qt = QuantileTransformer(output_distribution='normal')
            transformed = qt.fit_transform(data.values.reshape(-1, 1)).flatten()
            params = {'quantiles': qt.quantiles_.tolist()}
        
        else:
            return None, {'error': f'Unknown method: {method}'}
        
        # Test normality of transformed data
        shapiro_stat, shapiro_p = stats.shapiro(transformed[:5000])
        params['post_transform_normality'] = {
            'shapiro_p': float(shapiro_p),
            'is_normal': shapiro_p > 0.05
        }
        
        return pd.Series(transformed, index=data.index), params
    
    # =========================================================================
    # OUTLIER DETECTION
    # =========================================================================
    
    def detect_outliers(self, column: str, method: str = 'iqr',
                        threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in a column
        
        Args:
            column: Column to analyze
            method: Detection method
                   - 'iqr': Interquartile range method
                   - 'zscore': Z-score method
                   - 'mad': Median Absolute Deviation
                   - 'isolation_forest': Isolation Forest
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier indices and statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        data = self.df[column].dropna()
        
        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (data < lower) | (data > upper)
            bounds = {'lower': float(lower), 'upper': float(upper)}
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold
            bounds = {'threshold': threshold}
            
        elif method == 'mad':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z = 0.6745 * (data - median) / mad if mad > 0 else np.zeros(len(data))
            outliers = np.abs(modified_z) > threshold
            bounds = {'median': float(median), 'mad': float(mad)}
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination='auto', random_state=42)
            predictions = clf.fit_predict(data.values.reshape(-1, 1))
            outliers = pd.Series(predictions == -1, index=data.index)
            bounds = {}
            
        else:
            return {'error': f'Unknown method: {method}'}
        
        outlier_indices = data.index[outliers].tolist()
        
        return {
            'n_outliers': int(outliers.sum()),
            'pct_outliers': float(outliers.sum() / len(data) * 100),
            'outlier_indices': outlier_indices,
            'outlier_values': data[outliers].tolist(),
            'bounds': bounds,
            'method': method
        }
    
    def handle_outliers(self, column: str, method: str = 'winsorize',
                        outlier_method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Handle outliers in data
        
        Args:
            column: Column to process
            method: Handling method
                   - 'winsorize': Cap at bounds
                   - 'remove': Remove outliers (returns NaN)
                   - 'median': Replace with median
            outlier_method: Method to detect outliers
            threshold: Detection threshold
            
        Returns:
            Processed Series
        """
        if self.df is None:
            return None
        
        outliers = self.detect_outliers(column, outlier_method, threshold)
        data = self.df[column].copy()
        
        if method == 'winsorize':
            if 'lower' in outliers['bounds']:
                lower = outliers['bounds']['lower']
                upper = outliers['bounds']['upper']
                data = data.clip(lower=lower, upper=upper)
            else:
                # For z-score, use percentiles
                lower = data.quantile(0.01)
                upper = data.quantile(0.99)
                data = data.clip(lower=lower, upper=upper)
                
        elif method == 'remove':
            data.loc[outliers['outlier_indices']] = np.nan
            
        elif method == 'median':
            median_val = data.median()
            data.loc[outliers['outlier_indices']] = median_val
        
        return data
    
    # =========================================================================
    # DATA QUALITY REPORT
    # =========================================================================
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Returns:
            Dictionary with complete quality assessment
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        report = {
            'overview': {
                'n_rows': len(self.df),
                'n_columns': len(self.df.columns),
                'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024**2)
            },
            'column_types': {
                'numeric': self.df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
                'boolean': self.df.select_dtypes(include=['bool']).columns.tolist()
            },
            'missing_data': self.missing_data_summary(),
            'duplicates': {
                'n_duplicate_rows': int(self.df.duplicated().sum()),
                'pct_duplicate': float(self.df.duplicated().sum() / len(self.df) * 100)
            }
        }
        
        # Column-specific quality metrics
        column_quality = {}
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'n_unique': int(self.df[col].nunique()),
                'n_missing': int(self.df[col].isna().sum())
            }
            
            if self.df[col].dtype in [np.float64, np.int64, float, int]:
                col_info['statistics'] = {
                    'mean': float(self.df[col].mean()) if not self.df[col].isna().all() else None,
                    'std': float(self.df[col].std()) if not self.df[col].isna().all() else None,
                    'min': float(self.df[col].min()) if not self.df[col].isna().all() else None,
                    'max': float(self.df[col].max()) if not self.df[col].isna().all() else None,
                    'skewness': float(self.df[col].skew()) if not self.df[col].isna().all() else None,
                    'kurtosis': float(self.df[col].kurtosis()) if not self.df[col].isna().all() else None
                }
                
                # Quick outlier check
                outliers = self.detect_outliers(col, 'iqr', 1.5)
                if 'error' not in outliers:
                    col_info['n_outliers'] = outliers['n_outliers']
            
            column_quality[col] = col_info
        
        report['columns'] = column_quality
        
        # Quality score (0-100)
        missing_penalty = report['missing_data']['pct_missing_overall']
        duplicate_penalty = report['duplicates']['pct_duplicate']
        
        quality_score = max(0, 100 - missing_penalty - duplicate_penalty)
        report['quality_score'] = round(quality_score, 1)
        
        # Recommendations
        recommendations = []
        if missing_penalty > 5:
            recommendations.append("Consider imputation for missing values (>5% missing)")
        if duplicate_penalty > 1:
            recommendations.append("Review duplicate rows (>1% duplicates)")
        
        # Check for columns with low variance
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].std() == 0:
                recommendations.append(f"Column '{col}' has zero variance")
        
        report['recommendations'] = recommendations
        
        self.quality_report = report
        return report
    
    def plot_data_quality_dashboard(self) -> plt.Figure:
        """Create data quality visualization dashboard"""
        if self.df is None:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Missing data by column
        ax1 = axes[0, 0]
        missing_pct = self.df.isna().mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        if len(missing_pct) > 0:
            missing_pct.plot(kind='barh', ax=ax1, color='coral')
            ax1.set_xlabel('% Missing')
            ax1.set_title('Missing Data by Column')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', fontsize=14)
            ax1.set_title('Missing Data by Column')
        
        # 2. Data type distribution
        ax2 = axes[0, 1]
        type_counts = {
            'Numeric': len(self.df.select_dtypes(include=[np.number]).columns),
            'Text': len(self.df.select_dtypes(include=['object']).columns),
            'DateTime': len(self.df.select_dtypes(include=['datetime64']).columns),
            'Boolean': len(self.df.select_dtypes(include=['bool']).columns)
        }
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        ax2.pie(list(type_counts.values()), labels=list(type_counts.keys()), 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Column Types Distribution')
        
        # 3. Numeric distributions
        ax3 = axes[1, 0]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) > 0:
            self.df[numeric_cols].boxplot(ax=ax3)
            ax3.set_title('Numeric Column Distributions (first 5)')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center')
        
        # 4. Quality score gauge
        ax4 = axes[1, 1]
        if self.quality_report is None:
            self.generate_quality_report()
        
        score = self.quality_report.get('quality_score', 0)
        
        # Create gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax4.fill_between(x, 0, y, alpha=0.3, color='lightgray')
        
        # Color sections
        for i, (start, end, color) in enumerate([
            (0, 60, 'red'), (60, 80, 'orange'), (80, 100, 'green')
        ]):
            mask = (np.linspace(0, 100, 100) >= start) & (np.linspace(0, 100, 100) <= end)
            ax4.fill_between(x[mask], 0, y[mask], alpha=0.5, color=color)
        
        # Needle
        score_angle = np.pi - (score / 100) * np.pi
        ax4.arrow(0, 0, 0.8 * np.cos(score_angle), 0.8 * np.sin(score_angle),
                 head_width=0.05, head_length=0.05, fc='black', ec='black')
        
        ax4.text(0, -0.2, f'Quality Score: {score:.1f}%', ha='center', fontsize=14, weight='bold')
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-0.5, 1.2)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Data Quality Score')
        
        plt.tight_layout()
        return fig
