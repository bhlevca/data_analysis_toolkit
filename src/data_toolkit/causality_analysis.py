"""
Causality Analysis Module
Contains methods for causality testing including Granger causality,
lead-lag analysis, and correlation at different lags
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import accelerated functions - handle both package and direct run
try:
    from .rust_accelerated import (
        lead_lag_correlations as _accel_lead_lag,
        AccelerationSettings
    )
except ImportError:
    from rust_accelerated import (
        lead_lag_correlations as _accel_lead_lag,
        AccelerationSettings
    )


class CausalityAnalysis:
    """Causality analysis methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    def granger_causality(self, features: List[str], target: str, 
                         max_lag: int = 10) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Perform Granger causality test
        
        Args:
            features: List of feature column names
            target: Target column name
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with test results for each feature and lag
        """
        if self.df is None:
            return {}
        
        results = {}
        
        for feature in features:
            data = pd.DataFrame({
                'target': self.df[target],
                'feature': self.df[feature]
            }).dropna()
            
            if len(data) < max_lag + 2:
                results[feature] = {'error': 'Insufficient data'}
                continue
            
            try:
                gc_result = grangercausalitytests(
                    data[['target', 'feature']], 
                    maxlag=max_lag, 
                    verbose=False
                )
                
                feature_results = {}
                for lag in range(1, max_lag + 1):
                    feature_results[lag] = {
                        'ssr_ftest_pvalue': gc_result[lag][0]['ssr_ftest'][1],
                        'ssr_chi2test_pvalue': gc_result[lag][0]['ssr_chi2test'][1],
                        'lrtest_pvalue': gc_result[lag][0]['lrtest'][1],
                        'params_ftest_pvalue': gc_result[lag][0]['params_ftest'][1],
                        'is_significant': gc_result[lag][0]['ssr_ftest'][1] < 0.05
                    }
                
                results[feature] = feature_results
                
            except Exception as e:
                results[feature] = {'error': str(e)}
        
        return results
    
    def lead_lag_analysis(self, features: List[str], target: str,
                         max_lag: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Perform lead-lag correlation analysis
        
        Uses Rust acceleration when enabled for significant speedup.
        
        Returns:
            Dictionary with correlation at each lag for each feature
        """
        if self.df is None:
            return {}
        
        results = {}
        
        for feature in features:
            # Get clean data - must align indices first, then drop NaN together
            combined = self.df[[feature, target]].dropna()
            feat_data = combined[feature].values
            targ_data = combined[target].values
            
            # Use accelerated lead-lag (handles Rust/Python switching internally)
            lags, correlations = _accel_lead_lag(feat_data, targ_data, max_lag)
            lags = list(lags)
            correlations = list(correlations)
            
            # Find best lag
            best_idx = np.argmax(np.abs(correlations))
            best_lag = lags[best_idx]
            best_corr = correlations[best_idx]
            
            results[feature] = {
                'lags': lags,
                'correlations': correlations,
                'best_lag': best_lag,
                'best_correlation': best_corr,
                'feature_leads': best_lag < 0,
                'target_leads': best_lag > 0
            }
        
        return results
    
    def correlation_at_lags(self, features: List[str], target: str,
                           max_lag: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate correlation at different lags
        
        Returns:
            Dictionary with correlation data for each feature
        """
        if self.df is None:
            return {}
        
        results = {}
        
        for feature in features:
            lag_results = []
            best_corr = 0
            best_lag = 0
            
            for lag in range(-max_lag, max_lag + 1):
                try:
                    if lag == 0:
                        feat_data = self.df[feature].dropna()
                        targ_data = self.df[target].loc[feat_data.index].dropna()
                        common_idx = feat_data.index.intersection(targ_data.index)
                        corr, pval = pearsonr(feat_data.loc[common_idx], targ_data.loc[common_idx])
                    elif lag > 0:
                        if lag < len(self.df):
                            feat_data = self.df[feature][:-lag]
                            targ_data = self.df[target][lag:]
                            common_idx = feat_data.dropna().index.intersection(targ_data.dropna().index)
                            if len(common_idx) > 2:
                                corr, pval = pearsonr(feat_data.loc[common_idx], targ_data.loc[common_idx])
                            else:
                                continue
                        else:
                            continue
                    else:
                        abs_lag = abs(lag)
                        if abs_lag < len(self.df):
                            feat_data = self.df[feature][abs_lag:]
                            targ_data = self.df[target][:-abs_lag]
                            common_idx = feat_data.dropna().index.intersection(targ_data.dropna().index)
                            if len(common_idx) > 2:
                                corr, pval = pearsonr(feat_data.loc[common_idx], targ_data.loc[common_idx])
                            else:
                                continue
                        else:
                            continue
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                    
                    lag_results.append({
                        'lag': lag,
                        'correlation': corr,
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
                except:
                    continue
            
            results[feature] = {
                'lag_correlations': lag_results,
                'best_lag': best_lag,
                'best_correlation': best_corr
            }
        
        return results
    
    def bidirectional_causality(self, col1: str, col2: str, 
                               max_lag: int = 10) -> Dict[str, Any]:
        """
        Test causality in both directions
        
        Returns:
            Dictionary with bidirectional test results
        """
        if self.df is None:
            return {}
        
        # Test col1 -> col2
        data_12 = pd.DataFrame({
            'target': self.df[col2],
            'feature': self.df[col1]
        }).dropna()
        
        # Test col2 -> col1
        data_21 = pd.DataFrame({
            'target': self.df[col1],
            'feature': self.df[col2]
        }).dropna()
        
        results = {
            f'{col1}_causes_{col2}': {},
            f'{col2}_causes_{col1}': {}
        }
        
        try:
            gc_12 = grangercausalitytests(data_12[['target', 'feature']], maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                results[f'{col1}_causes_{col2}'][lag] = {
                    'p_value': gc_12[lag][0]['ssr_ftest'][1],
                    'significant': gc_12[lag][0]['ssr_ftest'][1] < 0.05
                }
        except Exception as e:
            results[f'{col1}_causes_{col2}'] = {'error': str(e)}
        
        try:
            gc_21 = grangercausalitytests(data_21[['target', 'feature']], maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                results[f'{col2}_causes_{col1}'][lag] = {
                    'p_value': gc_21[lag][0]['ssr_ftest'][1],
                    'significant': gc_21[lag][0]['ssr_ftest'][1] < 0.05
                }
        except Exception as e:
            results[f'{col2}_causes_{col1}'] = {'error': str(e)}
        
        return results
    
    def plot_lead_lag(self, results: Dict[str, Dict[str, Any]], target: str) -> plt.Figure:
        """Plot lead-lag analysis results"""
        if not results:
            return None
        
        n_features = len(results)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 5*n_features))
        
        if n_features == 1:
            axes = [axes]
        
        for i, (feature, data) in enumerate(results.items()):
            lags = data.get('lags', [])
            correlations = data.get('correlations', [])
            best_lag = data.get('best_lag', 0)
            
            axes[i].plot(lags, correlations, marker='o')
            axes[i].axvline(0, color='r', linestyle='--', alpha=0.5)
            axes[i].axvline(best_lag, color='g', linestyle='--', alpha=0.5, label=f'Best lag: {best_lag}')
            axes[i].axhline(0, color='k', linestyle='-', alpha=0.3)
            axes[i].set_title(f'Lead-Lag: {feature} vs {target}')
            axes[i].set_xlabel('Lag (negative = feature leads)')
            axes[i].set_ylabel('Correlation')
            axes[i].grid(True)
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_granger_results(self, results: Dict[str, Any], target: str) -> plt.Figure:
        """Plot Granger causality p-values"""
        if not results:
            return None
        
        # Filter out error entries
        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v}
        
        if not valid_results:
            return None
        
        n_features = len(valid_results)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
        
        if n_features == 1:
            axes = [axes]
        
        for i, (feature, lag_data) in enumerate(valid_results.items()):
            lags = list(lag_data.keys())
            pvalues = [lag_data[lag]['ssr_ftest_pvalue'] for lag in lags]
            
            axes[i].bar(lags, pvalues, alpha=0.7)
            axes[i].axhline(0.05, color='r', linestyle='--', label='α=0.05')
            axes[i].set_title(f'Granger Causality: {feature} → {target}')
            axes[i].set_xlabel('Lag')
            axes[i].set_ylabel('p-value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
