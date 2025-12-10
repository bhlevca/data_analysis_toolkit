"""
Bayesian Analysis Module
Contains methods for Bayesian regression, credible intervals,
posterior distributions, and model comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from scipy.stats import norm
from itertools import combinations
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class BayesianAnalysis:
    """Bayesian statistical analysis methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.model = None
        self.posterior_samples = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    def bayesian_regression(self, features: List[str], target: str, 
                           n_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform Bayesian linear regression with posterior inference
        
        Args:
            features: List of feature column names
            target: Target column name
            n_samples: Number of posterior samples
            
        Returns:
            Dictionary with regression results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X.values])
        
        # Priors (weakly informative)
        prior_mean = np.zeros(X_with_intercept.shape[1])
        prior_cov = np.eye(X_with_intercept.shape[1]) * 100
        
        # Posterior calculation (conjugate prior)
        noise_precision = 1.0
        
        posterior_cov = np.linalg.inv(
            np.linalg.inv(prior_cov) + noise_precision * X_with_intercept.T @ X_with_intercept
        )
        posterior_mean = posterior_cov @ (
            np.linalg.inv(prior_cov) @ prior_mean + 
            noise_precision * X_with_intercept.T @ y.values
        )
        
        # Sample from posterior
        self.posterior_samples = np.random.multivariate_normal(
            posterior_mean, posterior_cov, n_samples
        )
        
        # Calculate credible intervals (95%)
        credible_intervals = np.percentile(self.posterior_samples, [2.5, 97.5], axis=0)
        
        results = {
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'credible_intervals_lower': credible_intervals[0],
            'credible_intervals_upper': credible_intervals[1],
            'features': ['Intercept'] + list(features),
            'posterior_samples': self.posterior_samples
        }
        
        return results
    
    def credible_intervals(self, features: List[str], target: str,
                          confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate credible intervals using Bayesian Ridge regression
        
        Returns:
            Dictionary with prediction intervals
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        self.model = BayesianRidge()
        self.model.fit(X, y)
        
        y_pred, y_std = self.model.predict(X, return_std=True)
        
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha/2)
        
        ci_lower = y_pred - z_score * y_std
        ci_upper = y_pred + z_score * y_std
        
        # Calculate coverage
        coverage = np.mean((y.values >= ci_lower) & (y.values <= ci_upper))
        
        return {
            'y_pred': y_pred,
            'y_std': y_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'coverage': coverage,
            'mean_ci_width': np.mean(ci_upper - ci_lower),
            'y_actual': y.values,
            'confidence': confidence
        }
    
    def posterior_distributions(self, features: List[str], target: str) -> Dict[str, Any]:
        """
        Visualize posterior distributions of parameters
        
        Returns:
            Dictionary with posterior statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        model = BayesianRidge(compute_score=True)
        model.fit(X, y)
        
        return {
            'alpha': model.alpha_,
            'lambda': model.lambda_,
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'scores': model.scores_ if hasattr(model, 'scores_') else None
        }
    
    def bayesian_model_comparison(self, features: List[str], target: str) -> List[Dict[str, Any]]:
        """
        Compare models using Bayesian Information Criterion
        
        Returns:
            List of model results sorted by BIC
        """
        if self.df is None:
            return []
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        results = []
        
        # Try different feature subsets
        for i in range(1, len(features) + 1):
            for feature_combo in combinations(range(len(features)), i):
                X_subset = X.iloc[:, list(feature_combo)]
                model = BayesianRidge()
                model.fit(X_subset, y)
                
                # Calculate BIC
                n = len(y)
                k = len(feature_combo) + 1
                y_pred = model.predict(X_subset)
                rss = np.sum((y.values - y_pred)**2)
                bic = n * np.log(rss/n) + k * np.log(n)
                
                feat_names = [features[j] for j in feature_combo]
                results.append({
                    'bic': bic,
                    'features': feat_names,
                    'r2': model.score(X_subset, y),
                    'n_features': len(feat_names)
                })
        
        # Sort by BIC (lower is better)
        results.sort(key=lambda x: x['bic'])
        
        return results
    
    def prior_sensitivity(self, features: List[str], target: str) -> List[Dict[str, Any]]:
        """
        Analyze sensitivity to prior distributions
        
        Returns:
            List of results for different prior configurations
        """
        if self.df is None:
            return []
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        lambdas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        
        results = []
        
        for alpha in alphas:
            for lambda_ in lambdas:
                model = BayesianRidge(alpha_init=alpha, lambda_init=lambda_)
                model.fit(X, y)
                
                results.append({
                    'alpha_init': alpha,
                    'lambda_init': lambda_,
                    'r2': model.score(X, y),
                    'final_alpha': model.alpha_,
                    'final_lambda': model.lambda_,
                    'coefficients': model.coef_.copy()
                })
        
        return results
    
    def plot_posterior_distributions(self, results: Dict[str, Any], max_plots: int = 3) -> plt.Figure:
        """Plot posterior distributions of parameters"""
        samples = results.get('posterior_samples')
        features = results.get('features')
        posterior_mean = results.get('posterior_mean')
        ci_lower = results.get('credible_intervals_lower')
        ci_upper = results.get('credible_intervals_upper')
        
        if samples is None:
            return None
        
        n_plots = min(max_plots, len(features))
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        for i in range(n_plots):
            axes[i].hist(samples[:, i], bins=50, density=True, alpha=0.7)
            axes[i].axvline(posterior_mean[i], color='r', linestyle='--', label='Mean')
            axes[i].axvline(ci_lower[i], color='g', linestyle='--', label='95% CI')
            axes[i].axvline(ci_upper[i], color='g', linestyle='--')
            axes[i].set_title(f'Posterior: {features[i]}')
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_credible_intervals(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot predictions with credible intervals"""
        y_actual = results.get('y_actual')
        y_pred = results.get('y_pred')
        y_std = results.get('y_std')
        confidence = results.get('confidence', 0.95)
        
        if y_actual is None or y_pred is None:
            return None
        
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha/2)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_actual, y_pred, alpha=0.5, label='Predictions')
        ax.errorbar(y_actual, y_pred, yerr=z_score*y_std, fmt='none', alpha=0.2)
        ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
                'r--', label='Perfect prediction')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Predictions with {confidence*100:.0f}% Credible Intervals')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
