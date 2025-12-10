"""
Uncertainty Analysis Module
Contains methods for uncertainty quantification including bootstrap CI,
prediction intervals, confidence bands, error propagation, and Monte Carlo analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import t as t_dist, shapiro
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import accelerated functions - handle both package and direct run
try:
    from .rust_accelerated import (
        bootstrap_linear_regression as _accel_bootstrap,
        monte_carlo_predictions as _accel_monte_carlo,
        AccelerationSettings
    )
except ImportError:
    from rust_accelerated import (
        bootstrap_linear_regression as _accel_bootstrap,
        monte_carlo_predictions as _accel_monte_carlo,
        AccelerationSettings
    )


class UncertaintyAnalysis:
    """Uncertainty quantification and interval estimation methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    def bootstrap_ci(self, features: List[str], target: str,
                    n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for regression coefficients
        
        Uses Rust acceleration when enabled for significant speedup.
        
        Args:
            features: List of feature column names
            target: Target column name
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (0-1)
            
        Returns:
            Dictionary with bootstrap results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Use accelerated bootstrap (handles Rust/Python switching internally)
        mean_coefs, ci_lower, ci_upper = _accel_bootstrap(
            X.values, y.values, n_bootstrap, confidence
        )
        
        # Also compute bootstrap distribution for plotting (only needed for detailed analysis)
        # This part stays in Python as it's not the bottleneck
        n_samples = len(X)
        bootstrap_coefs = []
        bootstrap_scores = []
        
        # Run a smaller sample for distribution visualization
        for _ in range(min(n_bootstrap, 200)):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            bootstrap_coefs.append(model.coef_)
            bootstrap_scores.append(model.score(X_boot, y_boot))
        
        bootstrap_coefs = np.array(bootstrap_coefs)
        
        results = {
            'features': features,
            'mean_coefs': mean_coefs,
            'std_coefs': np.std(bootstrap_coefs, axis=0),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_coefs': bootstrap_coefs,
            'bootstrap_scores': np.array(bootstrap_scores),
            'confidence': confidence
        }
        
        return results
    
    def prediction_intervals(self, features: List[str], target: str,
                            confidence: float = 0.95, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Calculate prediction intervals for new observations
        
        Returns:
            Dictionary with prediction interval results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Estimate residual standard error
        residuals = y_train - y_pred_train
        dof = len(X_train) - len(features) - 1
        rse = np.sqrt(np.sum(residuals**2) / dof) if dof > 0 else np.std(residuals)
        
        # Calculate prediction intervals
        t_val = t_dist.ppf((1 + confidence) / 2, dof) if dof > 0 else 1.96
        margin = t_val * rse * np.sqrt(1 + 1/len(X_train))
        
        pi_lower = y_pred_test - margin
        pi_upper = y_pred_test + margin
        
        coverage = np.mean((y_test.values >= pi_lower) & (y_test.values <= pi_upper))
        
        return {
            'y_test': y_test.values,
            'y_pred': y_pred_test,
            'pi_lower': pi_lower,
            'pi_upper': pi_upper,
            'rse': rse,
            'pi_width': 2 * margin,
            'coverage': coverage,
            'confidence': confidence
        }
    
    def confidence_bands(self, feature: str, target: str,
                        confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate confidence bands for regression line (single feature)
        
        Returns:
            Dictionary with confidence band results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[feature].dropna().values.reshape(-1, 1)
        y = self.df[target].loc[self.df[feature].dropna().index].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        n = len(X)
        residuals = y - y_pred
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else np.std(residuals)
        
        t_val = t_dist.ppf((1 + confidence) / 2, n - 2) if n > 2 else 1.96
        
        X_mean = np.mean(X)
        X_ss = np.sum((X - X_mean)**2)
        se_fit = s_err * np.sqrt(1/n + (X - X_mean)**2 / X_ss) if X_ss > 0 else s_err
        
        ci_lower = y_pred - t_val * se_fit.flatten()
        ci_upper = y_pred + t_val * se_fit.flatten()
        
        # Sort for plotting
        sort_idx = np.argsort(X.flatten())
        
        return {
            'X': X.flatten(),
            'y': y,
            'y_pred': y_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sort_idx': sort_idx,
            'feature': feature,
            'target': target,
            'confidence': confidence
        }
    
    def error_propagation(self, features: List[str], target: str) -> Dict[str, Any]:
        """
        Uncertainty propagation through linear model
        
        Returns:
            Dictionary with error propagation results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        
        sigma_y = np.std(residuals)
        sigma_X = X.std(axis=0).values
        
        # Error propagation: σ²(y) = Σ (∂y/∂xi)² σ²(xi)
        propagated_error = np.sqrt(np.sum((model.coef_ * sigma_X)**2))
        
        return {
            'features': features,
            'target': target,
            'input_uncertainties': dict(zip(features, sigma_X)),
            'output_uncertainty': sigma_y,
            'propagated_uncertainty': propagated_error,
            'coefficients': model.coef_
        }
    
    def monte_carlo_analysis(self, features: List[str], target: str,
                            n_simulations: int = 1000, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Monte Carlo uncertainty analysis
        
        Returns:
            Dictionary with Monte Carlo results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        model = LinearRegression()
        model.fit(X, y)
        
        residuals = y - model.predict(X)
        dof = len(X) - len(features) - 1
        rse = np.sqrt(np.sum(residuals**2) / dof) if dof > 0 else np.std(residuals)
        
        predictions = []
        
        for _ in range(n_simulations):
            noisy_coef = model.coef_ + np.random.normal(0, rse/10, len(model.coef_))
            noisy_intercept = model.intercept_ + np.random.normal(0, rse)
            y_sim = X.values @ noisy_coef + noisy_intercept + np.random.normal(0, rse, len(X))
            predictions.append(y_sim)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(predictions, alpha * 100, axis=0)
        ci_upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return {
            'n_simulations': n_simulations,
            'mean_uncertainty': np.mean(std_pred),
            'max_uncertainty': np.max(std_pred),
            'min_uncertainty': np.min(std_pred),
            'mean_ci_width': np.mean(ci_upper - ci_lower),
            'std_distribution': std_pred,
            'confidence': confidence
        }
    
    def residual_analysis(self, features: List[str], target: str) -> Dict[str, Any]:
        """
        Comprehensive residual analysis with diagnostic tests
        
        Returns:
            Dictionary with residual analysis results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y.values - y_pred
        
        std_residuals = residuals / np.std(residuals)
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        
        # Breusch-Pagan test for heteroscedasticity
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        try:
            bp_test = het_breuschpagan(residuals, X_with_const)
            bp_pvalue = bp_test[1]
        except:
            bp_pvalue = np.nan
        
        # Normality test
        if len(residuals) >= 8:
            shapiro_stat, shapiro_p = shapiro(residuals[:5000])
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        return {
            'residuals': residuals,
            'std_residuals': std_residuals,
            'y_pred': y_pred,
            'durbin_watson': dw_stat,
            'bp_pvalue': bp_pvalue,
            'shapiro_stat': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'is_homoscedastic': bp_pvalue > 0.05 if not np.isnan(bp_pvalue) else None,
            'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
            'no_autocorrelation': 1.5 < dw_stat < 2.5
        }
    
    def plot_bootstrap_distributions(self, results: Dict[str, Any], max_plots: int = 3) -> plt.Figure:
        """Plot bootstrap coefficient distributions"""
        bootstrap_coefs = results.get('bootstrap_coefs')
        features = results.get('features')
        ci_lower = results.get('ci_lower')
        ci_upper = results.get('ci_upper')
        confidence = results.get('confidence', 0.95)
        
        if bootstrap_coefs is None:
            return None
        
        n_plots = min(max_plots, len(features))
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        for i in range(n_plots):
            axes[i].hist(bootstrap_coefs[:, i], bins=50, alpha=0.7)
            axes[i].axvline(ci_lower[i], color='r', linestyle='--', 
                          label=f'{confidence*100:.0f}% CI')
            axes[i].axvline(ci_upper[i], color='r', linestyle='--')
            axes[i].set_title(f'{features[i]} Bootstrap Distribution')
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot predictions with prediction intervals"""
        y_test = results.get('y_test')
        y_pred = results.get('y_pred')
        pi_width = results.get('pi_width', 0)
        confidence = results.get('confidence', 0.95)
        
        if y_test is None or y_pred is None:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
        ax.errorbar(y_test, y_pred, yerr=pi_width/2, fmt='none', alpha=0.2)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Predictions with {confidence*100:.0f}% Prediction Intervals')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_confidence_bands(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot regression line with confidence bands"""
        X = results.get('X')
        y = results.get('y')
        y_pred = results.get('y_pred')
        ci_lower = results.get('ci_lower')
        ci_upper = results.get('ci_upper')
        sort_idx = results.get('sort_idx')
        feature = results.get('feature')
        target = results.get('target')
        confidence = results.get('confidence', 0.95)
        
        if X is None:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(X, y, alpha=0.5, label='Data')
        ax.plot(X[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='Regression Line')
        ax.fill_between(X[sort_idx], ci_lower[sort_idx], ci_upper[sort_idx],
                       alpha=0.2, label=f'{confidence*100:.0f}% Confidence Band')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title('Regression with Confidence Bands')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_residual_diagnostics(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot residual diagnostic plots"""
        residuals = results.get('residuals')
        std_residuals = results.get('std_residuals')
        y_pred = results.get('y_pred')
        
        if residuals is None:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # Scale-Location plot
        axes[1, 0].scatter(y_pred, np.sqrt(np.abs(std_residuals)), alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        axes[1, 0].grid(True)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
