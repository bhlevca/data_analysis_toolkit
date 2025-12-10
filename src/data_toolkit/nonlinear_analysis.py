"""
Non-Linear Analysis Module
Contains methods for non-linear relationship detection and modeling including
mutual information, distance correlation, Gaussian processes, and more
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import UnivariateSpline
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import accelerated functions - handle both package and direct run
try:
    from .rust_accelerated import (
        distance_correlation as _accel_distance_correlation,
        mutual_information as _accel_mutual_information,
        AccelerationSettings
    )
except ImportError:
    from rust_accelerated import (
        distance_correlation as _accel_distance_correlation,
        mutual_information as _accel_mutual_information,
        AccelerationSettings
    )


class NonLinearAnalysis:
    """Non-linear analysis and modeling methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    def mutual_information(self, features: List[str], target: str) -> Dict[str, float]:
        """
        Calculate mutual information between features and target
        
        Returns:
            Dictionary of feature names and MI scores
        """
        if self.df is None:
            return {}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        return dict(zip(features, mi_scores))
    
    def distance_correlation(self, features: List[str], target: str) -> Dict[str, float]:
        """
        Calculate distance correlation (detects non-linear relationships)
        
        Uses Rust acceleration when enabled, otherwise falls back to Python.
        
        Returns:
            Dictionary of feature names and distance correlation values
        """
        if self.df is None:
            return {}
        
        results = {}
        for feature in features:
            x = self.df[feature].dropna().values
            y_data = self.df[target].loc[self.df[feature].dropna().index].values
            
            min_len = min(len(x), len(y_data))
            # Use the accelerated function (handles Rust/Python switching internally)
            results[feature] = _accel_distance_correlation(x[:min_len], y_data[:min_len])
        
        return results
    
    def maximal_information_coefficient(self, features: List[str], target: str, 
                                        bins: int = 10) -> Dict[str, float]:
        """
        Calculate approximation of Maximal Information Coefficient
        
        Returns:
            Dictionary of feature names and MIC scores
        """
        if self.df is None:
            return {}
        
        def mic_approx(x, y, max_bins):
            mi_scores = []
            for b in range(2, max_bins):
                try:
                    x_binned = pd.cut(x, bins=b, labels=False)
                    y_binned = pd.cut(y, bins=b, labels=False)
                    
                    mask = ~(pd.isna(x_binned) | pd.isna(y_binned))
                    if mask.sum() < 10:
                        continue
                    
                    mi = mutual_info_regression(
                        x_binned[mask].values.reshape(-1, 1),
                        y_binned[mask].values,
                        random_state=42
                    )[0]
                    mi_scores.append(mi / np.log(b) if np.log(b) > 0 else 0)
                except:
                    continue
            
            return max(mi_scores) if mi_scores else 0.0
        
        results = {}
        for feature in features:
            x = self.df[feature].dropna()
            y = self.df[target].loc[x.index]
            results[feature] = mic_approx(x.values, y.values, bins)
        
        return results
    
    def gaussian_process_regression(self, features: List[str], target: str,
                                   test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit Gaussian Process regression model
        
        Returns:
            Dictionary with GP results including uncertainty
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp.fit(X_train, y_train)
        
        y_pred, sigma = gp.predict(X_test, return_std=True)
        
        return {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mean_uncertainty': sigma.mean(),
            'y_test': y_test.values,
            'y_pred': y_pred,
            'sigma': sigma,
            'kernel': str(gp.kernel_)
        }
    
    def polynomial_regression(self, features: List[str], target: str,
                             max_degree: int = 5, test_size: float = 0.2) -> Dict[int, Dict[str, float]]:
        """
        Fit polynomial regression models of different degrees
        
        Returns:
            Dictionary with results for each polynomial degree
        """
        if self.df is None:
            return {}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        for degree in range(1, max_degree + 1):
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[degree] = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return results
    
    def spline_regression(self, feature: str, target: str,
                         smoothness_levels: List[float] = None) -> Dict[str, Any]:
        """
        Fit spline regression with different smoothness levels
        
        Returns:
            Dictionary with spline results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if smoothness_levels is None:
            smoothness_levels = [0.1, 0.5, 1.0, 5.0]
        
        x = self.df[feature].dropna()
        y = self.df[target].loc[x.index]
        
        sort_idx = np.argsort(x.values)
        x_sorted = x.values[sort_idx]
        y_sorted = y.values[sort_idx]
        
        results = {
            'x_sorted': x_sorted,
            'y_sorted': y_sorted,
            'splines': {}
        }
        
        for s in smoothness_levels:
            try:
                spline = UnivariateSpline(x_sorted, y_sorted, s=s*len(x))
                results['splines'][s] = spline(x_sorted)
            except:
                results['splines'][s] = None
        
        return results
    
    def neural_network_regression(self, features: List[str], target: str,
                                  hidden_layers: Tuple[int, ...] = (100, 50, 25),
                                  test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit neural network regression model
        
        Returns:
            Dictionary with NN results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        nn = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        nn.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = nn.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return {
            'architecture': hidden_layers,
            'iterations': nn.n_iter_,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'y_test': y_test.values,
            'y_pred': y_pred
        }
    
    def svm_regression(self, features: List[str], target: str,
                      kernels: List[str] = None, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Fit SVM regression with different kernels
        
        Returns:
            Dictionary with results for each kernel
        """
        if self.df is None:
            return {}
        
        if kernels is None:
            kernels = ['linear', 'rbf', 'poly']
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        for kernel in kernels:
            svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)
            
            results[kernel] = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return results
    
    def transfer_entropy(self, features: List[str], target: str, k: int = 1) -> Dict[str, Dict[str, float]]:
        """
        Calculate transfer entropy (directed information flow)
        
        Returns:
            Dictionary with transfer entropy in both directions
        """
        if self.df is None:
            return {}
        
        def compute_te(x, y, lag):
            """Simplified transfer entropy calculation"""
            n_bins = 10
            try:
                x_binned = pd.cut(x, bins=n_bins, labels=False)
                y_binned = pd.cut(y, bins=n_bins, labels=False)
                
                mask = ~(pd.isna(x_binned) | pd.isna(y_binned))
                x_binned = x_binned[mask]
                y_binned = y_binned[mask]
                
                if len(x_binned) < lag + 1:
                    return 0.0
                
                # Simplified approximation
                y_present = y_binned[lag:]
                y_past = y_binned[:-lag]
                x_past = x_binned[:-lag]
                
                n = len(y_present)
                te = 0.0
                
                for yp, ypa, xpa in zip(y_present, y_past, x_past):
                    if pd.notna(yp) and pd.notna(ypa) and pd.notna(xpa):
                        te += 1.0 / n
                
                return te
            except:
                return 0.0
        
        results = {}
        for feature in features:
            x = self.df[feature].dropna()
            y = self.df[target].loc[x.index]
            
            te_xy = compute_te(x.values, y.values, k)
            te_yx = compute_te(y.values, x.values, k)
            
            results[feature] = {
                f'{feature}_to_{target}': te_xy,
                f'{target}_to_{feature}': te_yx,
                'direction': feature if te_xy > te_yx else (target if te_yx > te_xy else 'none')
            }
        
        return results
    
    def plot_mutual_information(self, mi_scores: Dict[str, float]) -> plt.Figure:
        """Plot mutual information scores"""
        if not mi_scores:
            return None
        
        features = list(mi_scores.keys())
        scores = list(mi_scores.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(features, scores)
        ax.set_title('Mutual Information Scores')
        ax.set_ylabel('MI Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def plot_gp_predictions(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot Gaussian Process predictions with uncertainty"""
        y_test = results.get('y_test')
        y_pred = results.get('y_pred')
        sigma = results.get('sigma')
        
        if y_test is None or y_pred is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        indices = np.argsort(y_test)
        ax.plot(y_test[indices], label='True', marker='o', markersize=4)
        ax.plot(y_pred[indices], label='Predicted', marker='x', markersize=4)
        
        if sigma is not None:
            ax.fill_between(
                range(len(y_test)),
                (y_pred - 1.96*sigma)[indices],
                (y_pred + 1.96*sigma)[indices],
                alpha=0.3,
                label='95% Confidence'
            )
        
        ax.legend()
        ax.set_title('Gaussian Process Regression with Uncertainty')
        ax.set_xlabel('Sample (sorted)')
        ax.set_ylabel('Value')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_polynomial_comparison(self, results: Dict[int, Dict[str, float]]) -> plt.Figure:
        """Plot polynomial regression comparison"""
        if not results:
            return None
        
        degrees = list(results.keys())
        r2_scores = [results[d]['r2'] for d in degrees]
        rmse_scores = [results[d]['rmse'] for d in degrees]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(degrees, r2_scores, 'o-')
        axes[0].set_xlabel('Polynomial Degree')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² vs Polynomial Degree')
        axes[0].grid(True)
        
        axes[1].plot(degrees, rmse_scores, 'o-', color='orange')
        axes[1].set_xlabel('Polynomial Degree')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE vs Polynomial Degree')
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_spline_regression(self, results: Dict[str, Any], feature: str, target: str) -> plt.Figure:
        """Plot spline regression with different smoothness"""
        x_sorted = results.get('x_sorted')
        y_sorted = results.get('y_sorted')
        splines = results.get('splines', {})
        
        if x_sorted is None:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x_sorted, y_sorted, alpha=0.5, label='Data')
        
        for s, y_spline in splines.items():
            if y_spline is not None:
                ax.plot(x_sorted, y_spline, label=f'Smoothness={s}', linewidth=2)
        
        ax.legend()
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title('Spline Regression with Different Smoothness')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
