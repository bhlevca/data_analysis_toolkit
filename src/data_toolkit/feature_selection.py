"""
Feature Selection Module
=========================
Advanced feature selection methods for machine learning:
- Recursive Feature Elimination (RFE)
- Boruta algorithm
- SHAP-based feature selection
- Mutual Information selection
- Statistical feature selection (ANOVA, chi-square)
- Permutation importance selection
- L1-based selection (Lasso)
- Sequential feature selection

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

# sklearn imports
try:
    from sklearn.feature_selection import (
        RFE, RFECV, SelectKBest, SelectFromModel,
        f_classif, f_regression, mutual_info_classif, mutual_info_regression,
        chi2, SequentialFeatureSelector
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class FeatureSelection:
    """Advanced feature selection methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.selected_features = None
        self.feature_importances = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    def _prepare_data(self, feature_cols: List[str], target_col: str):
        """Prepare X and y arrays"""
        if self.df is None:
            return None, None, "No data loaded"
        
        df_clean = self.df[feature_cols + [target_col]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        return X, y, None
    
    # =========================================================================
    # RECURSIVE FEATURE ELIMINATION (RFE)
    # =========================================================================
    
    def recursive_feature_elimination(self, feature_cols: List[str], target_col: str,
                                       n_features: int = None, task: str = 'auto',
                                       step: int = 1, cv: int = None) -> Dict[str, Any]:
        """
        Recursive Feature Elimination
        
        Args:
            feature_cols: List of feature columns
            target_col: Target column
            n_features: Number of features to select (None = use CV)
            task: 'classification', 'regression', or 'auto'
            step: Number of features to remove at each iteration
            cv: Cross-validation folds (None = no CV, uses n_features)
            
        Returns:
            Dictionary with selected features and rankings
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Determine task type
        if task == 'auto':
            unique_vals = len(np.unique(y))
            task = 'classification' if unique_vals < 20 else 'regression'
        
        # Select estimator
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if cv is not None:
            # Use cross-validation
            selector = RFECV(estimator, step=step, cv=cv, scoring='accuracy' if task == 'classification' else 'r2')
            selector.fit(X_scaled, y)
            optimal_n = selector.n_features_
            cv_scores = selector.cv_results_['mean_test_score'].tolist()
        else:
            if n_features is None:
                n_features = max(1, len(feature_cols) // 2)
            selector = RFE(estimator, n_features_to_select=n_features, step=step)
            selector.fit(X_scaled, y)
            optimal_n = n_features
            cv_scores = None
        
        # Get results
        selected_mask = selector.support_
        rankings = selector.ranking_
        
        selected_features = [f for f, selected in zip(feature_cols, selected_mask) if selected]
        feature_rankings = {f: int(r) for f, r in zip(feature_cols, rankings)}
        
        self.selected_features = selected_features
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'feature_rankings': feature_rankings,
            'optimal_n_features': optimal_n,
            'cv_scores': cv_scores,
            'task': task
        }
    
    # =========================================================================
    # BORUTA ALGORITHM
    # =========================================================================
    
    def boruta_selection(self, feature_cols: List[str], target_col: str,
                         task: str = 'auto', n_iterations: int = 100,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Boruta feature selection algorithm
        
        Compares feature importance with shadow (randomized) features.
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            task: 'classification' or 'regression'
            n_iterations: Number of iterations
            alpha: Significance level
            
        Returns:
            Dictionary with confirmed, tentative, and rejected features
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Determine task type
        if task == 'auto':
            unique_vals = len(np.unique(y))
            task = 'classification' if unique_vals < 20 else 'regression'
        
        n_features = len(feature_cols)
        
        # Track hits (times feature beats max shadow)
        hits = np.zeros(n_features)
        
        for iteration in range(n_iterations):
            # Create shadow features
            X_shadow = np.apply_along_axis(np.random.permutation, 0, X)
            X_combined = np.hstack([X, X_shadow])
            
            # Train model
            if task == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=iteration, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=iteration, n_jobs=-1)
            
            model.fit(X_combined, y)
            
            # Get importances
            importances = model.feature_importances_
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]
            
            # Max shadow importance threshold
            max_shadow = np.max(shadow_importances)
            
            # Count hits
            hits += (real_importances > max_shadow).astype(int)
        
        # Statistical test for each feature
        confirmed = []
        tentative = []
        rejected = []
        
        for i, feature in enumerate(feature_cols):
            # Binomial test
            p_value = 1 - stats.binom.cdf(hits[i] - 1, n_iterations, 0.5)
            
            if p_value < alpha:
                confirmed.append(feature)
            elif p_value > (1 - alpha):
                rejected.append(feature)
            else:
                tentative.append(feature)
        
        self.selected_features = confirmed
        
        return {
            'confirmed_features': confirmed,
            'tentative_features': tentative,
            'rejected_features': rejected,
            'n_confirmed': len(confirmed),
            'hits': {f: int(h) for f, h in zip(feature_cols, hits)},
            'n_iterations': n_iterations,
            'alpha': alpha
        }
    
    # =========================================================================
    # SHAP-BASED SELECTION
    # =========================================================================
    
    def shap_selection(self, feature_cols: List[str], target_col: str,
                       n_features: int = None, threshold: float = None,
                       task: str = 'auto') -> Dict[str, Any]:
        """
        SHAP-based feature selection
        
        Selects features based on mean absolute SHAP values.
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            n_features: Number of features to select
            threshold: SHAP importance threshold (alternative to n_features)
            task: 'classification' or 'regression'
            
        Returns:
            Dictionary with selected features and SHAP values
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not installed. Install with: pip install shap'}
        
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Determine task type
        if task == 'auto':
            unique_vals = len(np.unique(y))
            task = 'classification' if unique_vals < 20 else 'regression'
        
        # Train model
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle classification (multi-output)
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dict
        shap_importance = {f: float(v) for f, v in zip(feature_cols, mean_abs_shap)}
        
        # Select features
        if n_features is not None:
            selected_indices = np.argsort(mean_abs_shap)[-n_features:]
        elif threshold is not None:
            selected_indices = np.where(mean_abs_shap >= threshold)[0]
        else:
            # Select top half
            n_features = max(1, len(feature_cols) // 2)
            selected_indices = np.argsort(mean_abs_shap)[-n_features:]
        
        selected_features = [feature_cols[i] for i in selected_indices]
        
        self.selected_features = selected_features
        self.feature_importances = shap_importance
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'shap_importance': shap_importance,
            'task': task
        }
    
    # =========================================================================
    # STATISTICAL SELECTION
    # =========================================================================
    
    def statistical_selection(self, feature_cols: List[str], target_col: str,
                               method: str = 'anova', k: int = None,
                               threshold: float = 0.05) -> Dict[str, Any]:
        """
        Statistical feature selection
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            method: Selection method
                   - 'anova': ANOVA F-test (classification)
                   - 'f_regression': F-test (regression)
                   - 'mutual_info': Mutual information
                   - 'chi2': Chi-squared test
            k: Number of features to select
            threshold: p-value threshold for significance
            
        Returns:
            Dictionary with selected features and statistics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Handle chi2 (requires non-negative)
        if method == 'chi2':
            X_scaled = X - X.min(axis=0)
        
        # Select scoring function
        if method == 'anova':
            score_func = f_classif
        elif method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info':
            unique_vals = len(np.unique(y))
            score_func = mutual_info_classif if unique_vals < 20 else mutual_info_regression
        elif method == 'chi2':
            score_func = chi2
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Calculate scores
        if k is not None:
            selector = SelectKBest(score_func, k=k)
            selector.fit(X_scaled, y)
        else:
            selector = SelectKBest(score_func, k='all')
            selector.fit(X_scaled, y)
        
        scores = selector.scores_
        p_values = selector.pvalues_ if hasattr(selector, 'pvalues_') and selector.pvalues_ is not None else None
        
        # Build results
        feature_scores = {f: float(s) for f, s in zip(feature_cols, scores)}
        
        if p_values is not None:
            feature_pvalues = {f: float(p) for f, p in zip(feature_cols, p_values)}
            # Select by p-value if no k specified
            if k is None:
                selected_features = [f for f, p in zip(feature_cols, p_values) if p < threshold]
            else:
                selected_mask = selector.get_support()
                selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
        else:
            feature_pvalues = None
            if k is not None:
                selected_mask = selector.get_support()
                selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
            else:
                # Select top half
                n_select = max(1, len(feature_cols) // 2)
                top_indices = np.argsort(scores)[-n_select:]
                selected_features = [feature_cols[i] for i in top_indices]
        
        self.selected_features = selected_features
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': feature_scores,
            'p_values': feature_pvalues,
            'method': method
        }
    
    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================
    
    def permutation_selection(self, feature_cols: List[str], target_col: str,
                               n_features: int = None, n_repeats: int = 10,
                               task: str = 'auto') -> Dict[str, Any]:
        """
        Permutation importance-based feature selection
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            n_features: Number of features to select
            n_repeats: Number of permutation repeats
            task: 'classification' or 'regression'
            
        Returns:
            Dictionary with selected features and importance scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Determine task
        if task == 'auto':
            unique_vals = len(np.unique(y))
            task = 'classification' if unique_vals < 20 else 'regression'
        
        # Train model
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = 'r2'
        
        model.fit(X, y)
        
        # Calculate permutation importance
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        
        importances = result.importances_mean
        importances_std = result.importances_std
        
        importance_dict = {f: {'importance': float(imp), 'std': float(std)}
                          for f, imp, std in zip(feature_cols, importances, importances_std)}
        
        # Select features
        if n_features is None:
            # Select features with positive importance
            selected_indices = np.where(importances > 0)[0]
            if len(selected_indices) == 0:
                n_features = max(1, len(feature_cols) // 2)
                selected_indices = np.argsort(importances)[-n_features:]
        else:
            selected_indices = np.argsort(importances)[-n_features:]
        
        selected_features = [feature_cols[i] for i in selected_indices]
        
        self.selected_features = selected_features
        self.feature_importances = importance_dict
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'importances': importance_dict,
            'task': task
        }
    
    # =========================================================================
    # L1-BASED SELECTION (LASSO)
    # =========================================================================
    
    def lasso_selection(self, feature_cols: List[str], target_col: str,
                        alpha: float = None, cv: int = 5) -> Dict[str, Any]:
        """
        L1 (Lasso) regularization-based feature selection
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            alpha: Regularization strength (None = use CV)
            cv: Cross-validation folds for alpha selection
            
        Returns:
            Dictionary with selected features and coefficients
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Lasso
        if alpha is None:
            lasso = LassoCV(cv=cv, random_state=42)
        else:
            lasso = Lasso(alpha=alpha, random_state=42)
        
        lasso.fit(X_scaled, y)
        
        # Get coefficients
        coefs = lasso.coef_
        
        # Selected features (non-zero coefficients)
        selected_mask = np.abs(coefs) > 1e-10
        selected_features = [f for f, selected in zip(feature_cols, selected_mask) if selected]
        
        coefficient_dict = {f: float(c) for f, c in zip(feature_cols, coefs)}
        
        self.selected_features = selected_features
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'coefficients': coefficient_dict,
            'alpha': float(lasso.alpha_) if hasattr(lasso, 'alpha_') else alpha,
            'r2_score': float(lasso.score(X_scaled, y))
        }
    
    # =========================================================================
    # SEQUENTIAL FEATURE SELECTION
    # =========================================================================
    
    def sequential_selection(self, feature_cols: List[str], target_col: str,
                              n_features: int = None, direction: str = 'forward',
                              task: str = 'auto', cv: int = 5) -> Dict[str, Any]:
        """
        Sequential feature selection (forward or backward)
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            n_features: Number of features to select
            direction: 'forward' or 'backward'
            task: 'classification' or 'regression'
            cv: Cross-validation folds
            
        Returns:
            Dictionary with selected features
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        X, y, error = self._prepare_data(feature_cols, target_col)
        if error:
            return {'error': error}
        
        # Determine task
        if task == 'auto':
            unique_vals = len(np.unique(y))
            task = 'classification' if unique_vals < 20 else 'regression'
        
        if n_features is None:
            n_features = max(1, len(feature_cols) // 2)
        
        # Select estimator
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            scoring = 'r2'
        
        # Sequential selection
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_features,
            direction=direction,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        sfs.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = sfs.get_support()
        selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
        
        self.selected_features = selected_features
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'direction': direction,
            'task': task
        }
    
    # =========================================================================
    # ENSEMBLE SELECTION
    # =========================================================================
    
    def ensemble_selection(self, feature_cols: List[str], target_col: str,
                            methods: List[str] = None, min_votes: int = None) -> Dict[str, Any]:
        """
        Ensemble feature selection using multiple methods
        
        Args:
            feature_cols: Feature columns
            target_col: Target column
            methods: List of methods to use (default: all)
            min_votes: Minimum votes for feature selection
            
        Returns:
            Dictionary with consensus selected features
        """
        if methods is None:
            methods = ['rfe', 'boruta', 'permutation', 'lasso', 'statistical']
        
        # Collect votes
        votes = {f: 0 for f in feature_cols}
        method_results = {}
        
        for method in methods:
            try:
                if method == 'rfe':
                    result = self.recursive_feature_elimination(
                        feature_cols, target_col, n_features=len(feature_cols)//2
                    )
                elif method == 'boruta':
                    result = self.boruta_selection(
                        feature_cols, target_col, n_iterations=50
                    )
                    if 'confirmed_features' in result:
                        result['selected_features'] = result['confirmed_features']
                elif method == 'permutation':
                    result = self.permutation_selection(
                        feature_cols, target_col
                    )
                elif method == 'lasso':
                    result = self.lasso_selection(feature_cols, target_col)
                elif method == 'statistical':
                    result = self.statistical_selection(
                        feature_cols, target_col
                    )
                else:
                    continue
                
                if 'selected_features' in result:
                    method_results[method] = result['selected_features']
                    for f in result['selected_features']:
                        votes[f] += 1
                        
            except Exception as e:
                method_results[method] = {'error': str(e)}
        
        # Select by votes
        if min_votes is None:
            min_votes = len(methods) // 2 + 1
        
        selected_features = [f for f, v in votes.items() if v >= min_votes]
        
        self.selected_features = selected_features
        
        return {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'votes': votes,
            'method_results': method_results,
            'min_votes': min_votes,
            'methods_used': methods
        }
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_feature_importance(self, importance_dict: Dict[str, float] = None,
                                 top_n: int = 20) -> plt.Figure:
        """Plot feature importance scores"""
        if importance_dict is None:
            importance_dict = self.feature_importances
        
        if importance_dict is None:
            return None
        
        # Handle nested dicts
        if isinstance(list(importance_dict.values())[0], dict):
            importance_dict = {k: v['importance'] for k, v in importance_dict.items()}
        
        # Sort and select top N
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        y_pos = range(len(features))
        colors = ['#2ecc71' if f in (self.selected_features or []) else '#3498db' for f in features]
        
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='Selected'),
                          Patch(facecolor='#3498db', label='Not Selected')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig
    
    def plot_selection_comparison(self, comparison_results: Dict[str, Any]) -> plt.Figure:
        """Plot comparison of different selection methods"""
        votes = comparison_results.get('votes', {})
        
        if not votes:
            return None
        
        sorted_features = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        features, vote_counts = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.25)))
        
        min_votes = comparison_results.get('min_votes', 0)
        colors = ['#2ecc71' if v >= min_votes else '#e74c3c' for v in vote_counts]
        
        y_pos = range(len(features))
        ax.barh(y_pos, vote_counts, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Methods Selecting Feature')
        ax.set_title(f'Feature Selection Consensus (threshold: {min_votes})')
        ax.axvline(x=min_votes - 0.5, color='black', linestyle='--', label='Selection Threshold')
        
        plt.tight_layout()
        return fig
