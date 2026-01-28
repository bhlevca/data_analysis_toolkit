"""
Model Validation Module
========================
Comprehensive model validation and evaluation tools including:
- Cross-validation (k-fold, stratified, leave-one-out, time series)
- Learning curves (bias/variance diagnosis)
- Calibration analysis (probability calibration)
- ROC/AUC analysis with confidence intervals
- Residual diagnostics (Q-Q, Cook's distance, leverage)
- Model comparison (AIC, BIC, likelihood ratio)

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
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, f1_score,
                             log_loss, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (KFold, LeaveOneOut, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_predict,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class ModelValidation:
    """Model validation and evaluation methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.last_results = {}
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    # =========================================================================
    # CROSS-VALIDATION METHODS
    # =========================================================================
    
    def cross_validate(self, model: BaseEstimator, features: List[str], target: str,
                       cv_type: str = 'kfold', n_splits: int = 5,
                       shuffle: bool = True, scoring: str = None,
                       return_train_score: bool = True) -> Dict[str, Any]:
        """
        Comprehensive cross-validation with multiple options
        
        Args:
            model: Sklearn-compatible model
            features: List of feature column names
            target: Target column name
            cv_type: 'kfold', 'stratified', 'loo', 'timeseries'
            n_splits: Number of folds (not used for LOO)
            shuffle: Whether to shuffle data
            scoring: Scoring metric (None for default)
            return_train_score: Also compute training scores
            
        Returns:
            Dictionary with CV results, scores, and statistics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) <= 20 and y.dtype in ['int64', 'int32', 'object', 'bool']
        
        # Select CV strategy
        if cv_type == 'stratified':
            if not is_classification:
                return {'error': 'Stratified CV only for classification'}
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        elif cv_type == 'loo':
            cv = LeaveOneOut()
            n_splits = len(X)
        elif cv_type == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_splits)
            shuffle = False  # Time series shouldn't be shuffled
        else:  # kfold
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        
        # Default scoring
        if scoring is None:
            scoring = 'accuracy' if is_classification else 'r2'
        
        # Perform cross-validation
        test_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Get predictions for all folds
        predictions = cross_val_predict(model, X, y, cv=cv)
        
        # Training scores if requested
        train_scores = None
        if return_train_score and cv_type != 'loo':
            train_scores = []
            for train_idx, _ in cv.split(X, y):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                if is_classification:
                    train_scores.append(accuracy_score(y_train, model_clone.predict(X_train)))
                else:
                    train_scores.append(r2_score(y_train, model_clone.predict(X_train)))
            train_scores = np.array(train_scores)
        
        # Calculate detailed metrics
        if is_classification:
            detailed_metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y, predictions, average='weighted', zero_division=0)
            }
        else:
            detailed_metrics = {
                'r2': r2_score(y, predictions),
                'mse': mean_squared_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'mae': mean_absolute_error(y, predictions)
            }
        
        results = {
            'cv_type': cv_type,
            'n_splits': n_splits,
            'scoring': scoring,
            'test_scores': test_scores.tolist(),
            'test_score_mean': float(test_scores.mean()),
            'test_score_std': float(test_scores.std()),
            'test_score_ci_lower': float(test_scores.mean() - 1.96 * test_scores.std() / np.sqrt(len(test_scores))),
            'test_score_ci_upper': float(test_scores.mean() + 1.96 * test_scores.std() / np.sqrt(len(test_scores))),
            'detailed_metrics': detailed_metrics,
            'predictions': predictions.tolist(),
            'actual': y.values.tolist(),
            'is_classification': is_classification
        }
        
        if train_scores is not None:
            results['train_scores'] = train_scores.tolist()
            results['train_score_mean'] = float(train_scores.mean())
            results['train_score_std'] = float(train_scores.std())
            results['overfit_ratio'] = float(train_scores.mean() / test_scores.mean()) if test_scores.mean() != 0 else float('inf')
        
        self.last_results = results
        return results
    
    def nested_cross_validation(self, model: BaseEstimator, features: List[str], target: str,
                                outer_cv: int = 5, inner_cv: int = 3,
                                param_grid: Dict = None) -> Dict[str, Any]:
        """
        Nested cross-validation for unbiased model evaluation with hyperparameter tuning
        
        Args:
            model: Base model
            features: Feature columns
            target: Target column
            outer_cv: Outer fold count (evaluation)
            inner_cv: Inner fold count (tuning)
            param_grid: Parameters to tune (if None, no tuning)
            
        Returns:
            Dictionary with nested CV results
        """
        from sklearn.model_selection import GridSearchCV
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        is_classification = len(np.unique(y)) <= 20
        outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42) if is_classification else KFold(n_splits=outer_cv, shuffle=True, random_state=42)
        
        outer_scores = []
        best_params_list = []
        
        for train_idx, test_idx in outer.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if param_grid:
                # Inner CV for hyperparameter tuning
                inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42) if is_classification else KFold(n_splits=inner_cv, shuffle=True, random_state=42)
                grid_search = GridSearchCV(clone(model), param_grid, cv=inner, scoring='accuracy' if is_classification else 'r2')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params_list.append(grid_search.best_params_)
            else:
                best_model = clone(model)
                best_model.fit(X_train, y_train)
            
            # Evaluate on outer fold
            if is_classification:
                score = accuracy_score(y_test, best_model.predict(X_test))
            else:
                score = r2_score(y_test, best_model.predict(X_test))
            outer_scores.append(score)
        
        return {
            'outer_cv': outer_cv,
            'inner_cv': inner_cv,
            'outer_scores': outer_scores,
            'mean_score': float(np.mean(outer_scores)),
            'std_score': float(np.std(outer_scores)),
            'best_params_per_fold': best_params_list if param_grid else None
        }
    
    # =========================================================================
    # LEARNING CURVES
    # =========================================================================
    
    def learning_curve_analysis(self, model: BaseEstimator, features: List[str], target: str,
                                cv: int = 5, train_sizes: np.ndarray = None,
                                scoring: str = None) -> Dict[str, Any]:
        """
        Generate learning curves to diagnose bias/variance
        
        Args:
            model: Model to evaluate
            features: Feature columns
            target: Target column
            cv: Cross-validation folds
            train_sizes: Array of training set sizes (fractions or absolute)
            scoring: Scoring metric
            
        Returns:
            Dictionary with learning curve data
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        is_classification = len(np.unique(y)) <= 20
        if scoring is None:
            scoring = 'accuracy' if is_classification else 'r2'
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring,
            shuffle=True, random_state=42, n_jobs=-1
        )
        
        # Calculate means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Diagnose bias/variance
        final_train = train_mean[-1]
        final_test = test_mean[-1]
        gap = final_train - final_test
        
        if final_test < 0.6:  # Low performance
            if gap < 0.1:
                diagnosis = "High bias (underfitting): Both training and test scores are low. Try more complex model or more features."
            else:
                diagnosis = "High variance (overfitting): Large gap between train and test. Try regularization, more data, or simpler model."
        else:
            if gap < 0.05:
                diagnosis = "Good fit: Model generalizes well with small train-test gap."
            elif gap < 0.15:
                diagnosis = "Slight overfitting: Consider regularization or more training data."
            else:
                diagnosis = "Overfitting: Significant gap between train and test scores."
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'test_scores_mean': test_mean.tolist(),
            'test_scores_std': test_std.tolist(),
            'final_train_score': float(final_train),
            'final_test_score': float(final_test),
            'train_test_gap': float(gap),
            'diagnosis': diagnosis,
            'scoring': scoring
        }
    
    def plot_learning_curve(self, results: Dict[str, Any], title: str = "Learning Curve") -> plt.Figure:
        """Plot learning curve from results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_sizes = results['train_sizes']
        train_mean = np.array(results['train_scores_mean'])
        train_std = np.array(results['train_scores_std'])
        test_mean = np.array(results['test_scores_mean'])
        test_std = np.array(results['test_scores_std'])
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add diagnosis annotation
        ax.annotate(results['diagnosis'], xy=(0.5, 0.02), xycoords='axes fraction',
                   fontsize=9, ha='center', style='italic',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # CALIBRATION ANALYSIS
    # =========================================================================
    
    def calibration_analysis(self, model: BaseEstimator, features: List[str], target: str,
                             n_bins: int = 10, strategy: str = 'uniform') -> Dict[str, Any]:
        """
        Analyze probability calibration for classification models
        
        Args:
            model: Trained classifier with predict_proba
            features: Feature columns
            target: Target column
            n_bins: Number of bins for calibration curve
            strategy: 'uniform' or 'quantile'
            
        Returns:
            Dictionary with calibration metrics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Get probability predictions via cross-validation
        from sklearn.model_selection import cross_val_predict
        
        try:
            y_prob = cross_val_predict(model, X, y, cv=5, method='predict_proba')
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]  # Binary classification
        except Exception as e:
            return {'error': f'Model must support predict_proba: {str(e)}'}
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy=strategy)
        
        # Brier score
        brier = brier_score_loss(y, y_prob)
        
        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y[mask].mean()
                bin_confidence = y_prob[mask].mean()
                ece += mask.sum() * abs(bin_accuracy - bin_confidence)
        ece /= len(y)
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0
        
        return {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier_score': float(brier),
            'expected_calibration_error': float(ece),
            'max_calibration_error': float(mce),
            'n_bins': n_bins,
            'interpretation': self._interpret_calibration(ece, brier)
        }
    
    def _interpret_calibration(self, ece: float, brier: float) -> str:
        """Interpret calibration metrics"""
        if ece < 0.05:
            ece_interp = "excellent calibration"
        elif ece < 0.10:
            ece_interp = "good calibration"
        elif ece < 0.20:
            ece_interp = "moderate calibration"
        else:
            ece_interp = "poor calibration"
        
        if brier < 0.1:
            brier_interp = "excellent probability estimates"
        elif brier < 0.2:
            brier_interp = "good probability estimates"
        else:
            brier_interp = "needs improvement"
        
        return f"ECE indicates {ece_interp}. Brier score suggests {brier_interp}."
    
    def plot_calibration_curve(self, results: Dict[str, Any], title: str = "Calibration Curve") -> plt.Figure:
        """Plot calibration curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        prob_true = results['prob_true']
        prob_pred = results['prob_pred']
        
        # Calibration plot
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 's-', label='Model')
        ax1.set_xlabel('Mean predicted probability')
        ax1.set_ylabel('Fraction of positives')
        ax1.set_title('Calibration Curve (Reliability Diagram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics
        metrics_text = f"Brier Score: {results['brier_score']:.4f}\n"
        metrics_text += f"ECE: {results['expected_calibration_error']:.4f}\n"
        metrics_text += f"MCE: {results['max_calibration_error']:.4f}"
        ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes, fontsize=14,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.text(0.5, 0.2, results['interpretation'], transform=ax2.transAxes,
                fontsize=10, ha='center', style='italic', wrap=True)
        ax2.axis('off')
        ax2.set_title('Calibration Metrics')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # ROC/AUC ANALYSIS
    # =========================================================================
    
    def roc_analysis(self, model: BaseEstimator, features: List[str], target: str,
                     n_bootstrap: int = 100, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Comprehensive ROC analysis with confidence intervals
        
        Args:
            model: Classifier
            features: Feature columns
            target: Target column
            n_bootstrap: Bootstrap samples for CI
            confidence: Confidence level
            
        Returns:
            Dictionary with ROC curve, AUC, and CIs
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Get predictions
        try:
            y_prob = cross_val_predict(model, X, y, cv=5, method='predict_proba')
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
        except:
            return {'error': 'Model must support predict_proba'}
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Bootstrap CI for AUC
        bootstrap_aucs = []
        n = len(y)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            try:
                bootstrap_aucs.append(roc_auc_score(y.iloc[idx], y_prob[idx]))
            except:
                pass
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
        
        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc),
            'auc_ci_lower': float(ci_lower),
            'auc_ci_upper': float(ci_upper),
            'optimal_threshold': float(optimal_threshold),
            'optimal_sensitivity': float(tpr[optimal_idx]),
            'optimal_specificity': float(1 - fpr[optimal_idx]),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'pr_auc': float(pr_auc),
            'interpretation': self._interpret_auc(roc_auc)
        }
    
    def _interpret_auc(self, auc_val: float) -> str:
        """Interpret AUC value"""
        if auc_val >= 0.9:
            return "Excellent discrimination"
        elif auc_val >= 0.8:
            return "Good discrimination"
        elif auc_val >= 0.7:
            return "Fair discrimination"
        elif auc_val >= 0.6:
            return "Poor discrimination"
        else:
            return "No discrimination (random guessing)"
    
    def plot_roc_curve(self, results: Dict[str, Any], title: str = "ROC Curve") -> plt.Figure:
        """Plot ROC and PR curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC curve
        ax1.plot(results['fpr'], results['tpr'], 'b-', lw=2,
                label=f"ROC (AUC = {results['auc']:.3f}, 95% CI: [{results['auc_ci_lower']:.3f}, {results['auc_ci_upper']:.3f}])")
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax1.scatter([1 - results['optimal_specificity']], [results['optimal_sensitivity']],
                   marker='o', s=100, c='red', label=f"Optimal (t={results['optimal_threshold']:.2f})")
        ax1.set_xlabel('False Positive Rate (1 - Specificity)')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # PR curve
        ax2.plot(results['recall'], results['precision'], 'g-', lw=2,
                label=f"PR (AUC = {results['pr_auc']:.3f})")
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # RESIDUAL DIAGNOSTICS
    # =========================================================================
    
    def residual_diagnostics(self, model: BaseEstimator, features: List[str], target: str,
                             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Comprehensive residual diagnostics for regression models
        
        Returns:
            Dictionary with residual analysis, influence measures, and diagnostics
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        from sklearn.model_selection import train_test_split
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Split and fit
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        y_pred = model_clone.predict(X_test)
        residuals = y_test.values - y_pred
        
        # Standardized residuals
        residual_std = np.std(residuals)
        standardized_residuals = residuals / residual_std if residual_std > 0 else residuals
        
        # Normality test
        if len(residuals) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        
        # Breusch-Pagan test for heteroscedasticity (if linear model with intercept)
        try:
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X_test)
            bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_const)
        except:
            bp_stat, bp_p = np.nan, np.nan
        
        # Leverage and Cook's distance (for linear regression)
        try:
            X_train_np = X_train.values
            H = X_train_np @ np.linalg.pinv(X_train_np.T @ X_train_np) @ X_train_np.T
            leverage = np.diag(H)
            
            # Cook's distance
            train_pred = model_clone.predict(X_train)
            train_residuals = y_train.values - train_pred
            mse = np.mean(train_residuals**2)
            p = X_train.shape[1]
            cooks_d = (train_residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)
            
            influential_points = np.where(cooks_d > 4 / len(X_train))[0]
        except:
            leverage = None
            cooks_d = None
            influential_points = []
        
        # Q-Q correlation
        sorted_residuals = np.sort(standardized_residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        qq_correlation = np.corrcoef(theoretical_quantiles, sorted_residuals)[0, 1]
        
        # Interpret diagnostics
        diagnostics = []
        if shapiro_p < 0.05:
            diagnostics.append("Residuals may not be normally distributed (Shapiro-Wilk p < 0.05)")
        if dw_stat < 1.5 or dw_stat > 2.5:
            diagnostics.append(f"Potential autocorrelation (Durbin-Watson = {dw_stat:.2f})")
        if bp_p < 0.05:
            diagnostics.append("Potential heteroscedasticity (Breusch-Pagan p < 0.05)")
        if len(influential_points) > 0:
            diagnostics.append(f"{len(influential_points)} influential points detected (Cook's D > 4/n)")
        if not diagnostics:
            diagnostics.append("No major violations detected")
        
        return {
            'residuals': residuals.tolist(),
            'standardized_residuals': standardized_residuals.tolist(),
            'predicted': y_pred.tolist(),
            'actual': y_test.values.tolist(),
            'shapiro_stat': float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            'shapiro_p': float(shapiro_p) if not np.isnan(shapiro_p) else None,
            'durbin_watson': float(dw_stat),
            'breusch_pagan_stat': float(bp_stat) if not np.isnan(bp_stat) else None,
            'breusch_pagan_p': float(bp_p) if not np.isnan(bp_p) else None,
            'leverage': leverage.tolist() if leverage is not None else None,
            'cooks_distance': cooks_d.tolist() if cooks_d is not None else None,
            'influential_points': influential_points.tolist() if len(influential_points) > 0 else [],
            'qq_correlation': float(qq_correlation),
            'diagnostics': diagnostics
        }
    
    def plot_residual_diagnostics(self, results: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive residual diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        residuals = np.array(results['residuals'])
        std_residuals = np.array(results['standardized_residuals'])
        predicted = np.array(results['predicted'])
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(predicted, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Q-Q plot
        stats.probplot(std_residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title(f'Normal Q-Q (r = {results["qq_correlation"]:.3f})')
        
        # 3. Scale-Location
        axes[1, 0].scatter(predicted, np.sqrt(np.abs(std_residuals)), alpha=0.5)
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('√|Standardized residuals|')
        axes[1, 0].set_title('Scale-Location')
        
        # 4. Histogram of residuals
        axes[1, 1].hist(std_residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        x_norm = np.linspace(std_residuals.min(), std_residuals.max(), 100)
        axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm), 'r-', lw=2, label='Normal')
        axes[1, 1].set_xlabel('Standardized Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residual Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================
    
    def model_comparison_aic_bic(self, models: Dict[str, BaseEstimator], 
                                  features: List[str], target: str) -> Dict[str, Any]:
        """
        Compare models using AIC and BIC
        
        Args:
            models: Dictionary of {name: model}
            features: Feature columns
            target: Target column
            
        Returns:
            Dictionary with comparison results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        n = len(X)
        
        results = {}
        
        for name, model in models.items():
            model_clone = clone(model)
            model_clone.fit(X, y)
            
            y_pred = model_clone.predict(X)
            residuals = y.values - y_pred
            
            # RSS (Residual Sum of Squares)
            rss = np.sum(residuals**2)
            
            # Number of parameters (approximate)
            if hasattr(model_clone, 'coef_'):
                k = len(np.atleast_1d(model_clone.coef_)) + 1  # +1 for intercept
            else:
                k = len(features) + 1  # Approximate
            
            # Log-likelihood (assuming Gaussian errors)
            log_likelihood = -n/2 * (np.log(2*np.pi) + np.log(rss/n) + 1)
            
            # AIC and BIC
            aic = 2*k - 2*log_likelihood
            bic = k*np.log(n) - 2*log_likelihood
            
            # Adjusted R²
            r2 = 1 - rss / np.sum((y.values - y.mean())**2)
            adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
            
            results[name] = {
                'aic': float(aic),
                'bic': float(bic),
                'log_likelihood': float(log_likelihood),
                'r2': float(r2),
                'adjusted_r2': float(adj_r2),
                'n_parameters': k,
                'rss': float(rss)
            }
        
        # Find best models
        aic_values = {k: v['aic'] for k, v in results.items()}
        bic_values = {k: v['bic'] for k, v in results.items()}
        
        best_aic = min(aic_values, key=aic_values.get)
        best_bic = min(bic_values, key=bic_values.get)
        
        # Calculate delta AIC and Akaike weights
        min_aic = min(aic_values.values())
        for name in results:
            delta_aic = results[name]['aic'] - min_aic
            results[name]['delta_aic'] = float(delta_aic)
        
        # Akaike weights
        exp_deltas = {k: np.exp(-0.5 * v['delta_aic']) for k, v in results.items()}
        sum_exp = sum(exp_deltas.values())
        for name in results:
            results[name]['akaike_weight'] = float(exp_deltas[name] / sum_exp)
        
        return {
            'models': results,
            'best_by_aic': best_aic,
            'best_by_bic': best_bic,
            'recommendation': f"Best model by AIC: {best_aic}, by BIC: {best_bic}"
        }
