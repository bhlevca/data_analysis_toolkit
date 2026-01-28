"""
Model Interpretability Module
==============================
Tools for interpreting machine learning model predictions:
- SHAP values (feature importance and interaction)
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
- Feature Importance (permutation-based)
- LIME explanations (local interpretable model)
- Interaction effects analysis

Version: 1.0

Requirements:
    pip install shap lime
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelInterpretability:
    """Model interpretability and explanation tools"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.model = None
        self.explainer = None
        self.shap_values = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    def set_model(self, model: BaseEstimator):
        """Set the model to interpret"""
        self.model = model
    
    # =========================================================================
    # SHAP VALUES
    # =========================================================================
    
    def shap_analysis(self, model: BaseEstimator, features: List[str], target: str = None,
                      X: np.ndarray = None, sample_size: int = 100,
                      explainer_type: str = 'auto') -> Dict[str, Any]:
        """
        Compute SHAP values for model interpretation
        
        Args:
            model: Trained model
            features: Feature column names
            target: Target column (for fitting if needed)
            X: Feature matrix (if not using self.df)
            sample_size: Background data sample size
            explainer_type: 'auto', 'tree', 'kernel', 'linear', 'deep'
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not installed. Install with: pip install shap'}
        
        # Prepare data
        if X is None:
            if self.df is None:
                return {'error': 'No data provided'}
            X = self.df[features].dropna().values
        
        # Sample background data
        if len(X) > sample_size:
            background_idx = np.random.choice(len(X), sample_size, replace=False)
            X_background = X[background_idx]
        else:
            X_background = X
        
        # Create explainer based on model type
        self.model = model
        
        if explainer_type == 'auto':
            # Auto-detect best explainer
            model_name = type(model).__name__.lower()
            if any(t in model_name for t in ['forest', 'tree', 'boost', 'xgb', 'lgb', 'catboost']):
                explainer_type = 'tree'
            elif 'linear' in model_name or 'logistic' in model_name:
                explainer_type = 'linear'
            else:
                explainer_type = 'kernel'
        
        try:
            if explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(model)
                shap_values = self.explainer.shap_values(X)
            elif explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_background)
                shap_values = self.explainer.shap_values(X)
            elif explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(model.predict, X_background)
                shap_values = self.explainer.shap_values(X)
            elif explainer_type == 'deep':
                self.explainer = shap.DeepExplainer(model, X_background)
                shap_values = self.explainer.shap_values(X)
            else:
                # Fallback to kernel
                self.explainer = shap.KernelExplainer(model.predict, X_background)
                shap_values = self.explainer.shap_values(X)
        except Exception as e:
            return {'error': f'SHAP calculation failed: {str(e)}'}
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For classification, use absolute mean across classes
            shap_values_abs = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values_abs = np.abs(shap_values)
        
        # Feature importance from SHAP
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.mean(shap_values_abs, axis=0)
        }).sort_values('importance', ascending=False)
        
        self.shap_values = shap_values
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance.to_dict('records'),
            'expected_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None,
            'explainer_type': explainer_type,
            'n_samples': len(X),
            'features': features
        }
    
    def plot_shap_summary(self, shap_values: np.ndarray = None, X: np.ndarray = None,
                          features: List[str] = None, plot_type: str = 'dot',
                          max_display: int = 20) -> plt.Figure:
        """
        Create SHAP summary plot
        
        Args:
            shap_values: SHAP values (or use stored values)
            X: Feature matrix
            features: Feature names
            plot_type: 'dot', 'bar', 'violin'
            max_display: Maximum features to display
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            return None
        
        if shap_values is None:
            shap_values = self.shap_values
        if shap_values is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        if plot_type == 'bar':
            importance = np.mean(np.abs(shap_values), axis=0)
            sorted_idx = np.argsort(importance)[-max_display:]
            
            ax.barh(range(len(sorted_idx)), importance[sorted_idx])
            if features:
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([features[i] for i in sorted_idx])
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('SHAP Feature Importance')
        else:
            # Use SHAP's built-in plotting
            plt.close(fig)
            shap.summary_plot(shap_values, X, feature_names=features, 
                            plot_type=plot_type, max_display=max_display, show=False)
            fig = plt.gcf()
        
        plt.tight_layout()
        return fig
    
    def plot_shap_dependence(self, feature: str, shap_values: np.ndarray = None,
                             X: pd.DataFrame = None, interaction_feature: str = None) -> plt.Figure:
        """
        Create SHAP dependence plot for a single feature
        
        Args:
            feature: Feature to plot
            shap_values: SHAP values
            X: Feature data
            interaction_feature: Feature for color interaction
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            return None
        
        if shap_values is None:
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        if X is not None:
            feature_idx = list(X.columns).index(feature) if feature in X.columns else 0
            shap.dependence_plot(feature_idx, shap_values, X, 
                               interaction_index=interaction_feature,
                               show=False)
        
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def explain_prediction(self, model: BaseEstimator, instance: Union[np.ndarray, pd.Series],
                          features: List[str], X_background: np.ndarray = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP
        
        Args:
            model: Trained model
            instance: Single instance to explain
            features: Feature names
            X_background: Background data for explainer
            
        Returns:
            Dictionary with individual prediction explanation
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not installed'}
        
        if isinstance(instance, pd.Series):
            instance = instance.values.reshape(1, -1)
        elif len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Create explainer if not exists
        if self.explainer is None or self.model != model:
            if X_background is None:
                return {'error': 'Background data required for new explainer'}
            self.explainer = shap.KernelExplainer(model.predict, X_background)
            self.model = model
        
        shap_values = self.explainer.shap_values(instance)
        
        # Get prediction
        prediction = model.predict(instance)[0]
        
        # Create explanation
        contributions = list(zip(features, shap_values[0] if isinstance(shap_values, list) else shap_values.flatten()))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': float(prediction),
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None,
            'contributions': [{'feature': f, 'contribution': float(c)} for f, c in contributions],
            'top_positive': [(f, float(c)) for f, c in contributions if c > 0][:5],
            'top_negative': [(f, float(c)) for f, c in contributions if c < 0][:5]
        }
    
    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================
    
    def permutation_feature_importance(self, model: BaseEstimator, features: List[str],
                                        target: str, n_repeats: int = 10,
                                        scoring: str = None) -> Dict[str, Any]:
        """
        Calculate permutation-based feature importance
        
        Args:
            model: Trained model
            features: Feature columns
            target: Target column
            n_repeats: Number of permutation repeats
            scoring: Scoring metric
            
        Returns:
            Dictionary with feature importance
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model if needed
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Calculate permutation importance
        result = permutation_importance(model_clone, X_test, y_test, 
                                        n_repeats=n_repeats, random_state=42,
                                        scoring=scoring, n_jobs=-1)
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return {
            'feature_importance': importance_df.to_dict('records'),
            'importances': result.importances.tolist(),
            'scoring': scoring,
            'n_repeats': n_repeats
        }
    
    def plot_permutation_importance(self, results: Dict[str, Any], 
                                    top_n: int = 20) -> plt.Figure:
        """Plot permutation importance"""
        importance = results['feature_importance'][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        features = [item['feature'] for item in importance]
        means = [item['importance_mean'] for item in importance]
        stds = [item['importance_std'] for item in importance]
        
        y_pos = range(len(features))
        ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean importance decrease')
        ax.set_title('Permutation Feature Importance')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # PARTIAL DEPENDENCE PLOTS
    # =========================================================================
    
    def partial_dependence(self, model: BaseEstimator, features: List[str],
                          target: str, pdp_features: List[Union[str, Tuple[str, str]]],
                          grid_resolution: int = 50) -> Dict[str, Any]:
        """
        Calculate partial dependence for specified features
        
        Args:
            model: Trained model
            features: All feature columns
            target: Target column
            pdp_features: Features to compute PDP for (can include tuples for interactions)
            grid_resolution: Grid resolution
            
        Returns:
            Dictionary with PDP results
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Fit model
        model_clone = clone(model)
        model_clone.fit(X, y)
        
        results = {}
        for feat in pdp_features:
            if isinstance(feat, tuple):
                feat_idx = [features.index(f) for f in feat]
            else:
                feat_idx = [features.index(feat)]
            
            from sklearn.inspection import partial_dependence as sklearn_pdp
            pdp_result = sklearn_pdp(model_clone, X, feat_idx, 
                                     grid_resolution=grid_resolution,
                                     kind='average')
            
            results[str(feat)] = {
                'values': pdp_result['average'][0].tolist(),
                'grid': [g.tolist() for g in pdp_result['grid_values']]
            }
        
        return {
            'pdp_results': results,
            'features_analyzed': pdp_features
        }
    
    def plot_partial_dependence(self, model: BaseEstimator, features: List[str],
                                target: str, pdp_features: List[Union[int, str, Tuple]],
                                kind: str = 'both') -> plt.Figure:
        """
        Create partial dependence plots
        
        Args:
            model: Trained model
            features: All feature columns
            target: Target column
            pdp_features: Features to plot (indices, names, or tuples)
            kind: 'average', 'individual', or 'both'
            
        Returns:
            Matplotlib figure
        """
        if self.df is None:
            return None
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Fit model
        model_clone = clone(model)
        model_clone.fit(X, y)
        
        # Convert feature names to indices
        if isinstance(pdp_features[0], str):
            pdp_features = [features.index(f) if isinstance(f, str) else f for f in pdp_features]
        
        fig, ax = plt.subplots(figsize=(12, 4 * ((len(pdp_features) + 1) // 2)))
        
        display = PartialDependenceDisplay.from_estimator(
            model_clone, X, pdp_features,
            kind=kind, subsample=min(1000, len(X)),
            n_jobs=-1, random_state=42, ax=ax
        )
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # LIME EXPLANATIONS
    # =========================================================================
    
    def lime_explain(self, model: BaseEstimator, instance: Union[np.ndarray, pd.Series],
                     features: List[str], X_train: np.ndarray = None,
                     mode: str = 'regression', num_features: int = 10) -> Dict[str, Any]:
        """
        Explain prediction using LIME
        
        Args:
            model: Trained model
            instance: Instance to explain
            features: Feature names
            X_train: Training data for explainer
            mode: 'regression' or 'classification'
            num_features: Number of features in explanation
            
        Returns:
            Dictionary with LIME explanation
        """
        if not LIME_AVAILABLE:
            return {'error': 'LIME not installed. Install with: pip install lime'}
        
        if X_train is None and self.df is not None:
            X_train = self.df[features].dropna().values
        elif X_train is None:
            return {'error': 'Training data required'}
        
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=features,
            mode=mode,
            discretize_continuous=True
        )
        
        if isinstance(instance, pd.Series):
            instance = instance.values
        
        # Get explanation
        if mode == 'classification':
            exp = explainer.explain_instance(instance, model.predict_proba, 
                                            num_features=num_features)
        else:
            exp = explainer.explain_instance(instance, model.predict,
                                            num_features=num_features)
        
        # Extract explanation
        explanation = exp.as_list()
        
        return {
            'prediction': float(model.predict(instance.reshape(1, -1))[0]),
            'explanation': [{'feature': f, 'weight': float(w)} for f, w in explanation],
            'intercept': float(exp.intercept[0]) if hasattr(exp, 'intercept') else None,
            'local_pred': float(exp.local_pred[0]) if hasattr(exp, 'local_pred') else None,
            'score': float(exp.score) if hasattr(exp, 'score') else None
        }
    
    # =========================================================================
    # INTERACTION ANALYSIS
    # =========================================================================
    
    def feature_interactions(self, model: BaseEstimator, features: List[str],
                             target: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP interaction values
        
        Args:
            model: Trained model (tree-based works best)
            features: Feature columns
            target: Target column
            top_n: Number of top interactions to return
            
        Returns:
            Dictionary with interaction analysis
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP required for interaction analysis'}
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]
        
        # Fit model
        model_clone = clone(model)
        model_clone.fit(X, y)
        
        # Try to get interaction values (only works for tree models)
        try:
            explainer = shap.TreeExplainer(model_clone)
            shap_interaction = explainer.shap_interaction_values(X.values[:min(500, len(X))])
            
            # Sum absolute interactions
            if isinstance(shap_interaction, list):
                shap_interaction = np.abs(np.array(shap_interaction)).mean(axis=0)
            
            interaction_matrix = np.abs(shap_interaction).mean(axis=0)
            
            # Get top interactions (excluding diagonal)
            n_features = len(features)
            interactions = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interactions.append({
                        'feature1': features[i],
                        'feature2': features[j],
                        'interaction_strength': float(interaction_matrix[i, j])
                    })
            
            interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            return {
                'top_interactions': interactions[:top_n],
                'interaction_matrix': interaction_matrix.tolist(),
                'features': features
            }
            
        except Exception as e:
            # Fallback: use correlation of SHAP values
            result = self.shap_analysis(model_clone, features, X=X.values)
            if 'error' in result:
                return result
            
            shap_values = result['shap_values']
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Correlation of SHAP values as proxy for interaction
            shap_corr = np.corrcoef(shap_values.T)
            
            interactions = []
            n_features = len(features)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interactions.append({
                        'feature1': features[i],
                        'feature2': features[j],
                        'interaction_strength': float(abs(shap_corr[i, j]))
                    })
            
            interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            return {
                'top_interactions': interactions[:top_n],
                'method': 'shap_correlation',
                'features': features
            }
    
    def plot_interaction_heatmap(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot feature interaction heatmap"""
        if 'interaction_matrix' not in results:
            return None
        
        matrix = np.array(results['interaction_matrix'])
        features = results['features']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap='RdBu_r')
        ax.set_xticks(range(len(features)))
        ax.set_yticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_yticklabels(features)
        
        plt.colorbar(im, ax=ax, label='Interaction Strength')
        ax.set_title('Feature Interaction Heatmap')
        
        plt.tight_layout()
        return fig
