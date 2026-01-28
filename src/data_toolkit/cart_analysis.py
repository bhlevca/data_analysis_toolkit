"""
CART Analysis Module with Sensitivity Analysis Integration

Provides Classification and Regression Trees (CART) analysis integrated with
Morris Screening sensitivity analysis for parameter importance workflow:

    Morris Screening → CART Analysis → Monte Carlo Simulations

This workflow is useful for:
- Environmental modeling (e.g., Cladophora biomass predictions)
- Parameter screening and dimensionality reduction
- Uncertainty quantification
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class CARTAnalysis:
    """
    Classification and Regression Trees (CART) analysis with sensitivity integration.
    
    Designed for workflows where:
    1. Morris Screening identifies important parameters
    2. CART models relationships using top parameters
    3. Monte Carlo quantifies prediction uncertainty
    """
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.is_classifier = False
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze."""
        self.df = df
    
    def cart_regression(
        self,
        feature_cols: List[str],
        target_col: str,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit a CART regression tree.
        
        Args:
            feature_cols: List of feature column names
            target_col: Target column name
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf nodes
            test_size: Fraction for test set
            random_state: Random seed
            **kwargs: Additional DecisionTreeRegressor parameters
            
        Returns:
            Dictionary with model, metrics, feature importance, and tree structure
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        X = self.df[feature_cols].dropna()
        y = self.df.loc[X.index, target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit model
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
        self.model.fit(X_train, y_train)
        
        self.feature_names = feature_cols
        self.target_name = target_col
        self.is_classifier = False
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        # Feature importance ranking
        importance = self.model.feature_importances_
        importance_ranking = sorted(
            zip(feature_cols, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'model': self.model,
            'model_type': 'CART Regression',
            'feature_names': feature_cols,
            'target_name': target_col,
            'metrics': {
                'train_r2': float(r2_score(y_train, y_train_pred)),
                'test_r2': float(r2_score(y_test, y_test_pred)),
                'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std())
            },
            'feature_importance': {
                col: float(imp) for col, imp in zip(feature_cols, importance)
            },
            'importance_ranking': importance_ranking,
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves(),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def cart_classification(
        self,
        feature_cols: List[str],
        target_col: str,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit a CART classification tree.
        
        Args:
            feature_cols: List of feature column names
            target_col: Target column name (categorical)
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf nodes
            test_size: Fraction for test set
            random_state: Random seed
            **kwargs: Additional DecisionTreeClassifier parameters
            
        Returns:
            Dictionary with model, metrics, feature importance, and tree structure
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        X = self.df[feature_cols].dropna()
        y = self.df.loc[X.index, target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit model
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
        self.model.fit(X_train, y_train)
        
        self.feature_names = feature_cols
        self.target_name = target_col
        self.is_classifier = True
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        # Feature importance ranking
        importance = self.model.feature_importances_
        importance_ranking = sorted(
            zip(feature_cols, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'model': self.model,
            'model_type': 'CART Classification',
            'feature_names': feature_cols,
            'target_name': target_col,
            'classes': list(self.model.classes_),
            'metrics': {
                'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
                'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std())
            },
            'feature_importance': {
                col: float(imp) for col, imp in zip(feature_cols, importance)
            },
            'importance_ranking': importance_ranking,
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves(),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def from_morris_screening(
        self,
        morris_results: Dict[str, Any],
        target_col: str,
        top_n: int = None,
        mu_star_threshold: float = None,
        max_depth: int = 5,
        task: str = 'regression'
    ) -> Dict[str, Any]:
        """
        Build CART model using top parameters from Morris Screening.
        
        This is the integration point for the Morris → CART workflow.
        
        Args:
            morris_results: Results dictionary from SensitivityAnalysis.morris_screening()
            target_col: Target column name
            top_n: Select top N parameters by μ* (default: auto-select)
            mu_star_threshold: Select parameters with μ* above threshold
            max_depth: Maximum tree depth
            task: 'regression' or 'classification'
            
        Returns:
            CART results with selected parameters noted
        """
        if self.df is None:
            return {'error': 'No data loaded'}
        
        if 'parameter_ranking' not in morris_results:
            return {'error': 'Invalid Morris results - missing parameter_ranking'}
        
        ranking = morris_results['parameter_ranking']
        
        # Select parameters
        if mu_star_threshold is not None:
            # Select by threshold
            selected = [p['parameter'] for p in ranking if p['mu_star'] >= mu_star_threshold]
        elif top_n is not None:
            # Select top N
            selected = [p['parameter'] for p in ranking[:top_n]]
        else:
            # Auto-select: parameters with μ* > 10% of max μ*
            max_mu = max(p['mu_star'] for p in ranking)
            threshold = 0.1 * max_mu
            selected = [p['parameter'] for p in ranking if p['mu_star'] >= threshold]
        
        if len(selected) == 0:
            return {'error': 'No parameters selected - try lowering threshold'}
        
        # Build CART
        if task == 'regression':
            result = self.cart_regression(selected, target_col, max_depth=max_depth)
        else:
            result = self.cart_classification(selected, target_col, max_depth=max_depth)
        
        # Add Morris context
        result['morris_integration'] = {
            'total_parameters_screened': len(ranking),
            'parameters_selected': len(selected),
            'selected_parameters': selected,
            'selection_method': f"top_{top_n}" if top_n else f"threshold_{mu_star_threshold}" if mu_star_threshold else "auto_10%",
            'morris_rankings': {p['parameter']: p for p in ranking if p['parameter'] in selected}
        }
        
        return result
    
    def generate_hypercube(
        self,
        feature_cols: List[str] = None,
        n_samples: int = 1000,
        bounds: Dict[str, Tuple[float, float]] = None,
        method: str = 'lhs'
    ) -> pd.DataFrame:
        """
        Generate Latin Hypercube samples for Monte Carlo simulation.
        
        Args:
            feature_cols: Columns to sample (default: current feature_names)
            n_samples: Number of samples to generate
            bounds: Dict of {column: (min, max)} bounds. If None, use data range.
            method: 'lhs' (Latin Hypercube) or 'random' (uniform random)
            
        Returns:
            DataFrame with sampled parameter combinations
        """
        if feature_cols is None:
            feature_cols = self.feature_names
        
        if feature_cols is None:
            return pd.DataFrame({'error': ['No features specified']})
        
        # Get bounds
        if bounds is None:
            bounds = {}
            for col in feature_cols:
                if self.df is not None and col in self.df.columns:
                    bounds[col] = (float(self.df[col].min()), float(self.df[col].max()))
                else:
                    bounds[col] = (0, 1)  # Default
        
        n_params = len(feature_cols)
        
        if method == 'lhs':
            # Latin Hypercube Sampling
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=n_params)
                sample = sampler.random(n=n_samples)
            except ImportError:
                # Fallback to simple LHS
                sample = np.zeros((n_samples, n_params))
                for i in range(n_params):
                    sample[:, i] = (np.random.permutation(n_samples) + np.random.random(n_samples)) / n_samples
        else:
            # Uniform random
            sample = np.random.random((n_samples, n_params))
        
        # Scale to bounds
        result = pd.DataFrame(columns=feature_cols)
        for i, col in enumerate(feature_cols):
            low, high = bounds[col]
            result[col] = low + sample[:, i] * (high - low)
        
        return result
    
    def monte_carlo_predictions(
        self,
        hypercube: pd.DataFrame = None,
        n_samples: int = 1000,
        perturbation_std: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo predictions using the fitted CART model.
        
        Args:
            hypercube: Pre-generated hypercube DataFrame (from generate_hypercube)
            n_samples: Number of samples if generating new hypercube
            perturbation_std: Std of parameter perturbations (fraction of range)
            
        Returns:
            Dictionary with prediction statistics and uncertainty measures
        """
        if self.model is None:
            return {'error': 'No model fitted - run cart_regression or cart_classification first'}
        
        # Generate hypercube if not provided
        if hypercube is None:
            hypercube = self.generate_hypercube(n_samples=n_samples)
        
        # Ensure columns match
        missing_cols = set(self.feature_names) - set(hypercube.columns)
        if missing_cols:
            return {'error': f'Hypercube missing columns: {missing_cols}'}
        
        # Make predictions
        X = hypercube[self.feature_names]
        predictions = self.model.predict(X)
        
        if self.is_classifier:
            # Classification: return class distribution
            unique, counts = np.unique(predictions, return_counts=True)
            class_dist = {str(u): int(c) for u, c in zip(unique, counts)}
            
            return {
                'n_simulations': len(predictions),
                'prediction_type': 'classification',
                'class_distribution': class_dist,
                'class_probabilities': {k: v/len(predictions) for k, v in class_dist.items()},
                'predictions': predictions.tolist(),
                'hypercube': hypercube
            }
        else:
            # Regression: return statistics
            return {
                'n_simulations': len(predictions),
                'prediction_type': 'regression',
                'statistics': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'median': float(np.median(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'p5': float(np.percentile(predictions, 5)),
                    'p25': float(np.percentile(predictions, 25)),
                    'p75': float(np.percentile(predictions, 75)),
                    'p95': float(np.percentile(predictions, 95)),
                    'iqr': float(np.percentile(predictions, 75) - np.percentile(predictions, 25)),
                    'cv': float(np.std(predictions) / np.mean(predictions)) if np.mean(predictions) != 0 else np.nan
                },
                'predictions': predictions.tolist(),
                'hypercube': hypercube
            }
    
    def get_tree_rules(self) -> str:
        """
        Get text representation of decision tree rules.
        
        Returns:
            String with tree rules in text format
        """
        if self.model is None:
            return "No model fitted"
        
        return export_text(self.model, feature_names=self.feature_names)
    
    def plot_tree(self, figsize: Tuple[int, int] = (20, 10), **kwargs) -> plt.Figure:
        """
        Plot the decision tree.
        
        Args:
            figsize: Figure size
            **kwargs: Additional arguments for sklearn.tree.plot_tree
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            ax=ax,
            **kwargs
        )
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, top_n: int = None) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            top_n: Show only top N features (default: all)
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        if top_n:
            indices = indices[:top_n]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.3)))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('CART Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_monte_carlo_distribution(
        self, 
        mc_results: Dict[str, Any],
        bins: int = 50
    ) -> plt.Figure:
        """
        Plot Monte Carlo prediction distribution.
        
        Args:
            mc_results: Results from monte_carlo_predictions
            bins: Number of histogram bins
            
        Returns:
            Matplotlib figure
        """
        if 'predictions' not in mc_results:
            return None
        
        predictions = mc_results['predictions']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if mc_results.get('prediction_type') == 'classification':
            # Bar chart for classes
            classes = list(mc_results['class_distribution'].keys())
            counts = list(mc_results['class_distribution'].values())
            ax.bar(classes, counts, color='steelblue', edgecolor='black')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'Monte Carlo Class Distribution (n={mc_results["n_simulations"]})')
        else:
            # Histogram for regression
            ax.hist(predictions, bins=bins, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='black')
            
            # Add statistics
            stats = mc_results['statistics']
            ax.axvline(stats['mean'], color='red', linestyle='--', 
                      label=f"Mean: {stats['mean']:.2f}")
            ax.axvline(stats['median'], color='green', linestyle=':', 
                      label=f"Median: {stats['median']:.2f}")
            ax.axvline(stats['p5'], color='orange', linestyle='--', alpha=0.5,
                      label=f"5th/95th: [{stats['p5']:.2f}, {stats['p95']:.2f}]")
            ax.axvline(stats['p95'], color='orange', linestyle='--', alpha=0.5)
            
            ax.set_xlabel(self.target_name or 'Prediction')
            ax.set_ylabel('Density')
            ax.set_title(f'Monte Carlo Prediction Distribution (n={mc_results["n_simulations"]})')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def sensitivity_to_cart_workflow(
    df: pd.DataFrame,
    output_col: str,
    input_cols: List[str],
    top_n_params: int = 25,
    num_trajectories: int = 20,
    cart_max_depth: int = 5,
    mc_samples: int = 1000
) -> Dict[str, Any]:
    """
    Complete Morris → CART → Monte Carlo workflow.
    
    This function implements the workflow described in environmental modeling papers
    for sensitivity analysis and uncertainty quantification.
    
    Args:
        df: Input DataFrame
        output_col: Name of output/target column
        input_cols: List of input parameter column names
        top_n_params: Number of top parameters to select from Morris
        num_trajectories: Number of Morris screening trajectories
        cart_max_depth: Maximum depth for CART tree
        mc_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary containing:
        - morris_results: Full Morris screening results
        - selected_parameters: Top parameters selected
        - cart_results: CART model results
        - monte_carlo_results: MC prediction statistics
        - workflow_summary: Summary of the analysis
        
    Example:
        >>> results = sensitivity_to_cart_workflow(
        ...     df, 'biomass', input_cols, 
        ...     top_n_params=25, mc_samples=1000
        ... )
        >>> print(f"Top parameter: {results['selected_parameters'][0]}")
        >>> print(f"Prediction uncertainty: {results['monte_carlo_results']['statistics']['std']:.2f}")
    """
    from .sensitivity_analysis import analyze_dataframe_sensitivity
    
    # Step 1: Morris Screening using the convenience function
    morris = analyze_dataframe_sensitivity(
        df,
        target=output_col,
        features=input_cols,
        method='morris',
        n_samples=num_trajectories,
        seed=42
    )
    
    if 'error' in morris:
        return {'error': f"Morris screening failed: {morris['error']}"}
    
    # Step 2: Select top parameters
    # Morris returns 'ranking' as a list of parameter names sorted by mu_star
    ranking = morris.get('ranking', input_cols[:top_n_params])
    selected = ranking[:top_n_params]
    
    # Step 3: CART Analysis
    cart = CARTAnalysis(df)
    cart_results = cart.cart_regression(
        selected, output_col, 
        max_depth=cart_max_depth
    )
    
    if 'error' in cart_results:
        return {'error': f"CART failed: {cart_results['error']}"}
    
    # Step 4: Monte Carlo Simulation
    hypercube = cart.generate_hypercube(selected, n_samples=mc_samples)
    mc_results = cart.monte_carlo_predictions(hypercube)
    
    return {
        'morris_results': morris,
        'selected_parameters': selected,
        'cart_results': cart_results,
        'monte_carlo_results': mc_results,
        'workflow_summary': {
            'total_input_parameters': len(input_cols),
            'parameters_selected': len(selected),
            'cart_r2': cart_results['metrics']['test_r2'],
            'cart_depth': cart_results['tree_depth'],
            'mc_mean': mc_results['statistics']['mean'],
            'mc_std': mc_results['statistics']['std'],
            'mc_95_ci': (mc_results['statistics']['p5'], mc_results['statistics']['p95'])
        }
    }
