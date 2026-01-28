"""
Sensitivity Analysis Module
Contains methods for global sensitivity analysis including Morris Screening,
Sobol indices, and elementary effects methods.

These methods help understand how input parameter variations affect model outputs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalysis:
    """
    Global sensitivity analysis methods for understanding parameter importance.
    
    Implements:
    - Morris Screening (Elementary Effects Method)
    - Sobol Sensitivity Indices (variance-based)
    - One-At-a-Time (OAT) Analysis
    
    These methods help identify which input parameters have the most influence
    on model outputs, which is essential for:
    - Model simplification
    - Uncertainty analysis
    - Experimental design
    - Factor prioritization
    """
    
    def __init__(self, model_func: Callable = None):
        """
        Initialize sensitivity analysis.
        
        Args:
            model_func: A callable function f(X) -> y where X is (n_samples, n_params)
                       and y is (n_samples,) or scalar
        """
        self.model_func = model_func
        self.results = {}
    
    def set_model(self, model_func: Callable):
        """Set the model function to analyze."""
        self.model_func = model_func
    
    def morris_screening(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Morris Screening Method (Elementary Effects).
        
        This is a one-at-a-time (OAT) screening method that efficiently identifies
        which input factors have:
        - Negligible effects
        - Linear effects (without interactions)
        - Nonlinear effects or interactions
        
        The method computes Elementary Effects (EE) by perturbing one factor at a time
        and measuring the change in output. Statistics computed:
        - μ (mu): Mean of EEs - indicates overall influence
        - μ* (mu_star): Mean of |EE| - indicates influence regardless of sign
        - σ (sigma): Std of EEs - indicates nonlinearity or interactions
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) bounds
            n_trajectories: Number of Morris trajectories (typically 10-50)
            n_levels: Number of levels in the parameter grid (typically 4-10)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with:
            - 'mu': Mean elementary effects
            - 'mu_star': Mean absolute elementary effects  
            - 'sigma': Std of elementary effects
            - 'elementary_effects': Raw EE values per trajectory
            - 'ranking': Parameters ranked by importance (mu_star)
            - 'classification': Classification as negligible/linear/nonlinear
            
        Reference:
            Morris, M.D. (1991). Factorial Sampling Plans for Preliminary
            Computational Experiments. Technometrics, 33(2), 161-174.
        """
        if self.model_func is None:
            return {'error': 'No model function set. Use set_model() first.'}
        
        if seed is not None:
            np.random.seed(seed)
        
        param_names = list(param_bounds.keys())
        k = len(param_names)  # Number of parameters
        
        # Generate Morris trajectories
        delta = n_levels / (2 * (n_levels - 1))  # Step size
        
        # Store elementary effects for each parameter
        elementary_effects = {name: [] for name in param_names}
        
        for _ in range(n_trajectories):
            # Generate base point (random levels)
            base = np.random.randint(0, n_levels - 1, k) / (n_levels - 1)
            
            # Random order for parameter perturbation
            order = np.random.permutation(k)
            
            # Build trajectory
            trajectory = [base.copy()]
            for i in order:
                new_point = trajectory[-1].copy()
                # Perturb parameter i
                if np.random.random() < 0.5:
                    new_point[i] = min(1.0, new_point[i] + delta)
                else:
                    new_point[i] = max(0.0, new_point[i] - delta)
                trajectory.append(new_point)
            
            trajectory = np.array(trajectory)
            
            # Scale trajectory to actual parameter bounds
            scaled_trajectory = np.zeros_like(trajectory)
            for j, name in enumerate(param_names):
                lo, hi = param_bounds[name]
                scaled_trajectory[:, j] = lo + trajectory[:, j] * (hi - lo)
            
            # Evaluate model at all trajectory points
            outputs = np.array([self.model_func(x) for x in scaled_trajectory])
            
            # Compute elementary effects
            for step, i in enumerate(order):
                name = param_names[i]
                # Actual delta in scaled space
                actual_delta = trajectory[step + 1, i] - trajectory[step, i]
                if abs(actual_delta) > 1e-10:
                    lo, hi = param_bounds[name]
                    scaled_delta = actual_delta * (hi - lo)
                    ee = (outputs[step + 1] - outputs[step]) / scaled_delta * (hi - lo)
                    elementary_effects[name].append(ee)
        
        # Compute statistics
        results = {
            'mu': {},
            'mu_star': {},
            'sigma': {},
            'elementary_effects': elementary_effects,
            'param_names': param_names,
            'n_trajectories': n_trajectories,
            'n_levels': n_levels
        }
        
        for name in param_names:
            ees = np.array(elementary_effects[name])
            if len(ees) > 0:
                results['mu'][name] = float(np.mean(ees))
                results['mu_star'][name] = float(np.mean(np.abs(ees)))
                results['sigma'][name] = float(np.std(ees))
            else:
                results['mu'][name] = 0.0
                results['mu_star'][name] = 0.0
                results['sigma'][name] = 0.0
        
        # Rank parameters by mu_star
        ranking = sorted(param_names, key=lambda x: results['mu_star'][x], reverse=True)
        results['ranking'] = ranking
        
        # Classify parameters
        classification = {}
        max_mu_star = max(results['mu_star'].values()) if results['mu_star'] else 1.0
        for name in param_names:
            mu_star = results['mu_star'][name]
            sigma = results['sigma'][name]
            
            # Classification rules (commonly used thresholds)
            if mu_star < 0.1 * max_mu_star:
                classification[name] = 'negligible'
            elif sigma < 0.5 * mu_star:
                classification[name] = 'linear'
            else:
                classification[name] = 'nonlinear/interaction'
        
        results['classification'] = classification
        
        self.results['morris'] = results
        return results
    
    def sobol_indices(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 1000,
        seed: int = None,
        calc_second_order: bool = False
    ) -> Dict[str, Any]:
        """
        Sobol Sensitivity Indices (Variance-based).
        
        Decomposes output variance into contributions from individual parameters
        and their interactions. Computes:
        - S1 (First-order): Direct contribution of each parameter
        - ST (Total-order): Total contribution including all interactions
        - S2 (Second-order): Pairwise interactions (if calc_second_order=True)
        
        Uses Saltelli's extension of the Sobol sequence for efficient sampling.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) bounds
            n_samples: Base sample size (total evaluations = n_samples * (2k + 2))
            seed: Random seed for reproducibility
            calc_second_order: Whether to calculate second-order indices
            
        Returns:
            Dictionary with:
            - 'S1': First-order indices
            - 'ST': Total-order indices
            - 'S2': Second-order indices (if requested)
            - 'S1_conf': Confidence intervals for S1
            - 'ST_conf': Confidence intervals for ST
            
        Reference:
            Saltelli, A. et al. (2010). Variance based sensitivity analysis of
            model output. Design and estimator for the total sensitivity index.
            Computer Physics Communications, 181(2), 259-270.
        """
        if self.model_func is None:
            return {'error': 'No model function set. Use set_model() first.'}
        
        if seed is not None:
            np.random.seed(seed)
        
        param_names = list(param_bounds.keys())
        k = len(param_names)
        
        # Generate base samples A and B (quasi-random would be better)
        A = np.random.random((n_samples, k))
        B = np.random.random((n_samples, k))
        
        # Scale to parameter bounds
        def scale(X):
            X_scaled = np.zeros_like(X)
            for j, name in enumerate(param_names):
                lo, hi = param_bounds[name]
                X_scaled[:, j] = lo + X[:, j] * (hi - lo)
            return X_scaled
        
        A_scaled = scale(A)
        B_scaled = scale(B)
        
        # Evaluate base matrices
        f_A = np.array([self.model_func(x) for x in A_scaled])
        f_B = np.array([self.model_func(x) for x in B_scaled])
        
        # Build AB_i matrices and evaluate
        f_AB = {}
        f_BA = {}
        for i, name in enumerate(param_names):
            # AB_i: A with i-th column from B
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            f_AB[name] = np.array([self.model_func(x) for x in scale(AB_i)])
            
            # BA_i: B with i-th column from A (for total-order)
            BA_i = B.copy()
            BA_i[:, i] = A[:, i]
            f_BA[name] = np.array([self.model_func(x) for x in scale(BA_i)])
        
        # Compute indices using Jansen estimator
        f_all = np.concatenate([f_A, f_B])
        var_total = np.var(f_all)
        
        if var_total < 1e-10:
            # Constant output - no sensitivity
            return {
                'S1': {name: 0.0 for name in param_names},
                'ST': {name: 0.0 for name in param_names},
                'S1_conf': {name: (0.0, 0.0) for name in param_names},
                'ST_conf': {name: (0.0, 0.0) for name in param_names},
                'warning': 'Model output has near-zero variance'
            }
        
        results = {
            'S1': {},
            'ST': {},
            'S1_conf': {},
            'ST_conf': {},
            'param_names': param_names,
            'n_samples': n_samples,
            'var_total': float(var_total)
        }
        
        for name in param_names:
            # First-order (Jansen estimator)
            V_i = np.mean(f_B * (f_AB[name] - f_A))
            S1 = V_i / var_total
            
            # Total-order
            VT_i = 0.5 * np.mean((f_A - f_AB[name])**2)
            ST = VT_i / var_total
            
            # Bootstrap confidence intervals
            n_boot = 100
            S1_boot = []
            ST_boot = []
            for _ in range(n_boot):
                idx = np.random.choice(n_samples, n_samples, replace=True)
                V_i_b = np.mean(f_B[idx] * (f_AB[name][idx] - f_A[idx]))
                VT_i_b = 0.5 * np.mean((f_A[idx] - f_AB[name][idx])**2)
                S1_boot.append(V_i_b / var_total)
                ST_boot.append(VT_i_b / var_total)
            
            S1_boot = np.array(S1_boot)
            ST_boot = np.array(ST_boot)
            
            results['S1'][name] = float(np.clip(S1, 0, 1))
            results['ST'][name] = float(np.clip(ST, 0, 1))
            results['S1_conf'][name] = (
                float(np.percentile(S1_boot, 2.5)),
                float(np.percentile(S1_boot, 97.5))
            )
            results['ST_conf'][name] = (
                float(np.percentile(ST_boot, 2.5)),
                float(np.percentile(ST_boot, 97.5))
            )
        
        # Second-order indices (if requested)
        if calc_second_order and k > 1:
            results['S2'] = {}
            for i in range(k):
                for j in range(i + 1, k):
                    name_i = param_names[i]
                    name_j = param_names[j]
                    
                    # V_ij estimator (simplified)
                    V_ij = (np.mean(f_AB[name_i] * f_AB[name_j]) - 
                           np.mean(f_A)**2 - 
                           results['S1'][name_i] * var_total -
                           results['S1'][name_j] * var_total)
                    S2_ij = V_ij / var_total
                    
                    key = f"{name_i}:{name_j}"
                    results['S2'][key] = float(np.clip(S2_ij, -1, 1))
        
        # Rank by total-order index
        results['ranking'] = sorted(param_names, key=lambda x: results['ST'][x], reverse=True)
        
        self.results['sobol'] = results
        return results
    
    def one_at_a_time(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        base_point: Dict[str, float] = None,
        n_steps: int = 20
    ) -> Dict[str, Any]:
        """
        One-At-a-Time (OAT) Local Sensitivity Analysis.
        
        Varies each parameter individually while holding others at base values.
        Simple but doesn't capture interactions.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) bounds
            base_point: Base parameter values (defaults to midpoints)
            n_steps: Number of steps for each parameter sweep
            
        Returns:
            Dictionary with sensitivity curves and local gradients
        """
        if self.model_func is None:
            return {'error': 'No model function set. Use set_model() first.'}
        
        param_names = list(param_bounds.keys())
        
        # Default base point is midpoint of bounds
        if base_point is None:
            base_point = {name: (lo + hi) / 2 for name, (lo, hi) in param_bounds.items()}
        
        base_array = np.array([base_point[name] for name in param_names])
        base_output = self.model_func(base_array)
        
        results = {
            'param_names': param_names,
            'base_point': base_point,
            'base_output': float(base_output),
            'sweeps': {},
            'gradients': {},
            'elasticities': {}
        }
        
        for i, name in enumerate(param_names):
            lo, hi = param_bounds[name]
            values = np.linspace(lo, hi, n_steps)
            outputs = []
            
            for val in values:
                x = base_array.copy()
                x[i] = val
                outputs.append(self.model_func(x))
            
            outputs = np.array(outputs)
            results['sweeps'][name] = {
                'values': values.tolist(),
                'outputs': outputs.tolist()
            }
            
            # Local gradient at base point
            if n_steps > 1:
                mid_idx = n_steps // 2
                dx = values[1] - values[0]
                if mid_idx > 0 and mid_idx < n_steps - 1:
                    gradient = (outputs[mid_idx + 1] - outputs[mid_idx - 1]) / (2 * dx)
                else:
                    gradient = (outputs[-1] - outputs[0]) / (values[-1] - values[0])
                
                results['gradients'][name] = float(gradient)
                
                # Elasticity (% change in output / % change in input)
                if abs(base_point[name]) > 1e-10 and abs(base_output) > 1e-10:
                    elasticity = gradient * base_point[name] / base_output
                else:
                    elasticity = 0.0
                results['elasticities'][name] = float(elasticity)
        
        # Rank by absolute gradient
        results['ranking'] = sorted(
            param_names, 
            key=lambda x: abs(results['gradients'].get(x, 0)), 
            reverse=True
        )
        
        self.results['oat'] = results
        return results
    
    def plot_morris(self, results: Dict[str, Any] = None) -> plt.Figure:
        """
        Create Morris screening plot (μ* vs σ).
        
        Points in lower-left are unimportant, upper-right have nonlinear/interaction effects.
        """
        if results is None:
            results = self.results.get('morris')
        if results is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        param_names = results['param_names']
        mu_star = [results['mu_star'][name] for name in param_names]
        sigma = [results['sigma'][name] for name in param_names]
        
        # Color by classification
        colors = []
        for name in param_names:
            cls = results['classification'].get(name, 'unknown')
            if cls == 'negligible':
                colors.append('gray')
            elif cls == 'linear':
                colors.append('blue')
            else:
                colors.append('red')
        
        scatter = ax.scatter(mu_star, sigma, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add parameter labels
        for i, name in enumerate(param_names):
            ax.annotate(name, (mu_star[i], sigma[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add diagonal line (σ = 0.5 * μ*) as rough boundary
        max_val = max(max(mu_star), max(sigma)) * 1.1
        ax.plot([0, max_val], [0, 0.5 * max_val], 'k--', alpha=0.3, label='σ = 0.5μ*')
        
        ax.set_xlabel('μ* (Mean of |Elementary Effects|)', fontsize=12)
        ax.set_ylabel('σ (Std of Elementary Effects)', fontsize=12)
        ax.set_title('Morris Screening Plot', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Negligible'),
            Patch(facecolor='blue', label='Linear'),
            Patch(facecolor='red', label='Nonlinear/Interaction')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        
        plt.tight_layout()
        return fig
    
    def plot_sobol(self, results: Dict[str, Any] = None) -> plt.Figure:
        """
        Create Sobol indices bar plot comparing S1 and ST.
        
        Large gap between S1 and ST indicates strong interaction effects.
        """
        if results is None:
            results = self.results.get('sobol')
        if results is None:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        param_names = results['param_names']
        n = len(param_names)
        
        S1 = [results['S1'][name] for name in param_names]
        ST = [results['ST'][name] for name in param_names]
        
        x = np.arange(n)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, S1, width, label='First-order (S1)', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, ST, width, label='Total-order (ST)', color='darkorange', alpha=0.8)
        
        # Add confidence intervals if available
        if 'S1_conf' in results:
            S1_err = [[S1[i] - results['S1_conf'][name][0] for i, name in enumerate(param_names)],
                      [results['S1_conf'][name][1] - S1[i] for i, name in enumerate(param_names)]]
            ax.errorbar(x - width/2, S1, yerr=S1_err, fmt='none', color='black', capsize=3)
        
        if 'ST_conf' in results:
            ST_err = [[ST[i] - results['ST_conf'][name][0] for i, name in enumerate(param_names)],
                      [results['ST_conf'][name][1] - ST[i] for i, name in enumerate(param_names)]]
            ax.errorbar(x + width/2, ST, yerr=ST_err, fmt='none', color='black', capsize=3)
        
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Sensitivity Index', fontsize=12)
        ax.set_title('Sobol Sensitivity Indices', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_oat(self, results: Dict[str, Any] = None) -> plt.Figure:
        """
        Create OAT sensitivity sweep plots.
        """
        if results is None:
            results = self.results.get('oat')
        if results is None:
            return None
        
        param_names = results['param_names']
        n = len(param_names)
        
        # Determine subplot layout
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, name in enumerate(param_names):
            ax = axes[i]
            sweep = results['sweeps'][name]
            ax.plot(sweep['values'], sweep['outputs'], 'b-', linewidth=2)
            ax.axhline(results['base_output'], color='gray', linestyle='--', alpha=0.5, label='Base')
            ax.axvline(results['base_point'][name], color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(name, fontsize=10)
            ax.set_ylabel('Output', fontsize=10)
            ax.set_title(f'{name}\n(gradient: {results["gradients"].get(name, 0):.3f})', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('One-At-a-Time Sensitivity Sweeps', fontsize=14)
        plt.tight_layout()
        return fig
    
    def summary_table(self) -> pd.DataFrame:
        """
        Create summary table of all sensitivity analysis results.
        """
        data = []
        
        # Morris results
        if 'morris' in self.results:
            morris = self.results['morris']
            for name in morris['param_names']:
                data.append({
                    'Parameter': name,
                    'Morris μ*': morris['mu_star'].get(name, np.nan),
                    'Morris σ': morris['sigma'].get(name, np.nan),
                    'Morris Class': morris['classification'].get(name, ''),
                })
        
        # Sobol results
        if 'sobol' in self.results:
            sobol = self.results['sobol']
            for i, name in enumerate(sobol['param_names']):
                if i < len(data):
                    data[i]['Sobol S1'] = sobol['S1'].get(name, np.nan)
                    data[i]['Sobol ST'] = sobol['ST'].get(name, np.nan)
                else:
                    data.append({
                        'Parameter': name,
                        'Sobol S1': sobol['S1'].get(name, np.nan),
                        'Sobol ST': sobol['ST'].get(name, np.nan)
                    })
        
        # OAT results
        if 'oat' in self.results:
            oat = self.results['oat']
            for i, name in enumerate(oat['param_names']):
                if i < len(data):
                    data[i]['OAT Gradient'] = oat['gradients'].get(name, np.nan)
                    data[i]['OAT Elasticity'] = oat['elasticities'].get(name, np.nan)
                else:
                    data.append({
                        'Parameter': name,
                        'OAT Gradient': oat['gradients'].get(name, np.nan),
                        'OAT Elasticity': oat['elasticities'].get(name, np.nan)
                    })
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)


def analyze_dataframe_sensitivity(
    df: pd.DataFrame,
    target: str,
    features: List[str] = None,
    method: str = 'morris',
    n_samples: int = 100,
    seed: int = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze sensitivity of a regression model on a DataFrame.
    
    Fits a linear regression model and analyzes sensitivity of predictions
    to input feature variations.
    
    Args:
        df: Input DataFrame
        target: Target column name
        features: List of feature columns (default: all numeric except target)
        method: 'morris', 'sobol', or 'oat'
        n_samples: Number of samples for analysis
        seed: Random seed
        
    Returns:
        Sensitivity analysis results
    """
    from sklearn.linear_model import LinearRegression
    
    if features is None:
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create model function
    def model_func(x):
        return model.predict(x.reshape(1, -1))[0]
    
    # Get parameter bounds from data
    param_bounds = {
        name: (float(X[name].min()), float(X[name].max()))
        for name in features
    }
    
    # Run analysis
    sa = SensitivityAnalysis(model_func)
    
    if method == 'morris':
        results = sa.morris_screening(param_bounds, n_trajectories=n_samples, seed=seed)
    elif method == 'sobol':
        results = sa.sobol_indices(param_bounds, n_samples=n_samples, seed=seed)
    elif method == 'oat':
        results = sa.one_at_a_time(param_bounds, n_steps=n_samples)
    else:
        return {'error': f'Unknown method: {method}'}
    
    results['model_coefficients'] = dict(zip(features, model.coef_))
    results['model_intercept'] = model.intercept_
    
    return results
