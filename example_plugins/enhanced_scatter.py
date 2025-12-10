"""
Custom Scatter Plot with Regression
Enhanced scatter plot with multiple regression options
"""

import numpy as np

PLUGIN_INFO = {
    'name': 'Enhanced Scatter Plot',
    'description': 'Create scatter plots with linear/polynomial regression',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'visualization',
    'inputs': ['dataframe'],
    'outputs': ['figure'],
}

PLUGIN_PARAMETERS = {
    'poly_degree': {
        'type': 'int',
        'default': 1,
        'min': 1,
        'max': 5,
        'description': 'Polynomial degree (1=linear)'
    },
    'show_ci': {
        'type': 'bool',
        'default': True,
        'description': 'Show confidence interval'
    },
    'point_size': {
        'type': 'int',
        'default': 50,
        'min': 10,
        'max': 200,
        'description': 'Scatter point size'
    },
    'alpha': {
        'type': 'float',
        'default': 0.6,
        'min': 0.1,
        'max': 1.0,
        'description': 'Point transparency'
    },
}

def process(data, columns=None, target=None, **params):
    """Create enhanced scatter plot."""
    if target is None:
        return None, "⚠️ Please select a target column (Y-axis)"
    if columns is None or len(columns) == 0:
        return None, "⚠️ Please select a feature column (X-axis)"
    
    x_col = columns[0]  # Use first selected column as X
    
    if x_col not in data.columns or target not in data.columns:
        return None, "⚠️ Selected columns not found in data"
    
    # Get data
    valid_idx = data[[x_col, target]].dropna().index
    x = data.loc[valid_idx, x_col].values
    y = data.loc[valid_idx, target].values
    
    if len(x) < 3:
        return None, "⚠️ Not enough valid data points"
    
    # Get parameters
    poly_degree = params.get('poly_degree', 1)
    show_ci = params.get('show_ci', True)
    point_size = params.get('point_size', 50)
    alpha = params.get('alpha', 0.6)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(x, y, s=point_size, alpha=alpha, c='#2563eb', 
                        edgecolors='white', linewidth=0.5)
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, poly_degree)
    poly = np.poly1d(coeffs)
    
    # Create smooth line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = poly(x_line)
    
    # Plot regression line
    label = f'Degree {poly_degree} fit'
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=label)
    
    # Calculate R²
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Confidence interval (approximate)
    if show_ci:
        n = len(x)
        residuals = y - y_pred
        std_err = np.sqrt(np.sum(residuals**2) / (n - poly_degree - 1))
        
        # Simple CI approximation
        ci = 1.96 * std_err
        ax.fill_between(x_line, y_line - ci, y_line + ci, 
                       alpha=0.2, color='red', label='95% CI')
    
    # Labels and title
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(target, fontsize=12)
    ax.set_title(f'{x_col} vs {target}\nR² = {r_squared:.4f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    message = f"✅ Scatter plot created\nR² = {r_squared:.4f}"
    if poly_degree > 1:
        message += f"\nPolynomial coefficients: {[round(c, 4) for c in coeffs]}"
    
    return fig, message
