"""
Outlier Removal Plugin
Remove or mark outliers using various methods
"""

import numpy as np
from scipy import stats

PLUGIN_INFO = {
    'name': 'Outlier Removal',
    'description': 'Remove or mark outliers using IQR or Z-score method',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'preprocessing',
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],
}

PLUGIN_PARAMETERS = {
    'method': {
        'type': 'str',
        'default': 'iqr',
        'choices': ['iqr', 'zscore'],
        'description': 'Outlier detection method'
    },
    'threshold': {
        'type': 'float',
        'default': 1.5,
        'min': 0.5,
        'max': 5.0,
        'description': 'IQR multiplier or Z-score threshold'
    },
    'action': {
        'type': 'str',
        'default': 'remove',
        'choices': ['remove', 'mark', 'cap'],
        'description': 'Action: remove rows, mark with flag, or cap values'
    },
}

def process(data, columns=None, target=None, **params):
    """Remove or handle outliers in the data."""
    if columns is None or len(columns) == 0:
        return data, "⚠️ No columns selected for outlier detection"
    
    result = data.copy()
    method = params.get('method', 'iqr')
    threshold = params.get('threshold', 1.5)
    action = params.get('action', 'remove')
    
    # Track outliers
    outlier_mask = pd.Series(False, index=result.index)
    stats_report = []
    
    for col in columns:
        if col not in result.columns:
            continue
        
        col_data = result[col].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            col_outliers = (result[col] < lower) | (result[col] > upper)
        else:  # zscore
            z_scores = np.abs(stats.zscore(col_data))
            z_series = pd.Series(index=col_data.index, data=z_scores)
            col_outliers = result.index.isin(z_series[z_series > threshold].index)
        
        n_outliers = col_outliers.sum()
        pct = (n_outliers / len(col_data)) * 100
        stats_report.append(f"  {col}: {n_outliers} outliers ({pct:.1f}%)")
        
        if action == 'cap':
            # Cap values at bounds
            if method == 'iqr':
                result.loc[result[col] < lower, col] = lower
                result.loc[result[col] > upper, col] = upper
        
        outlier_mask = outlier_mask | col_outliers
    
    # Apply action
    n_total_outliers = outlier_mask.sum()
    
    if action == 'remove':
        result = result[~outlier_mask]
        action_msg = f"Removed {n_total_outliers} rows"
    elif action == 'mark':
        result['is_outlier'] = outlier_mask.astype(int)
        action_msg = f"Marked {n_total_outliers} rows as outliers"
    else:  # cap
        action_msg = f"Capped outlier values in {len(columns)} columns"
    
    message = (f"✅ {action_msg}\n"
               f"Method: {method.upper()} (threshold={threshold})\n"
               f"Outliers by column:\n" + "\n".join(stats_report))
    
    return result, message


# Need pandas import for the plugin
import pandas as pd
