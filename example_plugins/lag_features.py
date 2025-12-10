"""
Lag Features Plugin
Create lagged versions of columns for time series analysis
"""

PLUGIN_INFO = {
    'name': 'Lag Features Generator',
    'description': 'Create lagged versions of columns (useful for time series)',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'preprocessing',
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],
}

PLUGIN_PARAMETERS = {
    'max_lag': {
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 50,
        'description': 'Maximum lag to create'
    },
    'include_diff': {
        'type': 'bool',
        'default': False,
        'description': 'Include difference features (change from previous)'
    },
    'include_pct_change': {
        'type': 'bool',
        'default': False,
        'description': 'Include percent change features'
    },
}

def process(data, columns=None, target=None, **params):
    """Create lag features for selected columns."""
    if columns is None or len(columns) == 0:
        return data, "⚠️ No columns selected"
    
    result = data.copy()
    max_lag = params.get('max_lag', 3)
    include_diff = params.get('include_diff', False)
    include_pct_change = params.get('include_pct_change', False)
    
    features_added = 0
    
    for col in columns:
        if col in result.columns:
            # Create lag features
            for lag in range(1, max_lag + 1):
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                features_added += 1
            
            # Optionally add difference features
            if include_diff:
                for lag in range(1, max_lag + 1):
                    result[f'{col}_diff_{lag}'] = result[col].diff(lag)
                    features_added += 1
            
            # Optionally add percent change features
            if include_pct_change:
                for lag in range(1, max_lag + 1):
                    result[f'{col}_pct_change_{lag}'] = result[col].pct_change(lag)
                    features_added += 1
    
    return result, f"✅ Added {features_added} lag features for {len(columns)} column(s)"
