# Example Plugins

This directory contains example plugins for the Advanced Data Analysis Toolkit.

## Available Plugins

### 🔧 Preprocessing

| Plugin | Description |
|--------|-------------|
| `lag_features.py` | Create lagged versions of columns for time series analysis |
| `outlier_removal.py` | Remove, mark, or cap outliers using IQR or Z-score |

### 📈 Visualization

| Plugin | Description |
|--------|-------------|
| `enhanced_scatter.py` | Scatter plots with polynomial regression and confidence intervals |

## How to Use

### Load from GUI

1. Go to the **🔌 Plugins** tab
2. Click **📁 Load File**
3. Select a plugin file
4. Configure parameters
5. Click **▶️ Run Plugin**

### Load Programmatically

```python
from data_toolkit import PluginManager

manager = PluginManager()

# Load from file
plugin, msg = manager.load_from_file('example_plugins/lag_features.py')
print(msg)

# Execute
result, msg = manager.execute_plugin(
    plugin.id,
    data=my_dataframe,
    columns=['col1', 'col2'],
    max_lag=5
)
print(msg)
```

## Creating Your Own Plugins

See the template in the GUI or use this structure:

```python
"""
My Custom Plugin
"""

PLUGIN_INFO = {
    'name': 'My Plugin',
    'description': 'What it does',
    'author': 'Your Name',
    'version': '1.0',
    'category': 'analysis',  # preprocessing, analysis, postprocessing, visualization
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],  # or 'dict', 'figure'
}

PLUGIN_PARAMETERS = {
    'param1': {
        'type': 'float',  # float, int, str, bool
        'default': 1.0,
        'description': 'Parameter description'
    },
}

def process(data, columns=None, target=None, **params):
    """
    Args:
        data: pandas DataFrame
        columns: list of selected feature columns
        target: target column name (or None)
        **params: parameter values from PLUGIN_PARAMETERS
    
    Returns:
        result: output (DataFrame, dict, or Figure)
        message: status message
    """
    result = data.copy()
    # Your processing logic here
    return result, "✅ Done!"
```

## Plugin Categories

- **preprocessing** - Data cleaning, feature engineering
- **analysis** - Statistical analysis, custom metrics
- **postprocessing** - Result transformation
- **visualization** - Custom plots and charts
