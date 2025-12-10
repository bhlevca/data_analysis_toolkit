"""
Plugin System for Advanced Data Analysis Toolkit
================================================

Allows users to extend the toolkit with custom processing functions.

Features:
- Load plugins from .py files
- Paste plugin code directly
- Configure plugin parameters via GUI
- Chain plugins in processing pipelines
- Sandboxed execution for safety

Plugin Template:
---------------
```python
PLUGIN_INFO = {
    'name': 'My Custom Plugin',
    'description': 'What this plugin does',
    'author': 'Your Name',
    'version': '1.0',
    'category': 'analysis',  # preprocessing, analysis, postprocessing, visualization
    'inputs': ['dataframe'],  # dataframe, series, array, dict
    'outputs': ['dataframe'], # dataframe, series, array, dict, figure
}

PLUGIN_PARAMETERS = {
    'param1': {'type': 'float', 'default': 0.5, 'min': 0, 'max': 1, 'description': 'A parameter'},
    'param2': {'type': 'int', 'default': 10, 'description': 'Another parameter'},
    'param3': {'type': 'str', 'default': 'option1', 'choices': ['option1', 'option2'], 'description': 'Choose one'},
    'param4': {'type': 'bool', 'default': True, 'description': 'Enable feature'},
}

def process(data, columns=None, target=None, **params):
    '''
    Main processing function.
    
    Args:
        data: Input data (pandas DataFrame)
        columns: Selected feature columns (list of str)
        target: Target column name (str or None)
        **params: Parameters from PLUGIN_PARAMETERS
        
    Returns:
        result: Processing result (type should match 'outputs' in PLUGIN_INFO)
        message: Status message to display (str)
    '''
    # Your processing code here
    result = data.copy()
    message = "Processing complete!"
    return result, message
```
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import hashlib
import json
import os


# ============================================================================
# PLUGIN DATA STRUCTURES
# ============================================================================

@dataclass
class PluginInfo:
    """Information about a plugin"""
    name: str
    description: str
    author: str = "Unknown"
    version: str = "1.0"
    category: str = "analysis"  # preprocessing, analysis, postprocessing, visualization
    inputs: List[str] = field(default_factory=lambda: ["dataframe"])
    outputs: List[str] = field(default_factory=lambda: ["dataframe"])
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'version': self.version,
            'category': self.category,
            'inputs': self.inputs,
            'outputs': self.outputs,
        }


@dataclass
class PluginParameter:
    """A single plugin parameter"""
    name: str
    type: str  # float, int, str, bool
    default: Any
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[str]] = None
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """Validate a parameter value"""
        try:
            if self.type == 'float':
                value = float(value)
                if self.min is not None and value < self.min:
                    return False, f"{self.name} must be >= {self.min}"
                if self.max is not None and value > self.max:
                    return False, f"{self.name} must be <= {self.max}"
            elif self.type == 'int':
                value = int(value)
                if self.min is not None and value < self.min:
                    return False, f"{self.name} must be >= {self.min}"
                if self.max is not None and value > self.max:
                    return False, f"{self.name} must be <= {self.max}"
            elif self.type == 'str':
                if self.choices and value not in self.choices:
                    return False, f"{self.name} must be one of {self.choices}"
            elif self.type == 'bool':
                if not isinstance(value, bool):
                    value = str(value).lower() in ('true', '1', 'yes')
            return True, ""
        except Exception as e:
            return False, str(e)


@dataclass 
class Plugin:
    """A loaded plugin"""
    id: str  # Unique identifier (hash of code)
    info: PluginInfo
    parameters: Dict[str, PluginParameter]
    process_func: Callable
    source_code: str
    source_path: Optional[str] = None  # Path if loaded from file
    enabled: bool = True
    
    def execute(self, data: pd.DataFrame, columns: List[str] = None, 
                target: str = None, **params) -> Tuple[Any, str]:
        """Execute the plugin"""
        # Merge default parameters with provided ones
        final_params = {}
        for name, param in self.parameters.items():
            if name in params:
                final_params[name] = params[name]
            else:
                final_params[name] = param.default
        
        try:
            result, message = self.process_func(data, columns=columns, 
                                                target=target, **final_params)
            return result, message
        except Exception as e:
            return None, f"Error: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# PLUGIN MANAGER
# ============================================================================

class PluginManager:
    """Manages loading, storing, and executing plugins"""
    
    # Allowed imports for plugins (safety)
    ALLOWED_IMPORTS = {
        'pandas', 'pd',
        'numpy', 'np', 
        'scipy', 'stats',
        'sklearn',
        'matplotlib', 'plt',
        'seaborn', 'sns',
        'math',
        'statistics',
        'collections',
        'itertools',
        'functools',
    }
    
    def __init__(self, plugin_dir: str = None):
        """Initialize the plugin manager"""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dir = plugin_dir or os.path.expanduser("~/.data_toolkit/plugins")
        self._listeners: List[Callable] = []
        
        # Create plugin directory if it doesn't exist
        Path(self.plugin_dir).mkdir(parents=True, exist_ok=True)
        
        # Load saved plugins
        self._load_saved_plugins()
    
    def add_listener(self, callback: Callable) -> None:
        """Add a callback to be notified when plugins change"""
        self._listeners.append(callback)
    
    def _notify_listeners(self) -> None:
        """Notify all listeners of plugin changes"""
        for callback in self._listeners:
            try:
                callback()
            except:
                pass
    
    def _generate_id(self, code: str) -> str:
        """Generate a unique ID for a plugin based on its code"""
        return hashlib.md5(code.encode()).hexdigest()[:12]
    
    def _create_safe_globals(self) -> dict:
        """Create a restricted global namespace for plugin execution"""
        safe_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'enumerate': enumerate, 'filter': filter,
                'float': float, 'int': int, 'len': len, 'list': list,
                'map': map, 'max': max, 'min': min, 'print': print,
                'range': range, 'round': round, 'set': set, 'sorted': sorted,
                'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
                'zip': zip, 'isinstance': isinstance, 'hasattr': hasattr,
                'getattr': getattr, 'setattr': setattr,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'KeyError': KeyError,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
        }
        
        # Add scipy if available
        try:
            from scipy import stats as scipy_stats
            safe_globals['stats'] = scipy_stats
            safe_globals['scipy'] = __import__('scipy')
        except ImportError:
            pass
        
        # Add sklearn if available
        try:
            import sklearn
            safe_globals['sklearn'] = sklearn
        except ImportError:
            pass
        
        # Add seaborn if available
        try:
            import seaborn as sns
            safe_globals['sns'] = sns
        except ImportError:
            pass
        
        return safe_globals
    
    def load_from_code(self, code: str, source_path: str = None) -> Tuple[Optional[Plugin], str]:
        """
        Load a plugin from source code.
        
        Args:
            code: Python source code defining the plugin
            source_path: Optional path where the code came from
            
        Returns:
            Tuple of (Plugin or None, status message)
        """
        try:
            # Create safe execution environment
            plugin_globals = self._create_safe_globals()
            plugin_locals = {}
            
            # Execute the code
            exec(code, plugin_globals, plugin_locals)
            
            # Extract plugin info
            if 'PLUGIN_INFO' not in plugin_locals:
                return None, "Error: Plugin must define PLUGIN_INFO dict"
            
            info_dict = plugin_locals['PLUGIN_INFO']
            info = PluginInfo(
                name=info_dict.get('name', 'Unnamed Plugin'),
                description=info_dict.get('description', ''),
                author=info_dict.get('author', 'Unknown'),
                version=info_dict.get('version', '1.0'),
                category=info_dict.get('category', 'analysis'),
                inputs=info_dict.get('inputs', ['dataframe']),
                outputs=info_dict.get('outputs', ['dataframe']),
            )
            
            # Extract parameters
            params_dict = plugin_locals.get('PLUGIN_PARAMETERS', {})
            parameters = {}
            for name, pinfo in params_dict.items():
                parameters[name] = PluginParameter(
                    name=name,
                    type=pinfo.get('type', 'str'),
                    default=pinfo.get('default', None),
                    description=pinfo.get('description', ''),
                    min=pinfo.get('min'),
                    max=pinfo.get('max'),
                    choices=pinfo.get('choices'),
                )
            
            # Extract process function
            if 'process' not in plugin_locals:
                return None, "Error: Plugin must define process() function"
            
            process_func = plugin_locals['process']
            
            # Create plugin
            plugin_id = self._generate_id(code)
            plugin = Plugin(
                id=plugin_id,
                info=info,
                parameters=parameters,
                process_func=process_func,
                source_code=code,
                source_path=source_path,
            )
            
            # Register plugin
            self.plugins[plugin_id] = plugin
            self._notify_listeners()
            
            return plugin, f"✅ Plugin '{info.name}' loaded successfully!"
            
        except SyntaxError as e:
            return None, f"Syntax Error: {e}"
        except Exception as e:
            return None, f"Error loading plugin: {str(e)}\n{traceback.format_exc()}"
    
    def load_from_file(self, filepath: str) -> Tuple[Optional[Plugin], str]:
        """Load a plugin from a .py file"""
        try:
            path = Path(filepath)
            if not path.exists():
                return None, f"File not found: {filepath}"
            
            if path.suffix != '.py':
                return None, "Plugin file must be a .py file"
            
            code = path.read_text(encoding='utf-8')
            return self.load_from_code(code, source_path=str(path))
            
        except Exception as e:
            return None, f"Error reading file: {str(e)}"
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        if plugin_id in self.plugins:
            del self.plugins[plugin_id]
            self._notify_listeners()
            return True
        return False
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID"""
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_category(self, category: str) -> List[Plugin]:
        """Get all plugins in a category"""
        return [p for p in self.plugins.values() if p.info.category == category]
    
    def get_all_plugins(self) -> List[Plugin]:
        """Get all loaded plugins"""
        return list(self.plugins.values())
    
    def execute_plugin(self, plugin_id: str, data: pd.DataFrame,
                      columns: List[str] = None, target: str = None,
                      **params) -> Tuple[Any, str]:
        """Execute a plugin by ID"""
        plugin = self.plugins.get(plugin_id)
        if plugin is None:
            return None, f"Plugin not found: {plugin_id}"
        
        if not plugin.enabled:
            return None, f"Plugin '{plugin.info.name}' is disabled"
        
        return plugin.execute(data, columns, target, **params)
    
    def save_plugin(self, plugin_id: str) -> Tuple[bool, str]:
        """Save a plugin to the plugin directory"""
        plugin = self.plugins.get(plugin_id)
        if plugin is None:
            return False, "Plugin not found"
        
        try:
            filename = f"{plugin.info.name.lower().replace(' ', '_')}_{plugin_id}.py"
            filepath = Path(self.plugin_dir) / filename
            filepath.write_text(plugin.source_code, encoding='utf-8')
            return True, f"Saved to {filepath}"
        except Exception as e:
            return False, f"Error saving: {str(e)}"
    
    def _load_saved_plugins(self) -> None:
        """Load all plugins from the plugin directory"""
        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            return
        
        for filepath in plugin_path.glob("*.py"):
            try:
                self.load_from_file(str(filepath))
            except:
                pass  # Skip failed plugins
    
    def export_plugin(self, plugin_id: str, filepath: str) -> Tuple[bool, str]:
        """Export a plugin to a file"""
        plugin = self.plugins.get(plugin_id)
        if plugin is None:
            return False, "Plugin not found"
        
        try:
            Path(filepath).write_text(plugin.source_code, encoding='utf-8')
            return True, f"Exported to {filepath}"
        except Exception as e:
            return False, f"Error exporting: {str(e)}"


# ============================================================================
# EXAMPLE PLUGINS
# ============================================================================

EXAMPLE_PLUGIN_ZSCORE_NORMALIZE = '''"""
Z-Score Normalization Plugin
Normalizes selected columns to have mean=0 and std=1
"""

PLUGIN_INFO = {
    'name': 'Z-Score Normalization',
    'description': 'Normalize columns using Z-score (mean=0, std=1)',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'preprocessing',
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],
}

PLUGIN_PARAMETERS = {
    'replace_original': {
        'type': 'bool',
        'default': False,
        'description': 'Replace original columns (vs. create new ones)'
    },
}

def process(data, columns=None, target=None, **params):
    if columns is None or len(columns) == 0:
        return data, "⚠️ No columns selected"
    
    result = data.copy()
    replace = params.get('replace_original', False)
    
    for col in columns:
        if col in result.columns:
            mean_val = result[col].mean()
            std_val = result[col].std()
            
            if std_val == 0:
                normalized = result[col] - mean_val
            else:
                normalized = (result[col] - mean_val) / std_val
            
            if replace:
                result[col] = normalized
            else:
                result[f'{col}_zscore'] = normalized
    
    return result, f"✅ Normalized {len(columns)} column(s)"
'''

EXAMPLE_PLUGIN_CUSTOM_METRIC = '''"""
Custom Correlation Metric Plugin
Calculate a custom weighted correlation metric
"""

PLUGIN_INFO = {
    'name': 'Custom Weighted Correlation',
    'description': 'Calculate correlation with custom weighting',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'analysis',
    'inputs': ['dataframe'],
    'outputs': ['dict'],
}

PLUGIN_PARAMETERS = {
    'weight_recent': {
        'type': 'float',
        'default': 0.7,
        'min': 0.0,
        'max': 1.0,
        'description': 'Weight for recent data (0-1)'
    },
    'split_point': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 0.9,
        'description': 'Where to split data (fraction)'
    },
}

def process(data, columns=None, target=None, **params):
    if target is None:
        return None, "⚠️ Please select a target column"
    if columns is None or len(columns) == 0:
        return None, "⚠️ No feature columns selected"
    
    weight_recent = params.get('weight_recent', 0.7)
    split_point = params.get('split_point', 0.5)
    
    results = {}
    n = len(data)
    split_idx = int(n * split_point)
    
    for col in columns:
        if col in data.columns and col != target:
            # Calculate correlation for early and recent data
            early_corr = data[[col, target]].iloc[:split_idx].corr().iloc[0, 1]
            recent_corr = data[[col, target]].iloc[split_idx:].corr().iloc[0, 1]
            
            # Weighted combination
            weighted_corr = (1 - weight_recent) * early_corr + weight_recent * recent_corr
            
            results[col] = {
                'early_correlation': round(early_corr, 4),
                'recent_correlation': round(recent_corr, 4),
                'weighted_correlation': round(weighted_corr, 4),
            }
    
    message = f"✅ Calculated weighted correlations for {len(results)} features"
    return results, message
'''

EXAMPLE_PLUGIN_ROLLING_FEATURES = '''"""
Rolling Features Plugin
Create rolling window features for time series
"""

PLUGIN_INFO = {
    'name': 'Rolling Window Features',
    'description': 'Generate rolling mean, std, min, max features',
    'author': 'Data Toolkit',
    'version': '1.0',
    'category': 'preprocessing',
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],
}

PLUGIN_PARAMETERS = {
    'window_size': {
        'type': 'int',
        'default': 10,
        'min': 2,
        'max': 100,
        'description': 'Rolling window size'
    },
    'include_mean': {'type': 'bool', 'default': True, 'description': 'Include rolling mean'},
    'include_std': {'type': 'bool', 'default': True, 'description': 'Include rolling std'},
    'include_min': {'type': 'bool', 'default': False, 'description': 'Include rolling min'},
    'include_max': {'type': 'bool', 'default': False, 'description': 'Include rolling max'},
}

def process(data, columns=None, target=None, **params):
    if columns is None or len(columns) == 0:
        return data, "⚠️ No columns selected"
    
    result = data.copy()
    window = params.get('window_size', 10)
    
    features_added = 0
    for col in columns:
        if col in result.columns:
            if params.get('include_mean', True):
                result[f'{col}_roll_mean_{window}'] = result[col].rolling(window).mean()
                features_added += 1
            if params.get('include_std', True):
                result[f'{col}_roll_std_{window}'] = result[col].rolling(window).std()
                features_added += 1
            if params.get('include_min', False):
                result[f'{col}_roll_min_{window}'] = result[col].rolling(window).min()
                features_added += 1
            if params.get('include_max', False):
                result[f'{col}_roll_max_{window}'] = result[col].rolling(window).max()
                features_added += 1
    
    return result, f"✅ Added {features_added} rolling features"
'''


def get_example_plugins() -> Dict[str, str]:
    """Get all example plugin templates"""
    return {
        'Z-Score Normalization': EXAMPLE_PLUGIN_ZSCORE_NORMALIZE,
        'Custom Weighted Correlation': EXAMPLE_PLUGIN_CUSTOM_METRIC,
        'Rolling Window Features': EXAMPLE_PLUGIN_ROLLING_FEATURES,
    }


# ============================================================================
# PLUGIN TEMPLATE
# ============================================================================

PLUGIN_TEMPLATE = '''"""
My Custom Plugin
Description of what your plugin does
"""

PLUGIN_INFO = {
    'name': 'My Custom Plugin',
    'description': 'Describe what this plugin does',
    'author': 'Your Name',
    'version': '1.0',
    'category': 'analysis',  # preprocessing, analysis, postprocessing, visualization
    'inputs': ['dataframe'],
    'outputs': ['dataframe'],  # dataframe, dict, figure
}

PLUGIN_PARAMETERS = {
    'param1': {
        'type': 'float',
        'default': 1.0,
        'min': 0,
        'max': 10,
        'description': 'Description of param1'
    },
    'param2': {
        'type': 'bool',
        'default': True,
        'description': 'Enable/disable something'
    },
}

def process(data, columns=None, target=None, **params):
    """
    Main processing function.
    
    Args:
        data: pandas DataFrame with the loaded data
        columns: list of selected feature column names
        target: name of target column (or None)
        **params: values from PLUGIN_PARAMETERS
        
    Returns:
        result: Your output (DataFrame, dict, or matplotlib Figure)
        message: Status message to display
    """
    # Your code here
    result = data.copy()
    
    # Example: work with selected columns
    if columns:
        for col in columns:
            # Do something with each column
            pass
    
    message = "✅ Processing complete!"
    return result, message
'''


def get_plugin_template() -> str:
    """Get the plugin template code"""
    return PLUGIN_TEMPLATE
