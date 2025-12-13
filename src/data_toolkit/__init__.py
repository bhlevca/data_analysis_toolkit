"""
Advanced Data Analysis Toolkit
==============================

A comprehensive, modular data analysis application with modern GUI.

Modules
-------
- data_loading_methods: Data loading and preprocessing
- statistical_analysis: Statistical analysis tools
- ml_models: Machine learning models
- neural_networks: Deep learning models (MLP, LSTM, Autoencoder)
- bayesian_analysis: Bayesian statistical methods
- uncertainty_analysis: Uncertainty quantification
- nonlinear_analysis: Non-linear analysis methods
- timeseries_analysis: Time series analysis
- causality_analysis: Causality testing
- visualization_methods: Data visualization

Quick Start
-----------
GUI Application::

    from data_toolkit.main_gui import main
    main()

Or use as a library::

    from data_toolkit import DataLoader, StatisticalAnalysis, NeuralNetworkModels
    
    loader = DataLoader()
    loader.load_csv('data.csv')
    
    stats = StatisticalAnalysis(loader.df)
    results = stats.descriptive_stats(['col1', 'col2'])
    
    # Neural Networks
    nn = NeuralNetworkModels(loader.df)
    lstm_results = nn.lstm_forecast('price', sequence_length=30)
"""

from .data_loading_methods import DataLoader
from .statistical_analysis import StatisticalAnalysis
from .ml_models import MLModels
from .bayesian_analysis import BayesianAnalysis
from .uncertainty_analysis import UncertaintyAnalysis
from .nonlinear_analysis import NonLinearAnalysis
from .timeseries_analysis import TimeSeriesAnalysis
from .causality_analysis import CausalityAnalysis
from .visualization_methods import VisualizationMethods

# Neural Networks (optional - requires TensorFlow)
try:
    from .neural_networks import NeuralNetworkModels
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError:
    NeuralNetworkModels = None
    NEURAL_NETWORKS_AVAILABLE = False

# Optional Rust-accelerated functions (with Python fallback)
from .rust_accelerated import (
    # Settings and status
    AccelerationSettings,
    is_rust_available,
    is_rust_enabled,
    set_rust_enabled,
    get_backend_name,
    # Accelerated functions
    distance_correlation,
    bootstrap_linear_regression,
    monte_carlo_predictions,
    transfer_entropy,
    lead_lag_correlations,
    detect_outliers_iqr,
    mutual_information,
    rolling_statistics,
)

# Plugin system
from .plugin_system import (
    PluginManager,
    Plugin,
    PluginInfo,
    PluginParameter,
    get_plugin_template,
    get_example_plugins,
)

__version__ = "10.0.0"
__author__ = "Data Analysis Toolkit Contributors"
__license__ = "MIT"

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    # Core classes
    'DataLoader',
    'StatisticalAnalysis',
    'MLModels',
    'NeuralNetworkModels',
    'NEURAL_NETWORKS_AVAILABLE',
    'BayesianAnalysis',
    'UncertaintyAnalysis',
    'NonLinearAnalysis',
    'TimeSeriesAnalysis',
    'CausalityAnalysis',
    'VisualizationMethods',
    # Acceleration settings
    'AccelerationSettings',
    'is_rust_available',
    'is_rust_enabled',
    'set_rust_enabled',
    'get_backend_name',
    # Rust-accelerated functions
    'distance_correlation',
    'bootstrap_linear_regression',
    'monte_carlo_predictions',
    'transfer_entropy',
    'lead_lag_correlations',
    'detect_outliers_iqr',
    'mutual_information',
    'rolling_statistics',
    # Plugin system
    'PluginManager',
    'Plugin',
    'PluginInfo',
    'PluginParameter',
    'get_plugin_template',
    'get_example_plugins',
]
