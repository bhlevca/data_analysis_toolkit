"""
Advanced Data Analysis Toolkit
==============================

A comprehensive, modular data analysis application with modern GUI.

Modules
-------
Core Analysis:
- data_loading_methods: Data loading and preprocessing
- statistical_analysis: Statistical analysis tools (with multiple testing, VIF, robust stats)
- ml_models: Machine learning models
- neural_networks: Deep learning models (MLP, LSTM, Autoencoder)

Statistical Methods:
- bayesian_analysis: Bayesian statistical methods
- uncertainty_analysis: Uncertainty quantification
- effect_sizes: Effect size calculations (Cohen's d, eta-squared, etc.)

Time Series & Signal:
- timeseries_analysis: Time series analysis
- advanced_timeseries: Prophet, changepoint detection, DTW, VAR
- signal_analysis: Signal processing methods

Specialized Analysis:
- nonlinear_analysis: Non-linear analysis methods
- causality_analysis: Causality testing
- survival_analysis: Kaplan-Meier, Cox regression
- domain_specific: Ecology, climate, biostatistics methods

Data & Model Quality:
- data_quality: Missing data analysis, imputation, transformations
- model_validation: Cross-validation, calibration, diagnostics
- feature_selection: RFE, Boruta, SHAP-based selection

Interpretation & Reporting:
- interpretability: SHAP, LIME, permutation importance
- report_generator: Automated reproducible reports

Visualization:
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

    # New modules
    from data_toolkit import EffectSizes, ModelValidation, SurvivalAnalysis

    effect = EffectSizes(loader.df)
    cohens_d = effect.cohens_d('group1_col', 'group2_col')
"""

from .bayesian_analysis import BayesianAnalysis
from .cart_analysis import CARTAnalysis, sensitivity_to_cart_workflow
from .causality_analysis import CausalityAnalysis
from .data_loading_methods import DataLoader
from .data_quality import DataQuality
# New analysis modules
from .effect_sizes import EffectSizes
from .extended_statistics import (DistributionOperations,
                                  ExtendedStatisticalTests)
from .feature_selection import FeatureSelection
from .ml_models import MLModels
from .model_validation import ModelValidation
from .nonlinear_analysis import NonLinearAnalysis
from .report_generator import ReportGenerator
from .sensitivity_analysis import (SensitivityAnalysis,
                                   analyze_dataframe_sensitivity)
from .statistical_analysis import StatisticalAnalysis
from .timeseries_analysis import TimeSeriesAnalysis
from .uncertainty_analysis import UncertaintyAnalysis
from .visualization_methods import VisualizationMethods

# Optional modules (may have additional dependencies)
try:
    from .interpretability import ModelInterpretability
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    ModelInterpretability = None
    INTERPRETABILITY_AVAILABLE = False

try:
    from .survival_analysis import SurvivalAnalysis
    SURVIVAL_AVAILABLE = True
except ImportError:
    SurvivalAnalysis = None
    SURVIVAL_AVAILABLE = False

try:
    from .advanced_timeseries import AdvancedTimeSeries
    ADVANCED_TS_AVAILABLE = True
except ImportError:
    AdvancedTimeSeries = None
    ADVANCED_TS_AVAILABLE = False

try:
    from .domain_specific import DomainSpecificAnalysis
    DOMAIN_SPECIFIC_AVAILABLE = True
except ImportError:
    DomainSpecificAnalysis = None
    DOMAIN_SPECIFIC_AVAILABLE = False

# Neural Networks (optional - requires TensorFlow)
try:
    from .neural_networks import NeuralNetworkModels
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError:
    NeuralNetworkModels = None
    NEURAL_NETWORKS_AVAILABLE = False

# Plugin system
from .plugin_system import (Plugin, PluginInfo, PluginManager, PluginParameter,
                            get_example_plugins, get_plugin_template)
# Optional Rust-accelerated functions (with Python fallback)
from .rust_accelerated import (  # Settings and status; Accelerated functions
    AccelerationSettings, bootstrap_linear_regression, detect_outliers_iqr,
    distance_correlation, get_backend_name, is_rust_available, is_rust_enabled,
    lead_lag_correlations, monte_carlo_predictions, mutual_information,
    rolling_statistics, set_rust_enabled, transfer_entropy)

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
    # New modules (v10.0)
    'EffectSizes',
    'ModelValidation',
    'ReportGenerator',
    'DataQuality',
    'FeatureSelection',
    'ModelInterpretability',
    'INTERPRETABILITY_AVAILABLE',
    'SurvivalAnalysis',
    'SURVIVAL_AVAILABLE',
    'AdvancedTimeSeries',
    'ADVANCED_TS_AVAILABLE',
    'DomainSpecificAnalysis',
    'DOMAIN_SPECIFIC_AVAILABLE',
    # Sensitivity analysis (v10.1)
    'SensitivityAnalysis',
    'analyze_dataframe_sensitivity',
    # Extended statistics (v10.1)
    'ExtendedStatisticalTests',
    'DistributionOperations',
    # CART analysis (v10.1)
    'CARTAnalysis',
    'sensitivity_to_cart_workflow',
]
