"""
Tab modules for the Data Analysis Toolkit

Each tab is now in its own file for better maintainability.
"""

from .data_tab import render_data_tab
from .statistical_tab import render_statistical_tab
from .statistical_tests_tab import render_statistical_tests_tab
from .ml_tab import render_ml_tab
from .neural_networks_tab import render_neural_networks_tab
from .pca_tab import render_pca_tab
from .bayesian_tab import render_bayesian_tab
from .uncertainty_tab import render_uncertainty_tab
from .nonlinear_tab import render_nonlinear_tab
from .timeseries_tab import render_timeseries_tab
from .causality_tab import render_causality_tab
from .visualization_tab import render_visualization_tab
from .clustering_tab import render_clustering_tab
from .anomaly_tab import render_anomaly_tab
from .signal_analysis_tab import render_signal_analysis_tab
from .dimreduction_tab import render_dimreduction_tab
from .tutorial_sidebar import render_tutorial_sidebar

__all__ = [
    'render_data_tab',
    'render_statistical_tab',
    'render_statistical_tests_tab',
    'render_ml_tab',
    'render_neural_networks_tab',
    'render_pca_tab',
    'render_bayesian_tab',
    'render_uncertainty_tab',
    'render_nonlinear_tab',
    'render_timeseries_tab',
    'render_causality_tab',
    'render_visualization_tab',
    'render_clustering_tab',
    'render_anomaly_tab',
    'render_signal_analysis_tab',
    'render_dimreduction_tab',
    'render_tutorial_sidebar',
]
