# Data Analysis Toolkit - Modular Refactor

## Overview

The `streamlit_app.py` has been refactored from a 5000+ line monolithic file into a clean modular structure:

```
src/data_toolkit/
├── streamlit_app.py          # Main entry point (~200 lines)
├── tabs/                     # Tab modules
│   ├── __init__.py           # Exports all tab functions
│   ├── data_tab.py           # Data loading
│   ├── statistical_tab.py    # Descriptive statistics
│   ├── statistical_tests_tab.py  # Hypothesis tests
│   ├── bayesian_tab.py       # Bayesian inference
│   ├── uncertainty_tab.py    # Uncertainty analysis
│   ├── signal_analysis_tab.py    # FFT/Wavelet
│   ├── timeseries_tab.py     # Time series analysis
│   ├── causality_tab.py      # Granger causality
│   ├── ml_tab.py             # Regression/Classification
│   ├── neural_networks_tab.py    # MLP/LSTM/Autoencoder
│   ├── pca_tab.py            # Principal Component Analysis
│   ├── clustering_tab.py     # Clustering algorithms
│   ├── anomaly_tab.py        # Anomaly detection
│   ├── dimreduction_tab.py   # Dimensionality reduction
│   ├── nonlinear_tab.py      # Non-linear analysis
│   ├── visualization_tab.py  # Interactive plots
│   ├── tutorial_sidebar.py   # Help documentation
│   └── shared.py             # Common imports
└── utils/                    # Utility modules
    ├── __init__.py
    ├── constants.py          # PLOTLY_TEMPLATE, etc.
    ├── datetime_utils.py     # Date parsing
    └── session_state.py      # Session initialization
```

## Installation

1. Replace your existing `src/data_toolkit/streamlit_app.py`
2. Copy the `tabs/` folder to `src/data_toolkit/tabs/`
3. Copy the `utils/` folder to `src/data_toolkit/utils/`
4. Replace `test_data/neural_network_predict.csv`

## Bug Fixes Included

1. **Neural Networks Tab**: Fixed UnboundLocalError from redundant imports
2. **Session State Conflicts**: Renamed storage keys to avoid widget conflicts
3. **Training Results**: Now persist when uploading prediction files
4. **MLP Prediction Plot**: Shows actual vs predicted by sample index (blue/red lines)
5. **Test Data**: neural_network_predict.csv now includes target column for comparison

## Running

```bash
cd data_analysis_toolkit
streamlit run src/data_toolkit/streamlit_app.py
```

## Benefits of Modular Structure

- **Maintainability**: Each tab is ~100-500 lines instead of one 5000+ line file
- **Testability**: Individual tabs can be tested independently
- **Collaboration**: Multiple developers can work on different tabs
- **Readability**: Clear separation of concerns
- **Debugging**: Errors point to specific tab files

Quick start and image recognition docs
- Quick Start (image demo): docs/QUICK_START.md
- Streamlit integration guide: docs/STREAMLIT_INTEGRATION_GUIDE.md

