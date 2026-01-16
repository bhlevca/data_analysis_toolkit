"""
Advanced Data Analysis Toolkit - Streamlit Version with Plotly
===============================================================

A comprehensive data analysis application with integrated tutorial guidance.
Uses Plotly for interactive, zoomable charts.

Version: 4.0.0 - Scientific Research Edition
"""

import json
import os
import sys
import warnings
from io import StringIO

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Advanced Data Analysis Toolkit v4.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR IMPROVED TAB NAVIGATION
# =============================================================================
st.markdown("""
<style>
/* Improve tab scrolling on smaller screens */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: #888 #f1f1f1;
    padding-bottom: 5px;
}

/* Custom scrollbar styling for webkit browsers */
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 6px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 3px;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Tab styling for better visibility */
.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
    padding: 8px 16px;
    font-size: 0.9rem;
}

/* Visual indicator that tabs are scrollable */
.stTabs [data-baseweb="tab-list"]::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 20px;
    background: linear-gradient(to right, transparent, white);
    pointer-events: none;
}

/* Nested tabs (subtabs) - slightly smaller */
.stTabs .stTabs [data-baseweb="tab"] {
    font-size: 0.85rem;
    padding: 6px 12px;
}
</style>
""", unsafe_allow_html=True)

# Add the package directory to path for direct running
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from bayesian_analysis import BayesianAnalysis
from causality_analysis import CausalityAnalysis
# Import analysis modules
from data_loading_methods import DataLoader
from ml_models import MLModels
from nonlinear_analysis import NonLinearAnalysis
from pca_visualization import (create_pca_biplot_with_vectors,
                               generate_pca_insights, interpret_vectors)
from rust_accelerated import (AccelerationSettings, get_backend_name,
                              is_rust_available)
from statistical_analysis import StatisticalAnalysis
from timeseries_analysis import TimeSeriesAnalysis
from uncertainty_analysis import UncertaintyAnalysis
from visualization_methods import VisualizationMethods

# Import new v4.0 modules
from effect_sizes import EffectSizes
from model_validation import ModelValidation
from data_quality import DataQuality
from feature_selection import FeatureSelection
from report_generator import ReportGenerator

# Optional v4.0 modules (may require extra dependencies)
try:
    from survival_analysis import SurvivalAnalysis
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False

try:
    from interpretability import ModelInterpretability
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False

try:
    from advanced_timeseries import AdvancedTimeSeries
    ADVANCED_TS_AVAILABLE = True
except ImportError:
    ADVANCED_TS_AVAILABLE = False

try:
    from domain_specific import DomainSpecificAnalysis
    DOMAIN_SPECIFIC_AVAILABLE = True
except ImportError:
    DOMAIN_SPECIFIC_AVAILABLE = False

# =============================================================================
# IMPORT ALL TABS FROM MODULES (Clean Architecture)
# =============================================================================
# Import all tab render functions from the tabs package.
# This is the single source of truth for all tab implementations.
# Inline definitions in this file are DEPRECATED and will be removed.

from tabs import (
    render_data_tab as _render_data_tab_module,
    render_statistical_tab as _render_statistical_tab_module,
    render_statistical_tests_tab as _render_statistical_tests_tab_module,
    render_ml_tab as _render_ml_tab_module,
    render_pca_tab as _render_pca_tab_module,
    render_bayesian_tab as _render_bayesian_tab_module,
    render_uncertainty_tab as _render_uncertainty_tab_module,
    render_nonlinear_tab as _render_nonlinear_tab_module,
    render_timeseries_tab as _render_timeseries_tab_module,
    render_causality_tab as _render_causality_tab_module,
    render_visualization_tab as _render_visualization_tab_module,
    render_clustering_tab as _render_clustering_tab_module,
    render_anomaly_tab as _render_anomaly_tab_module,
    render_signal_analysis_tab as _render_signal_analysis_tab_module,
    render_dimreduction_tab as _render_dimreduction_tab_module,
)

# Optional tabs that may not be installed
try:
    from tabs.image_tab import render_image_tab
    IMAGE_TAB_AVAILABLE = True
except ImportError:
    IMAGE_TAB_AVAILABLE = False
    def render_image_tab():
        import streamlit as st
        st.warning("Image Recognition tab not available. Check tabs/image_tab.py")

try:
    from tabs.biomass_tab import render_biomass_tab
    BIOMASS_TAB_AVAILABLE = True
except ImportError:
    BIOMASS_TAB_AVAILABLE = False
    def render_biomass_tab():
        import streamlit as st
        st.warning("Biomass Segmentation tab not available. Check tabs/biomass_tab.py")

# Flags for module availability (all core tabs are now always available from modules)
TABS_MODULE_AVAILABLE = True

# =============================================================================
# PLOTLY THEME CONFIGURATION
# =============================================================================
PLOTLY_TEMPLATE = "plotly_white"


# =============================================================================
# DATE/TIME PARSING UTILITIES
# =============================================================================
def detect_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect columns with date/time data and convert them to numeric timestamps.
    Handles various formats including:
    - MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD
    - HH:MM, HH:MM:SS (24h and 12h with AM/PM)
    - Combined datetime formats

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with datetime columns converted to numeric timestamps
    """
    df_copy = df.copy()

    # Common date/time format patterns to try
    date_formats = [
        '%Y-%m-%d',           # YYYY-MM-DD
        '%Y/%m/%d',           # YYYY/MM/DD
        '%d/%m/%Y',           # DD/MM/YYYY
        '%m/%d/%Y',           # MM/DD/YYYY
        '%d-%m-%Y',           # DD-MM-YYYY
        '%m-%d-%Y',           # MM-DD-YYYY
        '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
        '%Y/%m/%d %H:%M:%S',  # YYYY/MM/DD HH:MM:SS
        '%d/%m/%Y %H:%M:%S',  # DD/MM/YYYY HH:MM:SS
        '%m/%d/%Y %H:%M:%S',  # MM/DD/YYYY HH:MM:SS
        '%Y-%m-%d %H:%M',     # YYYY-MM-DD HH:MM
        '%d/%m/%Y %H:%M',     # DD/MM/YYYY HH:MM
        '%m/%d/%Y %H:%M',     # MM/DD/YYYY HH:MM
        '%H:%M:%S',           # HH:MM:SS
        '%H:%M',              # HH:MM
        '%I:%M:%S %p',        # HH:MM:SS AM/PM
        '%I:%M %p',           # HH:MM AM/PM
    ]

    converted_cols = []

    for col in df_copy.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            continue

        # Skip if column has too many nulls (>50%)
        if df_copy[col].isnull().sum() / len(df_copy) > 0.5:
            continue

        # Try to convert with pandas auto-detection first
        try:
            # Attempt automatic datetime parsing
            temp_series = pd.to_datetime(df_copy[col], errors='coerce', infer_datetime_format=True)

            # Check if conversion was successful for most values (>70%)
            success_rate = temp_series.notna().sum() / len(temp_series)

            if success_rate > 0.7:
                # Convert to Unix timestamp (seconds since epoch) for numeric analysis
                df_copy[col] = temp_series.astype('int64') / 10**9  # Convert nanoseconds to seconds
                converted_cols.append(col)
                continue
        except:
            pass

        # If auto-detection failed, try specific formats
        converted = False
        for fmt in date_formats:
            try:
                temp_series = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                success_rate = temp_series.notna().sum() / len(temp_series)

                if success_rate > 0.7:
                    # Convert to Unix timestamp
                    df_copy[col] = temp_series.astype('int64') / 10**9
                    converted_cols.append(col)
                    converted = True
                    break
            except:
                continue

        # If still not converted, check for time-only formats
        if not converted:
            try:
                # Try parsing as time only, convert to seconds since midnight
                temp_series = pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.time
                if temp_series.notna().sum() / len(temp_series) > 0.7:
                    df_copy[col] = pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.hour * 3600 + \
                                   pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.minute * 60 + \
                                   pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.second
                    converted_cols.append(col)
            except:
                pass

    # Log conversion info if any columns were converted
    if converted_cols:
        st.info(f"📅 Detected and converted {len(converted_cols)} date/time column(s) to numeric: {', '.join(converted_cols)}")

    return df_copy


# =============================================================================
# TUTORIAL CONTENT
# =============================================================================
TUTORIALS = {
    "getting_started": """
## 🚀 Getting Started with the Advanced Data Analysis Toolkit

Welcome! This toolkit provides comprehensive data analysis capabilities organized into **6 main tabs** with **subtabs** inside each group.

---

### 🆕 New in v4.0: Scientific Research Features

| Feature | Description |
|---------|-------------|
| **Effect Sizes** | Cohen's d, Hedges' g, odds ratios with CIs |
| **Model Validation** | Nested CV, calibration, ROC analysis |
| **Data Quality** | Missing data analysis, multiple imputation |
| **Feature Selection** | RFE, Boruta, SHAP, ensemble methods |
| **Survival Analysis** | Kaplan-Meier, Cox regression |
| **Interpretability** | SHAP, LIME, partial dependence |
| **Advanced Time Series** | Prophet, changepoints, DTW |
| **Domain-Specific** | Environmental, clinical, ecology tools |

---

### 📂 Tab Structure (6 Main Tabs → Subtabs)

**Click a main tab, then click a subtab inside to access specific features:**

| Main Tab | Subtabs Inside |
|----------|----------------|
| **📁 Data** | Data Loading (upload files, select columns) |
| **📊 Statistics** | Descriptive Statistics, Hypothesis Tests, Bayesian Inference, Uncertainty Analysis |
| **🔊 Signal Processing** | FFT/Wavelet (frequency and time-frequency analysis) |
| **⏱️ Time Series** | Time Series Analysis, Causality (Granger) |
| **🤖 Machine Learning** | Regression/Classification, PCA, Clustering, Anomaly Detection, Dimensionality Reduction, Non-Linear Analysis |
| **📈 Visualization** | Interactive Plots (scatter, box, 3D, regression lines) |

---

### Step-by-Step Workflow

**Step 1: Load Your Data**
1. Click **📁 Data** main tab
2. Click **"Browse files"** and select a CSV or Excel file
3. Your data preview appears automatically

**Step 2: Select Columns**
- **Feature Columns**: Independent variables (X) - what you use to predict
- **Target Column**: Dependent variable (Y) - what you want to predict

**Step 3: Navigate to Your Analysis**

| Your Question | Main Tab → Subtab |
|---------------|-------------------|
| Basic statistics? | 📊 Statistics → Descriptive Statistics |
| Significant difference? | 📊 Statistics → Hypothesis Tests |
| Effect size? | Use API: `EffectSizes` |
| Confidence intervals? | 📊 Statistics → Uncertainty Analysis |
| Frequency content? | 🔊 Signal Processing → FFT/Wavelet |
| Time patterns? | ⏱️ Time Series → Analysis |
| Does X cause Y? | ⏱️ Time Series → Causality |
| Predict values? | 🤖 ML → Regression/Classification |
| Validate model? | Use API: `ModelValidation` |
| Reduce dimensions? | 🤖 ML → PCA or Dimensionality Reduction |
| Find clusters? | 🤖 ML → Clustering |
| Find outliers? | 🤖 ML → Anomaly Detection |
| Missing data? | Use API: `DataQuality` |
| Select features? | Use API: `FeatureSelection` |
| Survival analysis? | Use API: `SurvivalAnalysis` |
| Visualize data? | 📈 Visualization → Plots |

---

### 📈 Interactive Charts (Plotly)

All charts are **fully interactive**:
| Action | How |
|--------|-----|
| **Zoom** | Click and drag |
| **Pan** | Hold Shift and drag |
| **Hover** | See exact values |
| **Download** | Click camera icon |
| **Reset** | Double-click |

---

### 📚 API for Advanced Features (v4.0)

New modules are accessible via Python API:

```python
from data_toolkit import (
    EffectSizes,     # Effect sizes with CIs
    ModelValidation,           # Cross-validation, calibration
    DataQuality,      # Missing data, imputation
    FeatureSelection,          # Feature selection methods
    ReportGenerator,          # Automated reports
)

# Optional (require extra packages):
from data_toolkit import (
    SurvivalAnalysis,         # pip install data-toolkit[survival]
    ModelInterpretability,         # pip install data-toolkit[interpretability]
    AdvancedTimeSeries,  # pip install data-toolkit[timeseries]
    DomainSpecificAnalysis,           # pip install data-toolkit[ecology]
)
```

See the **Help topics** in the sidebar for detailed guides on each feature!
""",

    "data_loading": """
## 📁 Data Loading Tab Guide

**Location:** Main Tab: 📁 Data

This is your starting point for any analysis.

### How to Load Data

1. **Click "Browse files"** button
2. **Select your file** (CSV or Excel supported)
3. **Preview appears** automatically showing first rows

### Supported File Formats
| Format | Extensions | Notes |
|--------|------------|-------|
| **CSV** | .csv | Comma-separated values |
| **Excel** | .xlsx, .xls | Microsoft Excel files |

### Selecting Columns

After loading data, you need to select:

**Feature Columns (X variables):**
- Independent variables / predictors
- Used as input for models
- Select multiple for multivariate analysis

**Target Column (Y variable):**
- Dependent variable / response
- What you want to predict or explain
- Select ONE column

### Column Selection Tips

| Analysis Type | Features | Target |
|---------------|----------|--------|
| Regression | Predictor variables | Numeric outcome |
| Classification | Predictor variables | Categorical label |
| Clustering | All variables to cluster | Not needed |
| PCA | All numeric variables | Not needed |
| Correlation | All variables of interest | Not needed |

### Data Quality Checks

The app automatically shows:
- **Row count**: Number of observations
- **Column count**: Number of variables
- **Data types**: Numeric, categorical, datetime
- **Missing values**: Highlighted if present

💡 **Tip**: Always check your data preview before analysis!
""",

    "statistical": """
## 📊 Descriptive Statistics Subtab Guide

**Location:** Main Tab: 📊 Statistics › Subtab: Descriptive Statistics

### Buttons Available:
1. **📈 Descriptive Statistics** - Summary stats for selected columns
2. **🔗 Correlation Matrix** - Relationship heatmap
3. **🎯 Outlier Detection** - Find unusual values

---

### 📈 Descriptive Statistics
Shows: Mean, Median, Std, Min, Max, Skewness, Kurtosis

**Use when:** First loading data to understand basic properties.

---

### 🔗 Correlation Matrix
**Methods:**
| Method | When to Use |
|--------|-------------|
| **Pearson** | Linear relationships (default) |
| **Spearman** | Non-normal data, ranks |
| **Kendall** | Small samples, ordinal |

**Reading values:**
- **|r| > 0.7**: Strong
- **|r| 0.3-0.7**: Moderate
- **|r| < 0.3**: Weak

---

### 🎯 Outlier Detection
| Method | Best For |
|--------|----------|
| **IQR** | Robust, any distribution |
| **Z-score** | Normal distributions |

| Method | How It Works | Best For |
|--------|--------------|----------|
| **IQR (Interquartile Range)** | Values outside Q1 - 1.5×IQR or Q3 + 1.5×IQR | Robust, any distribution |
| **Z-score** | Values more than 3 standard deviations from mean | Normal distributions |

💡 **Tip**: Always check for outliers before running machine learning models!
""",

    "machine_learning": """
## 🤖 ML Models Tab Guide

**Location:** 9th tab (🤖 ML Models) in the Machine Learning group

### Buttons Available:
1. **🎯 Train Model** - Train on current data
2. **🔄 Cross-Validation** - Test model robustness
3. **📊 Feature Importance** - See which features matter
4. **🔮 Predict** - Apply model to new data

---

### Workflow

**Step 1: Train Model**
- Select Task Type: Regression or Classification
- Choose a model from dropdown
- Click **🎯 Train Model**
- **Plot shows:** Training data actual values (blue dots)

**Step 2: Predict on New Data**
- Upload a new CSV with same feature columns
- Click **🔮 Predict**
- **Plot shows:**
  - 🔵 Blue = Training data (actual values)
  - 🔴 Red = Predictions for new data

---

### Regression Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | y = mx + b baseline |
| **Ridge** | L2 regularization |
| **Lasso** | L1 regularization (feature selection) |
| **ElasticNet** | L1 + L2 combined |
| **Decision Tree** | Tree-based splits |
| **KNN Regressor** | K-Nearest Neighbors |
| **SVR** | Support Vector Regression |
| **Random Forest** | Ensemble of trees |
| **Gradient Boosting** | Sequential boosting |

### Classification Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Probabilistic classifier |
| **SVM** | Support Vector Machine |
| **Decision Tree** | Rule-based |
| **KNN** | Majority vote |
| **Naive Bayes** | Probabilistic |
| **Random Forest** | Ensemble |
| **Gradient Boosting** | Max accuracy |

### Metrics
**Regression:** R², RMSE, MAE, MSE
**Classification:** Accuracy, Precision, Recall, F1

💡 **Tip**: Train first, then upload new data for prediction!
""",

    "pca": """
## 🔬 PCA (Principal Component Analysis) Guide

### What is PCA?
PCA is a **dimensionality reduction** technique that transforms your data into a new coordinate system where:
- **PC1 (Principal Component 1)** captures the most variance
- **PC2** captures the second-most variance
- And so on...

### When to Use PCA
- **Too many features**: Reduce from 100s of columns to a few components
- **Visualization**: Plot high-dimensional data in 2D or 3D
- **Multicollinearity**: Remove correlated features before regression
- **Feature extraction**: Create composite features
- **Noise reduction**: Keep only high-variance components

### Interpreting PCA Results

**Explained Variance:**
- Shows how much information each component captures
- **Cumulative variance**: Total explained by first N components
- **95% rule**: Often keep enough components to explain 95% of variance

**Scree Plot:**
- Bar chart of variance per component
- Look for the **"elbow"** - where gains become small
- Components after the elbow add little information

**Biplot (PCA with Vectors):**
- Points = data samples projected onto PC1/PC2
- Arrows = original feature loadings (direction and strength)
- Arrow length = feature importance
- Arrow direction = correlation with components

### Mathematical Background
PCA finds eigenvectors of the covariance matrix:
- Eigenvectors → Principal Component directions
- Eigenvalues → Variance explained by each component

💡 **Tip**: Always standardize data before PCA (handled automatically)!
""",

    "bayesian": """
## 📈 Bayesian Analysis Guide

### What is Bayesian Analysis?
Unlike classical statistics that gives single-point estimates, Bayesian analysis provides **probability distributions** for parameters.

**Key Concept:** `Posterior ∝ Likelihood × Prior`

### Methods Available

**📊 Bayesian Regression**
- Estimates regression coefficients as distributions
- Shows uncertainty in each coefficient
- Wider distribution = more uncertainty

**📐 Credible Intervals (CI)**
- Bayesian version of confidence intervals
- 95% CI: "There's 95% probability the true value is in this range"
- **Note**: Unlike frequentist CIs, this is a probability statement!

**📏 Posterior Distributions**
- Full probability distribution for each parameter
- Can answer: "What's the probability coefficient > 0?"
- More informative than single-point estimates

**🔄 Model Comparison (BIC)**
- **BIC (Bayesian Information Criterion)**: Lower = better model
- Penalizes complexity to prevent overfitting
- Compare multiple models: choose lowest BIC

### Interpreting Results
- **Narrow posteriors**: High certainty about parameter value
- **Wide posteriors**: High uncertainty (need more data)
- **Posterior mean**: Best estimate of parameter

💡 **Tip**: Wide credible intervals suggest you need more data!
""",

    "uncertainty": """
## 🎲 Uncertainty Analysis Guide

Understanding uncertainty is crucial for making reliable predictions.

### Bootstrap Confidence Intervals
**What it does:** Resamples your data 1000+ times to estimate confidence intervals.

**When to use:**
- Data may not be normally distributed
- Want robust confidence intervals
- Small sample sizes

**Interpretation:**
- 95% CI: True value falls in this range 95% of the time
- Narrow CI = high confidence
- Wide CI = high uncertainty

---

### Residual Analysis
**Always do this after fitting any regression model!**

**Residual = Actual - Predicted**

**What to check:**
| Test | Good Result | Problem If Not |
|------|-------------|----------------|
| **Q-Q Plot** | Points on diagonal | Non-normal errors |
| **Residuals vs Fitted** | Random scatter | Heteroscedasticity |
| **Durbin-Watson ≈ 2** | No autocorrelation | Patterns in time |

**Patterns to watch for:**
- **Funnel shape**: Variance changes with fitted values
- **Curve**: Non-linear relationship missed
- **Clusters**: Missing categorical variable

---

### Monte Carlo Simulation
**What it does:** Runs thousands of random simulations to propagate uncertainty.

**Use for:**
- Uncertainty propagation through complex models
- Risk assessment
- Sensitivity analysis

---

### Prediction Intervals
**Difference from Confidence Intervals:**
- **CI**: Uncertainty about mean prediction
- **PI**: Uncertainty about individual prediction (wider!)

💡 **Tip**: Always report prediction intervals, not just point predictions!
""",

    "nonlinear": """
## 🔀 Non-Linear Analysis Guide

Standard Pearson correlation only detects **linear** relationships. These methods detect **any** relationship!

### Distance Correlation
**What it measures:** Any type of dependence (linear, quadratic, periodic, etc.)

| Relationship | Pearson r | Distance Correlation |
|-------------|-----------|---------------------|
| y = x | High (~1) | High (~1) |
| y = x² | ~0 | High (>0.5) |
| y = sin(x) | ~0 | High (>0.5) |
| y = |x| | ~0 | High (>0.5) |
| Independent | ~0 | ~0 |

**Interpretation:**
- **0**: Independent (no relationship)
- **1**: Perfect dependence
- Values 0-1: Strength of any dependence

---

### Mutual Information (MI)
**Information-theoretic measure of dependence.**

Based on entropy: How much does knowing X reduce uncertainty about Y?

**Interpretation:**
- **MI = 0**: Independent (knowing X tells nothing about Y)
- **Higher MI**: More information shared
- No upper bound (depends on entropy of variables)

---

### Polynomial Regression
**Fits curves instead of straight lines:**
- Degree 2: y = ax² + bx + c (parabola)
- Degree 3: y = ax³ + bx² + cx + d (cubic)

**Caution:** High degrees can overfit!

---

### Gaussian Process Regression (GPR)
**Non-parametric Bayesian regression.**
- No fixed functional form
- Provides uncertainty estimates
- Good for small datasets with complex patterns

💡 **Tip**: If Pearson is low but Distance Correlation is high, you have a non-linear relationship!
""",

    "timeseries": """
## ⏱️ Time Series Analysis Guide

Time series data has observations ordered in time. Special methods account for temporal dependencies.

### Step 1: Check Stationarity (ADF Test)
**Stationarity**: Statistical properties don't change over time.

| ADF p-value | Interpretation | Action |
|-------------|----------------|--------|
| p < 0.05 | Stationary ✅ | Proceed with analysis |
| p ≥ 0.05 | Non-stationary ❌ | Difference the data |

**Why it matters:** Most time series methods assume stationarity!

---

### Step 2: ACF and PACF Plots
**ACF (Autocorrelation Function):** Correlation with lagged values.
**PACF (Partial ACF):** Correlation after removing intermediate lags.

**Pattern Recognition:**
| ACF | PACF | Model |
|-----|------|-------|
| Tails off | Cuts off at lag p | AR(p) - Autoregressive |
| Cuts off at lag q | Tails off | MA(q) - Moving Average |
| Tails off | Tails off | ARMA(p,q) |

---

### Step 3: Seasonal Decomposition
**Breaks time series into components:**
- **Trend**: Long-term direction
- **Seasonal**: Repeating patterns
- **Residual**: Random noise

**Model types:**
- **Additive**: Y = Trend + Seasonal + Residual
- **Multiplicative**: Y = Trend × Seasonal × Residual

---

### Step 4: ARIMA Modeling
**ARIMA(p, d, q):**
- **p**: Autoregressive order (past values)
- **d**: Differencing order (0 if stationary)
- **q**: Moving average order (past errors)

**Common choices:**
- ARIMA(1,1,1): Simple baseline
- ARIMA(1,1,0): AR(1) with one difference
- ARIMA(0,1,1): MA(1) with one difference

💡 **Tip**: Always check stationarity first!
""",

    "causality": """
## 🔗 Causality Analysis Guide

### ⚠️ Critical Warning: Correlation ≠ Causation!
These methods test **predictive** causality, not true causation.

### Granger Causality Test
**Question:** Does knowing X improve predictions of Y?

**Interpretation:**
| p-value | Result |
|---------|--------|
| p < 0.05 | X "Granger-causes" Y (X helps predict Y) |
| p ≥ 0.05 | X does NOT Granger-cause Y |

**Important notes:**
- "Granger causality" ≠ true causation
- Based on temporal precedence only
- Both variables might be caused by a third factor

---

### Lead-Lag Analysis
**Finds optimal time shift between variables.**

**Interpretation:**
| Lag | Meaning |
|-----|---------|
| Lag < 0 | Feature LEADS target (happens before) |
| Lag = 0 | Simultaneous (no lead/lag) |
| Lag > 0 | Target LEADS feature (happens before) |

**Use cases:**
- Economic indicators leading GDP
- Weather leading crop yields
- Advertising leading sales

---

### Bidirectional Causality
Tests both directions:
- Does X Granger-cause Y?
- Does Y Granger-cause X?

**Possible outcomes:**
- X → Y only (unidirectional)
- Y → X only (unidirectional)
- X ↔ Y (bidirectional)
- No causality in either direction

💡 **Tip**: Granger causality only works with stationary time series!
""",

    "visualization": """
## 📈 Visualization Guide

All charts use **Plotly** for full interactivity.

### Interactive Controls
| Action | How |
|--------|-----|
| **Zoom** | Click and drag |
| **Pan** | Shift + drag |
| **Hover** | Mouse over for values |
| **Download** | Camera icon → PNG |
| **Reset** | Double-click |

### Available Plot Types

| Plot | Best For |
|------|----------|
| **Scatter Matrix** | Overview of all pairwise relationships |
| **Correlation Heatmap** | Visualizing correlation matrix |
| **Box Plots** | Distribution and outliers |
| **Distribution Plots** | Histograms with density |
| **3D Scatter** | Three-variable relationships |
| **Parallel Coordinates** | High-dimensional data |
| **Linear Regression Plot** | Scatter with regression line + statistics |

### Linear Regression Plot
Shows scatter plot with:
- **Best-fit line**: y = slope × x + intercept
- **Slope**: Change in Y per unit change in X
- **Intercept**: Y value when X = 0
- **R² (R-squared)**: Variance explained (0-1)
- **p-value**: Statistical significance of slope

**Interpreting R²:**
| R² Value | Interpretation |
|----------|----------------|
| > 0.9 | Excellent fit |
| 0.7 - 0.9 | Good fit |
| 0.5 - 0.7 | Moderate fit |
| < 0.5 | Poor fit |

💡 **Tip**: Use scatter matrix first to explore, then zoom in on interesting pairs!
""",

    "anomaly": """
## 🚨 Anomaly Detection Guide

Anomaly detection identifies **unusual data points** that don't fit the normal pattern.

### Methods Available

| Method | Algorithm | Best For |
|--------|-----------|----------|
| **Isolation Forest** | Random tree isolation | General purpose, scales well |
| **LOF (Local Outlier Factor)** | Local density comparison | Local anomalies, clusters |
| **MCD (Minimum Covariance Determinant)** | Robust covariance | Elliptical distributions |

### Parameters

**Contamination:** Expected fraction of anomalies (0.01 = 1%, 0.1 = 10%)
- Lower = fewer anomalies detected
- Higher = more anomalies detected

### Output Interpretation
- **Normal points**: -1 label (inliers)
- **Anomalies**: 1 label (outliers)
- Scatter plot shows anomalies in **red**

### Use Cases
- Fraud detection
- System monitoring
- Quality control
- Data cleaning

💡 **Tip**: Start with Isolation Forest - it's fast and works well for most cases!
""",

    "clustering": """
## 🎯 Clustering Guide

Clustering groups similar data points together **without predefined labels**.

### Methods Available

| Method | Description | Clusters Must Be... |
|--------|-------------|---------------------|
| **K-Means** | Minimizes within-cluster variance | Spherical, similar size |
| **Hierarchical (Agglomerative)** | Builds tree of clusters | Any shape |
| **DBSCAN** | Density-based clustering | Arbitrary shape, handles noise |
| **Gaussian Mixture (GMM)** | Probabilistic soft clustering | Elliptical |

### Parameters

**K-Means & GMM:**
- **n_clusters**: Number of clusters (must specify)

**DBSCAN:**
- **eps**: Maximum distance between neighbors
- **min_samples**: Minimum points to form cluster

### Quality Metrics

| Metric | Range | Better |
|--------|-------|--------|
| **Silhouette Score** | -1 to 1 | Higher (>0.5 good) |
| **Davies-Bouldin Index** | 0 to ∞ | Lower |
| **Calinski-Harabasz** | 0 to ∞ | Higher |

### Choosing Number of Clusters
1. **Elbow method**: Plot within-cluster variance vs k
2. **Silhouette analysis**: Maximize silhouette score
3. **Domain knowledge**: What makes sense for your data?

💡 **Tip**: K-Means is fast but assumes spherical clusters. Try DBSCAN if shapes are irregular!
""",

    "dim_reduction": """
## 📉 Dimensionality Reduction Guide

Reduces high-dimensional data to fewer dimensions for visualization or preprocessing.

### Methods Available

| Method | Type | Preserves |
|--------|------|-----------|
| **PCA (Principal Component Analysis)** | Linear | Global variance |
| **SVD (Singular Value Decomposition)** | Linear | Works on sparse matrices |
| **t-SNE (t-distributed Stochastic Neighbor Embedding)** | Non-linear | Local structure |
| **UMAP (Uniform Manifold Approximation)** | Non-linear | Local + some global |
| **ICA (Independent Component Analysis)** | Linear | Statistical independence |

### When to Use Each

| Scenario | Best Method |
|----------|-------------|
| Feature extraction for ML | PCA |
| Visualization of clusters | t-SNE or UMAP |
| Sparse data (text) | SVD (TruncatedSVD) |
| Signal separation | ICA |
| Preserving distances | UMAP |

### Parameters

**t-SNE:**
- **Perplexity**: Balance local/global (5-50, default 30)
- Higher = more global structure

**UMAP:**
- **n_neighbors**: Local neighborhood size (5-50)
- **min_dist**: How tight clusters are (0.0-1.0)

### Interpretation
- **2D/3D plots**: Look for clusters, patterns
- **Not for prediction**: Just for visualization/preprocessing

💡 **Tip**: PCA is deterministic; t-SNE/UMAP may give different results each run!
""",

    "signal_analysis": """
## 🔊 Signal Analysis Guide

Analyze frequency content and time-frequency patterns in signals.

### FFT (Fast Fourier Transform)
**Transforms time domain → frequency domain**

**Output:**
- **Frequencies (Hz)**: X-axis
- **Magnitude**: Amplitude at each frequency
- **Dominant frequency**: Strongest component

**Interpretation:**
- **Peak** at frequency f → signal has component oscillating at f Hz
- **Multiple peaks** → multiple frequency components
- **Nyquist limit**: Can only detect up to (sampling_rate / 2) Hz

---

### PSD (Power Spectral Density)
**Power distribution across frequencies**

Uses Welch's method (windowed FFT) for:
- Noise reduction
- Smoother spectrum
- More reliable peak detection

---

### CWT (Continuous Wavelet Transform)
**Time-frequency representation**

Shows how frequency content **changes over time**.

**Output:**
- **X-axis**: Time
- **Y-axis**: Frequency (or scale)
- **Color**: Power/magnitude

**Features:**
- **COI (Cone of Influence)**: Edge effects region
- **Significance contours**: Statistically significant features

**Wavelet types:**
- **Morlet (morl)**: Good frequency resolution
- **Mexican hat (mexh)**: Good time resolution
- **Gaussian (gaus)**: General purpose

---

### DWT (Discrete Wavelet Transform)
**Multi-scale decomposition**

Decomposes signal into:
- **Approximation coefficients**: Low-frequency trend
- **Detail coefficients**: High-frequency details at each level

**Wavelet types:**
- **db4, db8**: Daubechies (good all-around)
- **haar**: Simplest, good for step detection
- **sym4**: Symmetric Daubechies
- **coif1**: Coiflet (good for smooth signals)

**Use cases:**
- Denoising
- Feature extraction
- Compression

💡 **Tip**: CWT for visualization, DWT for numerical analysis!
""",

    "tests": """
## 🧪 Statistical Hypothesis Tests Guide

Formal tests for statistical significance.

### Comparing Two Groups

| Test | Assumption | Null Hypothesis |
|------|------------|-----------------|
| **Independent t-test** | Normal, equal variance | μ₁ = μ₂ (means equal) |
| **Welch's t-test** | Normal, unequal variance | μ₁ = μ₂ |
| **Mann-Whitney U** | Non-parametric | Distributions equal |
| **Paired t-test** | Normal, paired samples | Mean difference = 0 |
| **Wilcoxon Signed-Rank** | Non-parametric, paired | Distributions equal |

### Comparing 3+ Groups

| Test | Assumption | Null Hypothesis |
|------|------------|-----------------|
| **One-way ANOVA** | Normal, equal variance | All means equal |
| **Kruskal-Wallis** | Non-parametric | All distributions equal |

### Chi-Square Tests

| Test | Use For |
|------|---------|
| **Chi-Square Independence** | Are two categorical variables related? |
| **Chi-Square Goodness-of-Fit** | Does data fit expected distribution? |

### Normality Tests

| Test | Description |
|------|-------------|
| **Shapiro-Wilk** | Best for small samples (<50) |
| **Kolmogorov-Smirnov** | Works for any sample size |
| **Anderson-Darling** | More sensitive to tails |

### Correlation Tests

Tests if correlation coefficient is significantly different from 0.

### Reading p-values

| p-value | Interpretation |
|---------|----------------|
| p < 0.001 | Very strong evidence against null |
| p < 0.01 | Strong evidence |
| p < 0.05 | Moderate evidence (common threshold) |
| p ≥ 0.05 | Insufficient evidence to reject null |

### Effect Size
p-value tells significance, not importance. Also report:
- **Cohen's d**: Standardized mean difference
- **R²**: Variance explained
- **Correlation coefficient**: Strength of relationship

💡 **Tip**: Statistical significance ≠ practical significance. Always consider effect size!
""",

    "effect_sizes": """
## 📏 Effect Sizes Guide (v4.0)

Effect sizes quantify the **magnitude** of an effect, essential for scientific reporting.

### Why Effect Sizes Matter
- **p-values** tell if an effect exists
- **Effect sizes** tell how large it is
- Required by many journals (APA, etc.)
- Essential for meta-analysis and power analysis

### Available Effect Sizes

| Measure | Use Case | Formula |
|---------|----------|---------|
| **Cohen's d** | Two-group means | (M₁ - M₂) / pooled SD |
| **Hedges' g** | Small samples | Cohen's d with correction |
| **Glass's Δ** | Unequal variances | (M₁ - M₂) / SD_control |
| **Eta-squared (η²)** | ANOVA | SS_between / SS_total |
| **Omega-squared (ω²)** | ANOVA (population) | Less biased than η² |
| **Cramér's V** | Chi-square | Categorical association |
| **Odds Ratio** | 2×2 tables | Case-control studies |
| **Risk Ratio** | 2×2 tables | Prospective studies |

### Interpretation (Cohen's d)

| d Value | Interpretation | Example |
|---------|----------------|---------|
| 0.2 | Small | Barely noticeable |
| 0.5 | Medium | Noticeable to careful observer |
| 0.8 | Large | Obvious to casual observer |
| 1.2+ | Very large | Striking difference |

### Confidence Intervals
All effect sizes include confidence intervals:
- **Narrow CI** = precise estimate
- **Wide CI** = more uncertainty
- **CI crossing 0** = may not be meaningful

### API Usage
```python
from data_toolkit.effect_sizes import EffectSizes

calc = EffectSizes(df)
result = calc.cohens_d('treatment', 'control')
print(f"d = {result['cohens_d']:.3f} ({result['interpretation']})")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

💡 **Tip**: Always report effect sizes alongside p-values for complete statistical reporting!
""",

    "model_validation": """
## ✅ Model Validation Guide (v4.0)

Rigorous validation prevents overfitting and ensures reproducible research.

### Cross-Validation Methods

| Method | Use Case |
|--------|----------|
| **K-Fold CV** | Standard validation |
| **Nested CV** | Unbiased hyperparameter tuning |
| **Stratified CV** | Imbalanced classification |
| **Leave-One-Out** | Very small datasets |

### Nested Cross-Validation
**Critical for proper hyperparameter tuning!**

**Problem with regular CV**: Using same data to tune and evaluate → optimistic bias

**Solution**: Nested CV
- Outer loop: Evaluate final performance
- Inner loop: Tune hyperparameters

```python
from data_toolkit.model_validation import ModelValidation

validator = ModelValidation(df)
result = validator.nested_cross_validation(
    model, param_grid, features, target,
    outer_cv=5, inner_cv=3
)
```

### Calibration Analysis
**Are probability predictions reliable?**

A well-calibrated model: 70% predicted probability → 70% actual positive rate

**Metrics:**
- **Brier Score**: Lower = better calibration
- **ECE (Expected Calibration Error)**: Lower = better

### Residual Diagnostics
**Must-do for regression models!**

| Check | What to Look For |
|-------|------------------|
| **Normality (Q-Q plot)** | Points on diagonal |
| **Homoscedasticity** | Random scatter |
| **Durbin-Watson** | ≈2 (no autocorrelation) |
| **Cook's Distance** | Influential points |

### Learning Curves
Diagnose overfitting/underfitting:
- **High training, low test** → Overfitting
- **Both low** → Underfitting
- **Both high, close together** → Good fit

💡 **Tip**: Use nested CV when publishing to avoid overly optimistic results!
""",

    "data_quality": """
## 🔍 Data Quality Analysis Guide (v4.0)

Comprehensive missing data and quality assessment.

### Missing Data Types

| Type | Meaning | Test | Handling |
|------|---------|------|----------|
| **MCAR** | Completely random | Little's test | Safe to delete or impute |
| **MAR** | Depends on observed data | Cannot test directly | Multiple imputation |
| **MNAR** | Depends on missing value itself | Cannot test | Requires domain expertise |

### Missing Data Workflow

1. **Quantify**: `missing_data_summary()`
2. **Visualize patterns**: `missing_pattern_analysis()`
3. **Test MCAR**: `little_mcar_test()`
4. **Impute appropriately**: `impute_missing()` or `multiple_imputation()`

### Imputation Methods

| Method | Best For |
|--------|----------|
| **Mean/Median** | Quick, single column |
| **KNN** | Multivariate, preserves correlations |
| **MICE** | Multiple imputations, complex patterns |
| **Regression** | Predictable relationships |

### Multiple Imputation
**Gold standard for missing data!**

Creates multiple imputed datasets → analyzes each → combines results using Rubin's rules

```python
from data_toolkit.data_quality import DataQuality

dqa = DataQuality(df)
result = dqa.multiple_imputation(columns, n_imputations=5)
# Use result['pooled_statistics'] for final estimates
```

### Outlier Detection

| Method | Best For |
|--------|----------|
| **IQR** | Robust, any distribution |
| **Z-score** | Normal distributions |
| **MAD** | Robust to extreme outliers |
| **Isolation Forest** | Multivariate outliers |

### Quality Report
`generate_quality_report()` provides:
- Missing data summary
- Outlier detection
- Distribution analysis
- Duplicate detection
- Data type issues

💡 **Tip**: Never delete missing data without understanding why it's missing!
""",

    "feature_selection": """
## 🎯 Feature Selection Guide (v4.0)

Selecting optimal features improves model performance and interpretability.

### Why Feature Selection?
- **Reduces overfitting**
- **Improves interpretability**
- **Speeds up training**
- **Removes noise**

### Methods Available

| Method | Type | Best For |
|--------|------|----------|
| **RFE** | Wrapper | Interpretable ranking |
| **Boruta** | Wrapper | All-relevant features |
| **SHAP** | Model-specific | Complex models |
| **Statistical** | Filter | Quick screening |
| **Lasso** | Embedded | Sparse solutions |
| **Sequential** | Wrapper | Forward/backward selection |
| **Ensemble** | Combined | Robust consensus |

### Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Quick screening | Statistical (mutual_info) |
| Linear models | Lasso or RFE |
| Tree-based models | Boruta or SHAP |
| Black-box models | Permutation or SHAP |
| Publication | Ensemble (multiple methods) |

### Ensemble Selection
**Most robust approach!**

Combines multiple methods and selects features with consensus:

```python
from data_toolkit.feature_selection import FeatureSelection

selector = FeatureSelection(df)
result = selector.ensemble_selection(features, target)
print("Consensus features:", result['consensus_features'])
print("Votes:", result['feature_votes'])
```

### Statistical Selection Methods

| Method | Use Case |
|--------|----------|
| `mutual_info` | Any relationship |
| `f_classif` | Classification, linear |
| `f_regression` | Regression, linear |
| `chi2` | Categorical features |

### Avoiding Pitfalls
- **Don't select on full data** → Use CV
- **Check multicollinearity** → VIF analysis
- **Domain knowledge** → Keep important features

💡 **Tip**: Use ensemble selection for publications to show robustness!
""",

    "survival": """
## ⏳ Survival Analysis Guide (v4.0)

Time-to-event analysis with censoring (requires `lifelines` package).

### When to Use
- Clinical trials (time to death/recovery)
- Customer churn (time to cancellation)
- Equipment failure (time to breakdown)
- Subscription duration

### Key Concepts

| Term | Meaning |
|------|---------|
| **Event** | The outcome of interest (death, churn, failure) |
| **Censoring** | Observation ended before event (lost to follow-up) |
| **Survival function** | Probability of surviving past time t |
| **Hazard function** | Instantaneous risk at time t |

### Methods Available

| Method | Use Case |
|--------|----------|
| **Kaplan-Meier** | Non-parametric survival curves |
| **Cox Regression** | Covariate effects (hazard ratios) |
| **Log-Rank Test** | Compare two groups |
| **Parametric** | When distribution is known |

### Kaplan-Meier Curves
```python
from data_toolkit.survival_analysis import SurvivalAnalysis

surv = SurvivalAnalysis(df)
result = surv.kaplan_meier('time', 'event', group_col='treatment')
print(f"Median survival: {result['median_survival']}")
```

### Cox Proportional Hazards
**Hazard Ratio interpretation:**
- HR = 1: No effect
- HR > 1: Increased risk
- HR < 1: Decreased risk (protective)

```python
result = surv.cox_regression('time', 'event', ['age', 'treatment'])
# result['coefficients'] contains hazard ratios
```

### Checking Assumptions
- **Proportional hazards**: Use `proportional_hazards_test`
- **Log-rank test**: Groups should not cross

💡 **Tip**: Always check the proportional hazards assumption for Cox regression!
""",

    "interpretability": """
## 🔮 Model Interpretability Guide (v4.0)

Explain model predictions (requires `shap` and `lime` packages).

### Why Interpretability?
- **Regulatory compliance** (GDPR, healthcare)
- **Scientific understanding**
- **Debugging models**
- **Stakeholder trust**

### Methods Available

| Method | Scope | Best For |
|--------|-------|----------|
| **SHAP** | Global + Local | Any model |
| **Permutation** | Global | Model-agnostic |
| **Partial Dependence** | Global | Feature effects |
| **LIME** | Local | Single predictions |
| **Feature Interactions** | Global | Interaction detection |

### SHAP Analysis
**Gold standard for feature importance!**

Based on game theory (Shapley values):
- Fair attribution of prediction to features
- Consistent across models

```python
from data_toolkit.interpretability import ModelInterpretability

interp = ModelInterpretability(model, df, features)
result = interp.shap_analysis()
# result['shap_values'] for plotting
# result['feature_importance'] for ranking
```

### LIME Explanations
**Local Interpretable Model-agnostic Explanations**

Explains individual predictions:

```python
# Why did the model predict this for row 42?
result = interp.lime_explain(instance_idx=42)
for feature, weight in result['explanation']:
    print(f"{feature}: {weight:+.3f}")
```

### Partial Dependence Plots
Shows average effect of a feature on predictions:
- X-axis: Feature values
- Y-axis: Average prediction

### Feature Interactions
Detects when two features work together:
```python
result = interp.feature_interactions()
print("Top interactions:", result['top_interactions'])
```

💡 **Tip**: Use SHAP for global importance, LIME for explaining specific predictions!
""",

    "advanced_ts": """
## 📈 Advanced Time Series Guide (v4.0)

Sophisticated time series methods (requires optional packages).

### Prophet Forecasting
Facebook's Prophet for robust forecasting:
- Handles missing data
- Automatic seasonality detection
- Holiday effects
- Uncertainty intervals

```python
from data_toolkit.advanced_timeseries import AdvancedTimeSeries

ats = AdvancedTimeSeries(df)
result = ats.prophet_forecast('date', 'sales', periods=30)
```

### Changepoint Detection
Find structural breaks in time series:

| Method | Description |
|--------|-------------|
| **PELT** | Optimal, exact solution |
| **BinSeg** | Binary segmentation |
| **Window** | Sliding window |

```python
result = ats.detect_changepoints('value', method='pelt')
print("Changepoints at:", result['changepoints'])
```

### Dynamic Time Warping (DTW)
Compare time series of different lengths/speeds:
- Shape-based similarity
- Handles time shifts
- Used for clustering similar patterns

```python
result = ats.dtw_distance(series1, series2)
print(f"DTW distance: {result['distance']}")
```

### VAR Analysis
Vector Autoregression for multivariate series:
- Models interactions between multiple time series
- Impulse response analysis
- Forecast error variance decomposition

### Granger Causality
**Does X help predict Y?**

```python
result = ats.granger_causality('cause_col', 'effect_col', maxlag=10)
if result['is_causal']:
    print(f"Optimal lag: {result['optimal_lag']}")
```

### Cointegration Testing
For non-stationary series that move together:
- Common in economics (prices, exchange rates)
- Required for error correction models

💡 **Tip**: Prophet is great for business forecasting with daily/weekly seasonality!
""",

    "domain_specific": """
## 🔬 Domain-Specific Analysis Guide (v4.0)

Specialized methods for environmental science, clinical research, and ecology.

### Environmental Science

| Method | Use Case |
|--------|----------|
| **Mann-Kendall** | Non-parametric trend detection |
| **Sen's Slope** | Robust trend magnitude |
| **SPI** | Standardized Precipitation Index |
| **EVA** | Extreme Value Analysis |

**Mann-Kendall Trend Test:**
```python
from data_toolkit.domain_specific import DomainSpecificAnalysis

domain = DomainSpecificAnalysis(df)
result = domain.mann_kendall_test('temperature')
print(f"Trend: {result['trend']} (p={result['p_value']:.4f})")
print(f"Sen's slope: {result.get('slope', 'N/A')}")
```

### Clinical Research

| Method | Use Case |
|--------|----------|
| **Bland-Altman** | Method comparison |
| **ICC** | Intraclass Correlation |
| **Cohen's Kappa** | Inter-rater agreement |

**Bland-Altman Analysis:**
```python
result = domain.bland_altman('method_a', 'method_b')
print(f"Bias: {result['mean_difference']:.2f}")
print(f"LoA: [{result['lower_loa']:.2f}, {result['upper_loa']:.2f}]")
```

**ICC Interpretation:**
| ICC | Reliability |
|-----|-------------|
| < 0.5 | Poor |
| 0.5-0.75 | Moderate |
| 0.75-0.9 | Good |
| > 0.9 | Excellent |

### Ecology

| Method | Use Case |
|--------|----------|
| **Shannon Diversity** | Species diversity (H') |
| **Simpson Diversity** | Dominance measure |
| **Moran's I** | Spatial autocorrelation |

**Diversity Indices:**
```python
result = domain.shannon_diversity(['species_a', 'species_b', 'species_c'])
print(f"Shannon H' = {result['diversity_index']:.3f}")
print(f"Evenness = {result['evenness']:.3f}")
```

**Moran's I:**
- I > 0: Positive spatial autocorrelation (clustering)
- I ≈ 0: Random spatial pattern
- I < 0: Negative autocorrelation (dispersion)

💡 **Tip**: These methods follow field-specific conventions and reporting standards!
""",

    "statistical_enhancements": """
## 📊 Statistical Enhancements Guide (v4.0)

New statistical methods added to StatisticalAnalysis.

### Multiple Testing Correction
**Critical when running many hypothesis tests!**

| Method | Type | Use Case |
|--------|------|----------|
| **Bonferroni** | FWER | Conservative, few tests |
| **Holm** | FWER | Step-down, more powerful |
| **FDR (BH)** | FDR | Many tests, discovery |
| **FDR (BY)** | FDR | Dependent tests |

```python
from data_toolkit.statistical_analysis import StatisticalAnalysis

sa = StatisticalAnalysis(df)
result = sa.multiple_testing_correction(
    [0.01, 0.04, 0.03, 0.08], 
    method='fdr_bh'
)
print("Significant:", result['reject'])
```

### Variance Inflation Factor (VIF)
**Detect multicollinearity!**

| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation |
| 1-5 | Moderate |
| 5-10 | High |
| >10 | Severe (remove feature) |

```python
result = sa.variance_inflation_factor(feature_cols)
print(result['vif_values'])
print("High VIF features:", result['high_vif_features'])
```

### Robust Statistics
**Resistant to outliers!**

| Measure | Description |
|---------|-------------|
| **Median** | Middle value |
| **MAD** | Median Absolute Deviation |
| **Trimmed Mean** | Remove extreme 10% |
| **Winsorized Mean** | Replace extreme values |
| **Huber M-estimator** | Optimal robust location |

```python
result = sa.robust_statistics('column')
print(f"Median: {result['median']}")
print(f"MAD: {result['mad']}")
print(f"Huber: {result['huber_location']}")
```

### Robust Regression
**Regression resistant to outliers!**

| Method | Description |
|--------|-------------|
| **Huber** | Downweights outliers |
| **RANSAC** | Finds inliers |
| **Theil-Sen** | Median of slopes |

```python
result = sa.robust_regression(features, target, method='huber')
print("Coefficients:", result['coefficients'])
```

💡 **Tip**: Always correct for multiple testing when running many statistical tests!
""",

    "image_recognition": """
## 🖼️ Image Recognition Guide

Train and use CNN models for image classification.

### Overview
The Image Recognition module provides:
- **Dataset Generation**: Create labeled datasets from digit folders
- **CNN Training**: Train custom convolutional neural networks
- **Image Prediction**: Classify new images using trained models
- **Interactive Labeling**: Manually label images for training

### Generating a Dataset
1. Organize images in folders named 0-9 (for digits)
2. Use "Generate Dataset from Folder" 
3. The toolkit creates a training CSV with image paths and labels

### Training a CNN
Configure your model:
- **Epochs**: Number of training iterations (start with 10-20)
- **Batch Size**: Images per training step (32 is common)
- **Validation Split**: Portion for validation (0.2 = 20%)

```python
# Behind the scenes:
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # ... more layers
    layers.Dense(num_classes, activation='softmax')
])
```

### Making Predictions
1. Load a trained model (.keras file)
2. Upload an image
3. Get classification probabilities

### Best Practices
- Use at least 100 images per class
- Ensure consistent image sizes
- Balance classes if possible
- Start with a simple model

💡 **Tip**: The model expects grayscale 28x28 images by default (like MNIST digits).
""",

    "biomass_segmentation": """
## 🌿 Biomass Segmentation Guide

Semantic segmentation for aquatic biomass detection using U-Net.

### Overview
Specialized for detecting:
- **Dreissena mussels** (zebra/quagga mussels)
- **Cladophora algae** (green filamentous algae)

### U-Net Architecture
The U-Net is ideal for biomedical/ecological segmentation:
- **Encoder**: Captures context (what)
- **Decoder**: Enables precise localization (where)
- **Skip connections**: Preserve fine details

### Training a Model
1. Prepare paired image-mask data
2. Configure model parameters:
   - **Input Size**: Image dimensions (256x256 default)
   - **Classes**: Number of segmentation classes
   - **Epochs**: Training iterations
3. Train and monitor loss curves

### Making Predictions
1. Load trained U-Net model
2. Upload underwater/aerial images
3. View segmentation masks with class overlays

### Data Requirements
- Images: RGB or grayscale
- Masks: Labeled regions (0=background, 1=class1, etc.)
- Matching filenames for image-mask pairs

### Applications
- Invasive species monitoring
- Algae bloom detection
- Underwater habitat mapping
- Environmental impact assessment

💡 **Tip**: Use data augmentation (rotation, flip, zoom) to improve model robustness!
"""
}


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = []
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True
    if 'current_tutorial' not in st.session_state:
        st.session_state.current_tutorial = "getting_started"
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'use_rust' not in st.session_state:
        st.session_state.use_rust = is_rust_available()


# =============================================================================
# TUTORIAL SIDEBAR
# =============================================================================
def render_tutorial_sidebar():
    """Render the tutorial sidebar matching the tab/subtab structure"""
    with st.sidebar:
        st.markdown("# 📚 Help & Documentation")

        st.session_state.show_tutorial = st.checkbox(
            "Show Documentation Panel",
            value=st.session_state.show_tutorial
        )

        if st.session_state.show_tutorial:
            st.markdown("---")

            # Match the exact tab/subtab structure
            st.markdown("**Select a topic:**")

            tutorial_topics = {
                # Getting started
                "getting_started": "🚀 Getting Started",
                # Data group
                "data_loading": "📁 Data › Data Loading",
                # Statistics group
                "statistical": "📊 Statistics › Descriptive Statistics",
                "tests": "📊 Statistics › Hypothesis Tests",
                "bayesian": "📊 Statistics › Bayesian Inference",
                "uncertainty": "📊 Statistics › Uncertainty Analysis",
                "effect_sizes": "📊 Statistics › Effect Sizes (v4)",
                "statistical_enhancements": "📊 Statistics › Multiple Testing & VIF (v4)",
                # Signal Processing group
                "signal_analysis": "🔊 Signal Processing › FFT/Wavelet",
                # Time Series group
                "timeseries": "⏱️ Time Series › Analysis",
                "causality": "⏱️ Time Series › Causality (Granger)",
                "advanced_ts": "⏱️ Time Series › Advanced (v4)",
                # Machine Learning group
                "machine_learning": "🤖 ML › Regression/Classification",
                "model_validation": "🤖 ML › Model Validation (v4)",
                "pca": "🤖 ML › PCA (Principal Components)",
                "clustering": "🤖 ML › Clustering",
                "anomaly": "🤖 ML › Anomaly Detection",
                "dim_reduction": "🤖 ML › Dimensionality Reduction",
                "nonlinear": "🤖 ML › Non-Linear Analysis",
                "feature_selection": "🤖 ML › Feature Selection (v4)",
                "interpretability": "🤖 ML › Interpretability (v4)",
                "image_recognition": "🖼️ ML › Image Recognition",
                "biomass_segmentation": "🌿 ML › Biomass Segmentation",
                # Data Quality group
                "data_quality": "🔍 Data Quality › Missing Data & QA (v4)",
                # Domain-specific group
                "survival": "🏥 Domain › Survival Analysis (v4)",
                "domain_specific": "🔬 Domain › Environmental/Clinical/Ecology (v4)",
                # Visualization group
                "visualization": "📈 Visualization › Plots",
            }

            selected = st.selectbox(
                "Documentation Topic",
                options=list(tutorial_topics.keys()),
                format_func=lambda x: tutorial_topics[x],
                index=list(tutorial_topics.keys()).index(st.session_state.current_tutorial)
            )
            st.session_state.current_tutorial = selected

            st.markdown("---")
            st.markdown(TUTORIALS[selected])

        st.markdown("---")

        # Backend toggle
        st.markdown("### ⚡ Performance")
        rust_available = is_rust_available()

        if rust_available:
            use_rust = st.checkbox(
                "🦀 Rust Acceleration",
                value=st.session_state.use_rust,
                help="Enable Rust backend for 10-50x speedup"
            )
            st.session_state.use_rust = use_rust
            AccelerationSettings.set_use_rust(use_rust)

            if use_rust:
                st.success("⚡ Using Rust (Fast)")
            else:
                st.info("🐍 Using Python")
        else:
            st.warning("🐍 Python only")
            st.caption("Run `maturin develop --release` in rust_extensions/ for speedup")


# =============================================================================
# DATA LOADING TAB
# =============================================================================
def render_data_tab():
    """Render the data loading tab - delegates to module version"""
    _render_data_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("📁 Data Loading & Column Selection")
    st.caption("Upload CSV or Excel files, preview data, and select feature/target columns for analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)

                # Detect and convert date/time columns to numeric
                st.session_state.df = detect_and_convert_datetime_columns(st.session_state.df)

                # Clear cached analysis results when loading new data
                st.session_state.analysis_results = {}

                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col2:
        st.markdown("### Or use sample data:")
        if st.button("🎲 Generate Sample Data"):
            np.random.seed(42)
            n = 200
            x1 = np.random.normal(50, 10, n)
            x2 = 0.7 * x1 + np.random.normal(0, 5, n)
            x3 = np.random.uniform(0, 100, n)
            target = 2.5 * x1 + 1.8 * x2 - 0.5 * x3 + np.random.normal(0, 15, n)

            st.session_state.df = pd.DataFrame({
                'feature_1': x1,
                'feature_2': x2,
                'feature_3': x3,
                'target': target
            })

            # Detect and convert any date/time columns in sample data
            st.session_state.df = detect_and_convert_datetime_columns(st.session_state.df)

            # Clear cached analysis results when generating new data
            st.session_state.analysis_results = {}

            st.success("✅ Sample data generated!")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("---")

        # Data info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        col4.metric("Missing", df.isnull().sum().sum())

        st.markdown("---")

        # Column selection
        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()

            # Initialize feature_cols in session state if not set or if columns changed
            if 'feature_cols_widget' not in st.session_state:
                # Set initial default - prefer numeric columns but allow any
                initial_features = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                st.session_state.feature_cols_widget = initial_features
            else:
                # Filter out any columns that no longer exist in the data
                st.session_state.feature_cols_widget = [
                    f for f in st.session_state.feature_cols_widget if f in all_cols
                ]

            st.multiselect(
                "📊 Select Feature Columns",
                options=all_cols,
                key="feature_cols_widget",
                help="Select one or more columns to use as input features for analysis (includes categorical columns for ANOVA, etc.)"
            )
            # Sync to the main session state key used elsewhere
            st.session_state.feature_cols = st.session_state.feature_cols_widget

        with col2:
            target_options = ['None'] + all_cols

            # Initialize target_col widget state if needed
            if 'target_col_widget' not in st.session_state:
                st.session_state.target_col_widget = 'None'
            elif st.session_state.target_col_widget not in target_options:
                st.session_state.target_col_widget = 'None'

            st.selectbox(
                "🎯 Select Target Column",
                options=target_options,
                key="target_col_widget",
                help="Select the variable you want to predict (for supervised learning)"
            )
            # Sync to main session state
            st.session_state.target_col = st.session_state.target_col_widget if st.session_state.target_col_widget != 'None' else None

        # Validation: Check for overlap between features and target
        validation_errors = []
        validation_warnings = []

        if st.session_state.target_col and st.session_state.target_col in st.session_state.feature_cols:
            validation_errors.append(
                f"⚠️ **Column Overlap Detected**: '{st.session_state.target_col}' is selected as BOTH a feature AND the target. "
                "Please remove it from features or choose a different target column."
            )

        if len(st.session_state.feature_cols) == 0:
            validation_warnings.append("💡 No feature columns selected. Select at least one feature column for analysis.")

        if st.session_state.target_col is None:
            validation_warnings.append("💡 No target column selected. Some analyses (regression, classification) require a target.")

        # Display validation messages
        for error in validation_errors:
            st.error(error)
        for warning in validation_warnings:
            st.info(warning)

        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Quick plot with Plotly - only if valid selection and numeric columns available
        if len(st.session_state.feature_cols) >= 1 and len(numeric_cols) >= 1:
            st.markdown("### Quick Visualization (Interactive!)")
            st.caption("💡 Scatter plots require numeric columns for X and Y axes")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                # Use numeric columns for visualization
                viz_numeric_cols = numeric_cols if len(numeric_cols) > 0 else all_cols
                x_col = st.selectbox("X-axis (numeric)", viz_numeric_cols,
                                    index=0 if len(viz_numeric_cols) > 0 else 0,
                                    key="quick_viz_x")

            with col2:
                default_y_idx = 1 if len(viz_numeric_cols) > 1 else 0
                # If target is set and numeric, use it as default
                if st.session_state.target_col and st.session_state.target_col in viz_numeric_cols:
                    default_y_idx = viz_numeric_cols.index(st.session_state.target_col)

                y_col = st.selectbox("Y-axis (numeric)", viz_numeric_cols,
                                    index=default_y_idx,
                                    key="quick_viz_y")

            if x_col and y_col:
                # Check if both columns are numeric
                x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
                y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
                
                if x_is_numeric and y_is_numeric:
                    try:
                        fig = px.scatter(
                            df, x=x_col, y=y_col,
                            trendline="ols",
                            title=f'{y_col} vs {x_col}',
                            template=PLOTLY_TEMPLATE
                        )
                        fig.update_layout(height=500,
                                        xaxis_title=x_col,
                                        yaxis_title=y_col)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate quick visualization: {e}")
                else:
                    st.info(f"📊 Select numeric columns for scatter plot. Current selection: X='{x_col}' ({'numeric' if x_is_numeric else 'non-numeric'}), Y='{y_col}' ({'numeric' if y_is_numeric else 'non-numeric'})")
        elif len(numeric_cols) == 0:
            st.info("📊 No numeric columns available for quick visualization. Load data with numeric columns to see scatter plots.")


# =============================================================================
# STATISTICAL ANALYSIS TAB
# =============================================================================
def render_statistical_tab():
    """Render statistical analysis tab - delegates to module version"""
    _render_statistical_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("📊 Descriptive Statistics & Correlation Analysis")
    st.caption("Compute summary statistics, correlation matrices, and detect outliers")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first in the Data Loading tab.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns in the Data Loading tab.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    # Initialize analyzer with dataframe
    stats = StatisticalAnalysis(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Descriptive Statistics", width='stretch'):
            st.session_state.analysis_results['descriptive'] = stats.descriptive_stats(features)

    with col2:
        corr_method = st.selectbox("Correlation Method", ['pearson', 'spearman', 'kendall'])
        if st.button("🔗 Correlation Matrix", width='stretch'):
            st.session_state.analysis_results['correlation'] = stats.correlation_matrix(features, method=corr_method)

    with col3:
        outlier_method = st.selectbox("Outlier Method", ['iqr', 'zscore'])
        if st.button("🎯 Outlier Detection", width='stretch'):
            # Correct method name: outlier_detection
            st.session_state.analysis_results['outliers'] = stats.outlier_detection(features, method=outlier_method)

    st.markdown("---")

    # Display results
    if 'descriptive' in st.session_state.analysis_results:
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(st.session_state.analysis_results['descriptive'], width='stretch')

    if 'correlation' in st.session_state.analysis_results:
        st.subheader("🔗 Correlation Matrix")
        corr_data = st.session_state.analysis_results['correlation']

        fig = px.imshow(
            corr_data,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Correlation Heatmap'
        )
        fig.update_layout(height=600, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, width='stretch')

    if 'outliers' in st.session_state.analysis_results:
        st.subheader("🎯 Outlier Detection Results")
        outlier_data = st.session_state.analysis_results['outliers']

        # Box plots
        # Use unique column names to avoid conflicts with existing DataFrame columns
        box_data = df[features].melt(var_name='_Feature_', value_name='_Value_')
        fig = px.box(box_data, x='_Feature_', y='_Value_', title='Box Plots with Outliers',
                    template=PLOTLY_TEMPLATE, points='outliers')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

        for col, info in outlier_data.items():
            n_outliers = info.get('n_outliers', 0)
            pct = info.get('percentage', 0)
            with st.expander(f"**{col}**: {n_outliers} outliers ({pct:.1f}%)"):
                if 'lower_bound' in info:
                    st.write(f"Lower bound: {info['lower_bound']:.4f}")
                if 'upper_bound' in info:
                    st.write(f"Upper bound: {info['upper_bound']:.4f}")


# =============================================================================
# MACHINE LEARNING TAB
# =============================================================================
def render_ml_tab():
    """Render Machine Learning tab - delegates to module version"""
    _render_ml_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🤖 Machine Learning: Regression & Classification Models")
    st.caption("Train supervised learning models to predict values (regression) or categories (classification)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns in the sidebar.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    # Check for duplicate column selection (feature also selected as target)
    if target in features:
        st.error(f"⚠️ Target column '{target}' is also selected as a feature. Please remove it from features or choose a different target.")
        return

    # Initialize with dataframe
    ml = MLModels(df)

    # Track if classification data is valid (will be set below)
    classification_data_invalid = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Selection")

        # Choose between Regression and Classification
        task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

        if task_type == "Regression":
            model_type = st.selectbox(
                "Choose Regression Model",
                ["Linear Regression", "Ridge Regression", "Lasso Regression",
                 "ElasticNet", "Random Forest Regressor", "Gradient Boosting Regressor",
                 "Decision Tree Regressor", "K-Nearest Neighbors Regressor", "Support Vector Regressor (SVR)"],
                help="Regression models predict continuous numerical values"
            )
        else:
            model_type = st.selectbox(
                "Choose Classification Model",
                ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier",
                 "Decision Tree Classifier", "K-Nearest Neighbors (KNN)",
                 "Support Vector Machine (SVM)", "Naive Bayes (Gaussian)"],
                help="Classification models predict categorical labels/classes"
            )
            # Check if target is actually categorical (not continuous)
            unique_values = df[target].nunique()
            is_float_target = df[target].dtype in ['float64', 'float32']
            has_decimals = is_float_target and (df[target] % 1 != 0).any()

            if has_decimals:
                classification_data_invalid = True
                st.error(f"⚠️ Target '{target}' contains continuous (decimal) values. Classification requires discrete classes. Switch to Regression or use a target with discrete categories.")
            elif unique_values > 20:
                st.warning(f"⚠️ Target '{target}' has {unique_values} unique values. Classification works best with fewer classes (typically < 20).")

        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2)

        # Show data info
        st.info(f"📊 Training data: {len(df)} samples, {len(features)} features")

    with col2:
        st.subheader("Model Training")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            # Disable train button if classification data is invalid
            train_disabled = task_type == "Classification" and classification_data_invalid
            if st.button("🎯 Train Model", width='stretch', disabled=train_disabled):
                with st.spinner("Training..."):
                    results = ml.train_model(features, target, model_type, test_size=test_size)
                    st.session_state.analysis_results['ml_model'] = results
                    # Only persist model if training succeeded
                    if 'error' not in results:
                        st.session_state.trained_model = ml.model
                        st.session_state.trained_scaler = ml.scaler
                        st.session_state.trained_features = features
                        st.session_state.trained_target = target
                        st.session_state.trained_task_type = task_type
                        st.success("✅ Model trained!")
                    else:
                        st.error(f"Training failed: {results['error']}")

        with col_b:
            if st.button("🔄 Cross-Validation", width='stretch'):
                with st.spinner("Running CV..."):
                    # Correct API: cross_validation(features, target, cv, model_name)
                    cv_results = ml.cross_validation(features, target, cv=cv_folds, model_name=model_type)
                    st.session_state.analysis_results['cv_results'] = cv_results
                    st.success("Cross-validation complete!")

        with col_c:
            if st.button("📊 Feature Importance", width='stretch'):
                with st.spinner("Calculating..."):
                    # Correct API: feature_importance(features, target)
                    importance = ml.feature_importance(features, target)
                    st.session_state.analysis_results['feature_importance'] = importance

    st.markdown("---")

    # Display results
    if 'ml_model' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ml_model']

        if 'error' in results:
            st.error(results['error'])
        else:
            is_classifier = results.get('is_classifier', False)

            if is_classifier:
                # Classification results
                st.subheader("📈 Classification Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
                col2.metric("Precision", f"{results.get('precision', 0):.4f}")
                col3.metric("Recall", f"{results.get('recall', 0):.4f}")
                col4.metric("F1 Score", f"{results.get('f1_score', 0):.4f}")

                # Confusion Matrix
                if 'confusion_matrix' in results:
                    st.markdown("**Confusion Matrix:**")
                    cm = np.array(results['confusion_matrix'])
                    classes = results.get('classes', list(range(len(cm))))

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=[f'Pred: {c}' for c in classes],
                        y=[f'Actual: {c}' for c in classes],
                        colorscale='Blues',
                        showscale=True,
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 14}
                    ))
                    fig_cm.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        template=PLOTLY_TEMPLATE,
                        height=400
                    )
                    st.plotly_chart(fig_cm, width='stretch')

                # Classification Report
                if 'classification_report' in results:
                    with st.expander("📋 Detailed Classification Report"):
                        st.text(results['classification_report'])
            else:
                # Regression results
                st.subheader("📈 Regression Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{results.get('r2', 0):.4f}",
                           help="Coefficient of determination: 1.0 = perfect fit, 0 = no predictive power")
                col2.metric("RMSE", f"{results.get('rmse', 0):.4f}",
                           help="Root Mean Square Error: average prediction error in target units")
                col3.metric("MSE", f"{results.get('mse', 0):.4f}",
                           help="Mean Squared Error: average of squared differences")
                # Calculate MAE if we have predictions
                if 'y_test' in results and 'predictions' in results:
                    mae = np.mean(np.abs(np.array(results['y_test']) - np.array(results['predictions'])))
                    col4.metric("MAE", f"{mae:.4f}",
                               help="Mean Absolute Error: average absolute difference")

                if 'coefficients' in results:
                    with st.expander("📋 Model Coefficients", expanded=False):
                        coef_df = pd.DataFrame({
                            'Feature': list(results['coefficients'].keys()),
                            'Coefficient': list(results['coefficients'].values())
                        })
                        st.dataframe(coef_df, width='stretch')
                        if 'intercept' in results:
                            st.write(f"**Intercept:** {results['intercept']:.4f}")

                # ═══════════════════════════════════════════════════════════════
                # PLOT 1: Training Data - Actual vs Model Predictions (Test Set)
                # ═══════════════════════════════════════════════════════════════
                st.markdown("### 📊 Plot 1: Model Evaluation on Test Set")

                if 'y_test' in results and 'predictions' in results:
                    try:
                        y_test = results['y_test']
                        y_pred = results['predictions']

                        # Convert to numpy arrays, handling pandas Series
                        if hasattr(y_test, 'values'):
                            y_test = y_test.values
                        y_test = np.array(y_test).flatten()
                        y_pred = np.array(y_pred).flatten()

                        st.caption(f"Showing {len(y_test)} test samples. Blue = Actual values, Orange = Model predictions.")

                        fig_train = go.Figure()

                        # Get X_test for x-axis if available
                        result_features = results.get('features', features)
                        if 'X_test' in results and len(result_features) > 0:
                            X_test = results['X_test']
                            x_feature = result_features[0]
                            if hasattr(X_test, 'values'):
                                x_vals = X_test[x_feature].values
                            else:
                                x_vals = np.array(X_test[x_feature])
                            x_label = x_feature

                            # Sort by x for cleaner visualization
                            sort_idx = np.argsort(x_vals)
                            x_sorted = x_vals[sort_idx]
                            y_test_sorted = y_test[sort_idx]
                            y_pred_sorted = y_pred[sort_idx]
                        else:
                            x_sorted = np.arange(len(y_test))
                            y_test_sorted = y_test
                            y_pred_sorted = y_pred
                            x_label = "Sample Index"

                        # Actual values (blue circles)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_test_sorted,
                            mode='markers',
                            name='Actual (Test Set)',
                            marker=dict(opacity=0.7, color='steelblue', size=10, symbol='circle')
                        ))

                        # Model predictions (orange diamonds)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_pred_sorted,
                            mode='markers',
                            name='Model Prediction',
                            marker=dict(opacity=0.8, color='darkorange', size=8, symbol='diamond')
                        ))

                        # Add trend line through predictions (sorted)
                        fig_train.add_trace(go.Scatter(
                            x=x_sorted,
                            y=y_pred_sorted,
                            mode='lines',
                            name='Prediction Trend',
                            line=dict(color='darkorange', width=2, dash='dash'),
                            showlegend=False
                        ))

                        fig_train.update_layout(
                            title=f'Model Evaluation: Actual vs Predicted ({target})',
                            xaxis_title=x_label,
                            yaxis_title=f'{target}',
                            template=PLOTLY_TEMPLATE,
                            height=500,
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                        )
                        st.plotly_chart(fig_train, width='stretch')

                        # Residual info
                        residuals = y_test_sorted - y_pred_sorted
                        with st.expander("📉 Residual Analysis"):
                            col_r1, col_r2, col_r3 = st.columns(3)
                            col_r1.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                            col_r2.metric("Std Residual", f"{np.std(residuals):.4f}")
                            col_r3.metric("Max |Residual|", f"{np.max(np.abs(residuals)):.4f}")
                    except Exception as e:
                        st.error(f"Plot 1 error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.warning("No test data available for plotting. Train the model first.")

    # Predict on new data - ALWAYS show this section
    st.markdown("---")
    st.subheader("🔮 Predict on New Data")
    trained_model = st.session_state.get('trained_model')
    trained_features = st.session_state.get('trained_features', features)
    trained_scaler = st.session_state.get('trained_scaler')

    # Always show the upload option, even if no model is trained yet
    data_option = st.radio("Prediction data source", ["Use current data", "Upload new CSV"], index=0, key='ml_pred_source')
    new_df = df

    if data_option == "Upload new CSV":
        upload = st.file_uploader("Upload CSV for prediction", type=['csv'], key='ml_predict_upload')
        if upload is not None:
            try:
                new_df = pd.read_csv(upload)
                st.success(f"✅ Loaded {len(new_df)} rows for prediction")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                new_df = None

    if trained_model is None:
        st.info("💡 Train a model above first, then click Predict to make predictions on this data.")

    # Always show predict button (disabled state handled by logic)
    predict_disabled = trained_model is None
    if st.button("🔮 Predict", width='stretch', disabled=predict_disabled):
        if new_df is None:
            st.error("No data available for prediction")
        else:
            ml_pred = MLModels(new_df)
            ml_pred.model = trained_model
            ml_pred.scaler = trained_scaler
            pred_results = ml_pred.predict_new_data(new_df, trained_features)
            if 'error' in pred_results:
                st.error(pred_results['error'])
            else:
                st.session_state.analysis_results['ml_predictions'] = pred_results
                st.session_state.ml_prediction_df = new_df[trained_features].copy()
                # Save full dataframe to check for actual target values
                st.session_state.ml_prediction_full_df = new_df.copy()
                st.success(f"✅ Predicted {pred_results['n_rows']} rows")

    if 'ml_predictions' in st.session_state.analysis_results:
        preds = st.session_state.analysis_results['ml_predictions']
        if 'error' not in preds:
            # ═══════════════════════════════════════════════════════════════
            # PLOT 2: New Data Predictions
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📊 Plot 2: Predictions on New Data")

            base_df = st.session_state.get('ml_prediction_df')
            full_df = st.session_state.get('ml_prediction_full_df')  # Full df with target column
            plot_features = st.session_state.get('trained_features', [])
            trained_target = st.session_state.get('trained_target', 'target')

            if base_df is not None:
                try:
                    preview_df = base_df.iloc[:len(preds['predictions'])].copy()
                    preview_df['prediction'] = preds['predictions']

                    # Check if new data has actual target values for comparison
                    has_actual = False
                    if full_df is not None and trained_target in full_df.columns:
                        has_actual = True
                        actual_values = full_df[trained_target].iloc[:len(preds['predictions'])].values
                        preview_df['actual'] = actual_values

                    # Prediction metrics (if actual values available)
                    if has_actual:
                        pred_vals = np.array(preds['predictions'])
                        actual_vals = np.array(actual_values)

                        # Filter out NaN values
                        mask = ~(np.isnan(pred_vals) | np.isnan(actual_vals))
                        if mask.sum() > 0:
                            pred_clean = pred_vals[mask]
                            actual_clean = actual_vals[mask]

                            pred_r2 = 1 - np.sum((actual_clean - pred_clean)**2) / np.sum((actual_clean - np.mean(actual_clean))**2)
                            pred_rmse = np.sqrt(np.mean((actual_clean - pred_clean)**2))
                            pred_mae = np.mean(np.abs(actual_clean - pred_clean))

                            st.markdown("**📈 Prediction Quality Metrics (comparing to actual values in new data):**")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("R² Score", f"{pred_r2:.4f}",
                                       help="How well predictions match actuals: 1.0 = perfect")
                            col2.metric("RMSE", f"{pred_rmse:.4f}",
                                       help="Root Mean Square Error")
                            col3.metric("MAE", f"{pred_mae:.4f}",
                                       help="Mean Absolute Error")
                            col4.metric("N Samples", f"{len(pred_clean)}",
                                       help="Number of samples compared")

                    # Summary stats for predictions
                    st.markdown("**📊 Prediction Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Min Prediction", f"{np.min(preds['predictions']):.4f}")
                    col2.metric("Max Prediction", f"{np.max(preds['predictions']):.4f}")
                    col3.metric("Mean Prediction", f"{np.mean(preds['predictions']):.4f}")
                    col4.metric("Std Prediction", f"{np.std(preds['predictions']):.4f}")

                    # Create visualization
                    if len(plot_features) >= 1:
                        x_feature = plot_features[0]
                        has_x_feature = x_feature in preview_df.columns

                        if has_x_feature:
                            # Sort by x for cleaner visualization
                            sort_idx = np.argsort(preview_df[x_feature].values)
                            x_sorted = preview_df[x_feature].values[sort_idx]
                            pred_sorted = preview_df['prediction'].values[sort_idx]

                            fig_pred = go.Figure()

                            # If we have actual values, show them
                            if has_actual:
                                actual_sorted = preview_df['actual'].values[sort_idx]
                                st.caption("🔵 Blue = Actual values (from new data) | 🔴 Red = Model predictions")

                                fig_pred.add_trace(go.Scatter(
                                    x=x_sorted,
                                    y=actual_sorted,
                                    mode='markers',
                                    name=f'Actual ({trained_target})',
                                    marker=dict(color='steelblue', size=10, opacity=0.7, symbol='circle')
                                ))
                            else:
                                st.caption("🔴 Red diamonds = Model predictions for new data inputs")

                            # Predictions
                            fig_pred.add_trace(go.Scatter(
                                x=x_sorted,
                                y=pred_sorted,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='crimson', size=10, opacity=0.9, symbol='diamond')
                            ))

                            # Prediction trend line
                            fig_pred.add_trace(go.Scatter(
                                x=x_sorted,
                                y=pred_sorted,
                                mode='lines',
                                name='Prediction Trend',
                                line=dict(color='crimson', width=2, dash='dash'),
                                showlegend=False
                            ))

                            fig_pred.update_layout(
                                title=f'New Data: Predictions vs {x_feature}',
                                xaxis_title=x_feature,
                                yaxis_title=trained_target,
                                template=PLOTLY_TEMPLATE,
                                height=500,
                                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                            )
                            st.plotly_chart(fig_pred, width='stretch')
                        else:
                            # Fallback: plot by index
                            st.caption("🔴 Red diamonds = Model predictions")
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=list(range(len(preview_df))),
                                y=preview_df['prediction'].values,
                                mode='markers+lines',
                                name='Predictions',
                                marker=dict(color='crimson', size=10, symbol='diamond'),
                                line=dict(color='crimson', width=1, dash='dot')
                            ))
                            fig_pred.update_layout(
                                title='Predictions by Sample Index',
                                xaxis_title='Sample Index',
                                yaxis_title='Predicted Value',
                                template=PLOTLY_TEMPLATE,
                                height=450
                            )
                            st.plotly_chart(fig_pred, width='stretch')

                    # Data table
                    with st.expander("📋 Predictions Table (first 100 rows)", expanded=False):
                        st.dataframe(preview_df.head(100), width='stretch')

                        # Download button
                        csv = preview_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.dataframe(pd.DataFrame({'prediction': preds['predictions']}).head(100), width='stretch')
            else:
                st.dataframe(pd.DataFrame({'prediction': preds['predictions']}).head(100), width='stretch')

    if 'cv_results' in st.session_state.analysis_results:
        cv = st.session_state.analysis_results['cv_results']
        st.subheader("🔄 Cross-Validation Results")
        st.write(f"Mean R²: {cv['mean']:.4f} ± {cv['std']:.4f}")

        fig = go.Figure(data=[
            go.Bar(x=[f'Fold {i+1}' for i in range(len(cv['scores']))],
                  y=cv['scores'], marker_color='steelblue')
        ])
        fig.add_hline(y=cv['mean'], line_dash='dash', line_color='red')
        fig.update_layout(title='CV Scores by Fold', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')

    if 'feature_importance' in st.session_state.analysis_results:
        importance = st.session_state.analysis_results['feature_importance']
        st.subheader("📊 Feature Importance")

        fig = go.Figure(data=[
            go.Bar(y=list(importance.keys()), x=list(importance.values()),
                  orientation='h', marker_color='steelblue')
        ])
        fig.update_layout(title='Feature Importance', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')


# =============================================================================
# PCA ANALYSIS TAB
# =============================================================================
def render_pca_tab():
    """Render PCA analysis tab - delegates to module version"""
    _render_pca_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🔬 PCA (Principal Component Analysis)")

    # Educational introduction
    with st.expander("ℹ️ About PCA - The Foundation of Multivariate Data Analysis", expanded=False):
        st.markdown("""
        **PCA is the mother method for Multivariate Data Analysis (MVDA)**

        PCA finds **lines, planes, and hyper-planes** in K-dimensional space that best approximate
        the data in a least-squares sense. As Pearson described it: *"finding lines and planes of
        closest fit to systems of points in space"*.

        **Key Concepts:**
        - **Scores**: The coordinates of observations projected onto the principal component plane
        - **Loadings**: The weights showing how each original variable contributes to each PC
        - **Biplot**: Combines scores and loadings to reveal relationships between observations AND variables
        - **Explained Variance**: How much of the total data variation each PC captures

        **What PCA reveals:**
        - 📊 **Trends & Patterns**: Similar observations cluster together
        - 🎯 **Outliers**: Points far from the main cluster
        - 🔗 **Variable Relationships**: Variables pointing in similar directions are correlated
        - 📉 **Data Structure**: The main "directions" of variation in your data
        """)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    # Settings
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        variance_threshold = st.slider("Variance threshold for component selection", 0.80, 0.99, 0.95,
                                       help="Select components that explain at least this much variance")
    with col_set2:
        scale_loadings = st.slider("Loading vector scale", 1.0, 5.0, 2.5,
                                   help="Scale factor for loading vectors in biplot")

    if st.button("🔬 Run PCA Analysis", width='stretch'):
        with st.spinner("Running PCA..."):
            results = ml.pca_analysis(features, variance_threshold=variance_threshold)
            st.session_state.analysis_results['pca'] = results

    st.markdown("---")

    if 'pca' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pca']

        if 'error' in results:
            st.error(results['error'])
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Components Selected", results['n_components_selected'],
                         help="Number of PCs needed to explain the variance threshold")
            with col2:
                total_var = results['total_variance_explained']
                st.metric("Variance Explained", f"{total_var*100:.1f}%")
            with col3:
                st.metric("Original Dimensions", len(features))

            # Get data for plotting
            explained_var = results['explained_variance']
            cumsum_var = results['cumulative_variance']
            components = results['components']
            feature_names = results['feature_names']
            scores = results['transformed_data']
            n_components = min(results['n_components_selected'], scores.shape[1])

            # ═══════════════════════════════════════════════════════════════
            # SCREE PLOT & CUMULATIVE VARIANCE
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📊 Variance Explained")

            fig_scree = make_subplots(rows=1, cols=2,
                                      subplot_titles=('Scree Plot (Individual Variance)',
                                                     'Cumulative Variance Explained'))

            # Individual variance bars
            fig_scree.add_trace(
                go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))],
                       y=explained_var * 100,
                       name='Individual',
                       marker_color='steelblue',
                       text=[f'{v*100:.1f}%' for v in explained_var],
                       textposition='outside'),
                row=1, col=1
            )

            # Cumulative line
            fig_scree.add_trace(
                go.Scatter(x=[f'PC{i+1}' for i in range(len(cumsum_var))],
                          y=cumsum_var * 100,
                          mode='lines+markers',
                          name='Cumulative',
                          marker=dict(size=10, color='darkorange'),
                          line=dict(width=3, color='darkorange')),
                row=1, col=2
            )
            # 95% threshold line
            fig_scree.add_hline(y=95, line_dash='dash', line_color='red',
                               annotation_text='95% threshold', row=1, col=2)

            fig_scree.update_layout(height=400, template=PLOTLY_TEMPLATE, showlegend=False)
            fig_scree.update_yaxes(title_text='Variance (%)', row=1, col=1)
            fig_scree.update_yaxes(title_text='Cumulative Variance (%)', row=1, col=2)
            st.plotly_chart(fig_scree, width='stretch')

            # ═══════════════════════════════════════════════════════════════
            # 2D SCORE PLOT WITH LOADING VECTORS
            # ═══════════════════════════════════════════════════════════════
            st.subheader("🎯 Score Plot with Loading Vectors")
            st.caption("**Dots**: Observations projected onto PC1-PC2 plane. **Arrows**: How each original variable contributes to the PCs (loading vectors). "
                      "Arrows pointing same direction = correlated variables. Color shows observation's PC1 score.")

            if scores.shape[1] >= 2:
                fig_scores = go.Figure()

                # Scatter plot of scores with meaningful colors
                fig_scores.add_trace(go.Scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(scores)),  # Color by observation index
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Observation #', x=1.02)
                    ),
                    text=[f'Obs {i+1}' for i in range(len(scores))],
                    hovertemplate='<b>Observation %{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                    name='Observations'
                ))

                # Add reference lines at origin
                fig_scores.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig_scores.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.5)

                # Add LOADING VECTORS for original variables
                # Scale loadings to fit the score space
                max_score = max(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1])))
                max_loading = np.max(np.abs(components[:2, :]))
                load_scale = max_score * 0.8 / max_loading if max_loading > 0 else 1

                colors = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Loading vector: how this variable projects onto PC1 and PC2
                    load_x = components[0, i] * load_scale  # PC1 loading
                    load_y = components[1, i] * load_scale  # PC2 loading
                    color = colors[i % len(colors)]

                    # Arrow line
                    fig_scores.add_trace(go.Scatter(
                        x=[0, load_x],
                        y=[0, load_y],
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'{feature}',
                        hoverinfo='name',
                        showlegend=True
                    ))

                    # Arrowhead
                    fig_scores.add_annotation(
                        x=load_x, y=load_y,
                        ax=0, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=color
                    )

                    # Label at arrow tip
                    fig_scores.add_annotation(
                        x=load_x * 1.1, y=load_y * 1.1,
                        text=feature,
                        showarrow=False,
                        font=dict(size=11, color=color, family='Arial Black')
                    )

                var1 = explained_var[0] * 100
                var2 = explained_var[1] * 100
                fig_scores.update_layout(
                    title=f'Score Plot with Loading Vectors',
                    xaxis_title=f'PC1 ({var1:.1f}% variance explained)',
                    yaxis_title=f'PC2 ({var2:.1f}% variance explained)',
                    template=PLOTLY_TEMPLATE,
                    height=600,
                    xaxis=dict(scaleanchor='y', scaleratio=1),
                    legend=dict(title='Original Variables', yanchor='top', y=0.99, xanchor='left', x=1.15)
                )
                st.plotly_chart(fig_scores, width='stretch')

                with st.expander("📖 How to interpret this plot"):
                    st.markdown("""
                    **This is a combined Score Plot + Loading Vectors visualization:**

                    - **Dots (Observations)**: Each dot is a data point projected onto PC1-PC2 space
                      - Color indicates observation number (low=purple, high=yellow)
                      - Dots close together = similar observations
                      - Dots far from origin = extreme values

                    - **Arrows (Loading Vectors)**: Show how original variables relate to PCs
                      - Arrow direction: which PC direction this variable aligns with
                      - Arrow length: strength of contribution to these PCs
                      - **Arrows pointing same way**: positively correlated variables
                      - **Arrows pointing opposite**: negatively correlated variables
                      - **Arrows at 90°**: uncorrelated variables

                    **Reading tip**: Project a dot perpendicularly onto an arrow to see
                    if that observation has high/low values for that variable.
                    """)

            # ═══════════════════════════════════════════════════════════════
            # BIPLOT - Scores + Loading Vectors
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📐 Biplot: Observations AND Variables")
            st.caption("Combines score plot with loading vectors. Arrows show how each original variable "
                      "contributes to the principal components. Variables pointing in similar directions are correlated.")

            if scores.shape[1] >= 2 and len(feature_names) > 0:
                fig_biplot = go.Figure()

                # Scale scores to fit with loadings
                score_scale = np.max(np.abs(components[:2, :])) * scale_loadings
                scores_scaled = scores[:, :2] / (np.max(np.abs(scores[:, :2])) / score_scale)

                # Plot scores (observations)
                fig_biplot.add_trace(go.Scatter(
                    x=scores_scaled[:, 0],
                    y=scores_scaled[:, 1],
                    mode='markers',
                    marker=dict(size=6, color='steelblue', opacity=0.6),
                    name='Observations',
                    hovertemplate='Obs %{text}<extra></extra>',
                    text=[str(i+1) for i in range(len(scores))]
                ))

                # Plot loading vectors (variables)
                colors = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Arrow from origin to loading
                    loading_x = components[0, i] * scale_loadings
                    loading_y = components[1, i] * scale_loadings

                    color = colors[i % len(colors)]

                    # Arrow line
                    fig_biplot.add_trace(go.Scatter(
                        x=[0, loading_x],
                        y=[0, loading_y],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Arrowhead (using annotation)
                    fig_biplot.add_annotation(
                        x=loading_x, y=loading_y,
                        ax=0, ay=0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=color
                    )

                    # Label
                    fig_biplot.add_trace(go.Scatter(
                        x=[loading_x * 1.15],
                        y=[loading_y * 1.15],
                        mode='text',
                        text=[feature],
                        textfont=dict(size=12, color=color, family='Arial Black'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Reference lines
                fig_biplot.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig_biplot.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.5)

                fig_biplot.update_layout(
                    title='Biplot: Scores (points) + Loadings (vectors)',
                    xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
                    yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
                    template=PLOTLY_TEMPLATE,
                    height=600,
                    xaxis=dict(scaleanchor='y', scaleratio=1)
                )
                st.plotly_chart(fig_biplot, width='stretch')

                # Interpretation help
                with st.expander("📖 How to interpret the Biplot"):
                    st.markdown("""
                    **Reading the Biplot:**

                    1. **Observation Points (blue dots)**:
                       - Close points = similar observations
                       - Far from center = extreme/unusual observations

                    2. **Loading Vectors (colored arrows)**:
                       - Arrow length = importance of variable for these PCs
                       - Arrow direction = how the variable relates to the PCs
                       - **Arrows pointing same direction** = positively correlated variables
                       - **Arrows pointing opposite directions** = negatively correlated variables
                       - **Arrows at 90°** = uncorrelated variables

                    3. **Projecting observations onto vectors**:
                       - Drop a perpendicular from an observation to a variable vector
                       - Where it lands indicates the observation's value for that variable
                    """)

            # ═══════════════════════════════════════════════════════════════
            # 3D SCORE PLOT WITH LOADING VECTORS (if 3+ components)
            # ═══════════════════════════════════════════════════════════════
            if scores.shape[1] >= 3:
                st.subheader("🌐 3D Score Plot with Loading Vectors")
                st.caption("**Dots**: Observations in PC1-PC2-PC3 space. **Arrows**: Loading vectors showing how each original variable projects into this 3D space.")

                fig_3d = go.Figure()

                # Add data points with color by observation index
                fig_3d.add_trace(go.Scatter3d(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    z=scores[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=np.arange(len(scores)),
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='Obs #', x=1.02)
                    ),
                    text=[f'Obs {i+1}' for i in range(len(scores))],
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
                    name='Observations'
                ))

                # Add LOADING VECTORS for original variables (scaled to score space)
                max_score_3d = max(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1])), np.max(np.abs(scores[:, 2])))
                max_loading_3d = np.max(np.abs(components[:3, :])) if components.shape[0] >= 3 else 1
                load_scale_3d = max_score_3d * 0.7 / max_loading_3d if max_loading_3d > 0 else 1

                colors_3d = px.colors.qualitative.Set1
                for i, feature in enumerate(feature_names):
                    # Loading vector in 3D: how this variable projects onto PC1, PC2, PC3
                    load_x = components[0, i] * load_scale_3d
                    load_y = components[1, i] * load_scale_3d
                    load_z = components[2, i] * load_scale_3d if components.shape[0] >= 3 else 0
                    color = colors_3d[i % len(colors_3d)]

                    # Arrow line
                    fig_3d.add_trace(go.Scatter3d(
                        x=[0, load_x], y=[0, load_y], z=[0, load_z],
                        mode='lines+text',
                        line=dict(color=color, width=5),
                        text=['', feature],
                        textposition='top center',
                        textfont=dict(size=10, color=color),
                        name=feature,
                        hoverinfo='name',
                        showlegend=True
                    ))

                    # Cone arrowhead
                    length = np.sqrt(load_x**2 + load_y**2 + load_z**2)
                    if length > 0:
                        fig_3d.add_trace(go.Cone(
                            x=[load_x], y=[load_y], z=[load_z],
                            u=[load_x/length * 0.2], v=[load_y/length * 0.2], w=[load_z/length * 0.2],
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            sizemode='absolute',
                            sizeref=0.12,
                            hoverinfo='skip',
                            showlegend=False
                        ))

                # Add origin marker
                fig_3d.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=6, color='black', symbol='diamond'),
                    name='Origin',
                    hoverinfo='name',
                    showlegend=True
                ))

                fig_3d.update_layout(
                    title='3D Score Plot with Variable Loading Vectors',
                    scene=dict(
                        xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
                        yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
                        zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
                        aspectmode='data'
                    ),
                    template=PLOTLY_TEMPLATE,
                    height=700,
                    legend=dict(
                        title='Original Variables',
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
                st.plotly_chart(fig_3d, width='stretch')

                with st.expander("📖 Understanding the 3D Score Plot"):
                    st.markdown("""
                    **This is a 3D version of the Score Plot + Loading Vectors:**

                    - **Dots (Observations)**: Data points projected onto PC1-PC2-PC3 space
                      - Color indicates observation number
                      - Clusters = groups of similar observations

                    - **Arrows (Loading Vectors)**: Original variables projected onto PC space
                      - Direction shows how variable relates to PC1, PC2, PC3
                      - Length shows strength of contribution
                      - Variables pointing same direction are correlated

                    **Tip**: Rotate the 3D plot to explore the data structure from different angles.
                    """)

            # ═══════════════════════════════════════════════════════════════
            # LOADING MATRIX HEATMAP
            # ═══════════════════════════════════════════════════════════════
            st.subheader("🔥 Loading Matrix Heatmap")
            st.caption("How each original variable contributes to each principal component. "
                      "Red = positive loading, Blue = negative loading.")

            n_show = min(5, components.shape[0])
            loadings_df = pd.DataFrame(
                components[:n_show].T,
                index=feature_names,
                columns=[f'PC{i+1}' for i in range(n_show)]
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=loadings_df.values,
                x=loadings_df.columns,
                y=loadings_df.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(loadings_df.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Loading')
            ))

            fig_heat.update_layout(
                title='Variable Loadings on Principal Components',
                xaxis_title='Principal Component',
                yaxis_title='Original Variable',
                template=PLOTLY_TEMPLATE,
                height=max(300, len(feature_names) * 30)
            )
            st.plotly_chart(fig_heat, width='stretch')

            # Loadings table
            with st.expander("📋 Detailed Loading Values"):
                st.dataframe(loadings_df.round(4), width='stretch')

            # ═══════════════════════════════════════════════════════════════
            # EXPORT PCA RESULTS
            # ═══════════════════════════════════════════════════════════════
            st.subheader("📥 Export PCA Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Export scores (transformed data)
                scores_df = pd.DataFrame(
                    scores[:, :n_show],
                    columns=[f'PC{i+1}' for i in range(n_show)]
                )
                csv_scores = scores_df.to_csv(index=True)
                st.download_button(
                    label="📥 PC Scores (CSV)",
                    data=csv_scores,
                    file_name="pca_scores.csv",
                    mime="text/csv"
                )

            with col2:
                # Export loadings
                csv_loadings = loadings_df.to_csv(index=True)
                st.download_button(
                    label="📥 Loadings Matrix (CSV)",
                    data=csv_loadings,
                    file_name="pca_loadings.csv",
                    mime="text/csv"
                )

            with col3:
                # Export variance explained
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                    'Variance_Explained': explained_var,
                    'Cumulative_Variance': cumsum_var
                })
                csv_var = var_df.to_csv(index=False)
                st.download_button(
                    label="📥 Variance Explained (CSV)",
                    data=csv_var,
                    file_name="pca_variance.csv",
                    mime="text/csv"
                )


# =============================================================================
# BAYESIAN ANALYSIS TAB
# =============================================================================
def render_bayesian_tab():
    """Render Bayesian analysis tab - delegates to module version"""
    _render_bayesian_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("📈 Bayesian Inference & Analysis")
    st.caption("Bayesian regression with posterior distributions, credible intervals (CI), and model comparison using BIC")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    bayesian = BayesianAnalysis(df)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Bayesian Regression", width='stretch'):
            with st.spinner("Fitting Bayesian model..."):
                # Correct API: bayesian_regression(features, target)
                results = bayesian.bayesian_regression(features, target)
                st.session_state.analysis_results['bayesian'] = results

    with col2:
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95)
        if st.button("📊 Credible Intervals", width='stretch'):
            with st.spinner("Computing intervals..."):
                # Correct API: credible_intervals(features, target, confidence)
                results = bayesian.credible_intervals(features, target, confidence)
                st.session_state.analysis_results['credible'] = results

    st.markdown("---")

    if 'bayesian' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['bayesian']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎲 Bayesian Regression Results")

            posterior_mean = results.get('posterior_mean', [])
            feat_names = results.get('features', [])
            ci_lower = results.get('credible_intervals_lower', [])
            ci_upper = results.get('credible_intervals_upper', [])

            if len(posterior_mean) > 0:
                coef_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Posterior Mean': posterior_mean,
                    '95% CI Lower': ci_lower,
                    '95% CI Upper': ci_upper
                })
                st.dataframe(coef_df, width='stretch')

                # Plot coefficients with error bars
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=feat_names, y=posterior_mean,
                    error_y=dict(type='data',
                                array=np.array(ci_upper) - np.array(posterior_mean),
                                arrayminus=np.array(posterior_mean) - np.array(ci_lower)),
                    marker_color='steelblue'
                ))
                fig.update_layout(title='Posterior Coefficients with 95% CI',
                                template=PLOTLY_TEMPLATE, height=400)
                st.plotly_chart(fig, width='stretch')

    if 'credible' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['credible']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("📊 Credible Intervals")
            st.metric("Coverage", f"{results.get('coverage', 0)*100:.1f}%")
            st.metric("Mean CI Width", f"{results.get('mean_ci_width', 0):.4f}")


# =============================================================================
# UNCERTAINTY ANALYSIS TAB
# =============================================================================
def render_uncertainty_tab():
    """Render uncertainty analysis tab - delegates to module version"""
    _render_uncertainty_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🎲 Uncertainty Analysis")
    st.caption("Bootstrap confidence intervals, residual analysis, Monte Carlo simulation, and prediction intervals")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    uncertainty = UncertaintyAnalysis(df)

    col1, col2 = st.columns(2)

    with col1:
        n_bootstrap = st.slider("Bootstrap Samples", 100, 2000, 500)
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, key='boot_conf')

    with col2:
        n_simulations = st.slider("Monte Carlo Simulations", 100, 5000, 1000)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Bootstrap CI", width='stretch'):
            with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
                # Correct API: bootstrap_ci(features, target, n_bootstrap, confidence)
                results = uncertainty.bootstrap_ci(features, target, n_bootstrap, confidence)
                st.session_state.analysis_results['bootstrap'] = results

    with col2:
        if st.button("🎯 Residual Analysis", width='stretch'):
            with st.spinner("Analyzing residuals..."):
                # Correct API: residual_analysis(features, target)
                results = uncertainty.residual_analysis(features, target)
                st.session_state.analysis_results['residuals'] = results

    with col3:
        if st.button("🎲 Monte Carlo", width='stretch'):
            with st.spinner(f"Running {n_simulations} simulations..."):
                # Correct API: monte_carlo(features, target, n_simulations, confidence)
                results = uncertainty.monte_carlo_analysis(features, target, n_simulations, confidence)
                st.session_state.analysis_results['monte_carlo'] = results

    st.markdown("---")

    if 'bootstrap' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['bootstrap']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🔄 Bootstrap Results")

            boot_df = pd.DataFrame({
                'Feature': results.get('features', []),
                'Mean': results.get('mean_coefs', []),
                'Std Error': results.get('std_coefs', []),
                'CI Lower': results.get('ci_lower', []),
                'CI Upper': results.get('ci_upper', [])
            })
            st.dataframe(boot_df, width='stretch')

    if 'residuals' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['residuals']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎯 Residual Analysis")

            col1, col2, col3 = st.columns(3)
            col1.metric("Durbin-Watson", f"{results.get('durbin_watson', 0):.4f}")
            col2.metric("Shapiro p-value", f"{results.get('shapiro_pvalue', 0):.4f}")
            col3.metric("Mean Residual", f"{results.get('mean', 0):.4f}")

            # Residual plots
            residuals = results.get('residuals', [])
            y_pred = results.get('y_pred', [])

            if len(residuals) > 0:
                fig = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Residuals vs Fitted', 'Residual Distribution'))

                fig.add_trace(
                    go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(opacity=0.6)),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)

                fig.add_trace(
                    go.Histogram(x=residuals, nbinsx=30),
                    row=1, col=2
                )

                fig.update_layout(height=400, template=PLOTLY_TEMPLATE, showlegend=False)
                st.plotly_chart(fig, width='stretch')

    if 'monte_carlo' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['monte_carlo']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🎲 Monte Carlo Results")

            col1, col2 = st.columns(2)
            col1.metric("Mean Uncertainty", f"{results.get('mean_uncertainty', 0):.4f}")
            col2.metric("Mean CI Width", f"{results.get('mean_ci_width', 0):.4f}")


# =============================================================================
# NON-LINEAR ANALYSIS TAB
# =============================================================================
def render_nonlinear_tab():
    """Render non-linear analysis tab - delegates to module version"""
    _render_nonlinear_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🔀 Non-Linear Analysis")
    st.caption("Distance correlation, mutual information (MI), polynomial regression, and Gaussian Process Regression (GPR)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    nonlinear = NonLinearAnalysis(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Distance Correlation", width='stretch'):
            with st.spinner("Computing..."):
                # Correct API: distance_correlation(features, target)
                results = nonlinear.distance_correlation(features, target)
                st.session_state.analysis_results['dist_corr'] = results

    with col2:
        if st.button("🔮 Mutual Information", width='stretch'):
            with st.spinner("Computing..."):
                # Correct API: mutual_information(features, target)
                results = nonlinear.mutual_information(features, target)
                st.session_state.analysis_results['mutual_info'] = results

    with col3:
        max_degree = st.slider("Max Polynomial Degree", 2, 5, 3)
        if st.button("📈 Polynomial Regression", width='stretch'):
            with st.spinner("Fitting polynomials..."):
                # Correct API: polynomial_regression(features, target, max_degree)
                results = nonlinear.polynomial_regression(features, target, max_degree)
                st.session_state.analysis_results['polynomial'] = results

    st.markdown("---")

    if 'dist_corr' in st.session_state.analysis_results:
        st.subheader("📊 Distance Correlation vs Pearson")

        dist_corr = st.session_state.analysis_results['dist_corr']

        # Calculate Pearson for comparison
        pearson_corr = {}
        for feat in features:
            pearson_corr[feat] = df[feat].corr(df[target])

        comparison_df = pd.DataFrame({
            'Feature': features,
            'Pearson |r|': [abs(pearson_corr.get(f, 0)) for f in features],
            'Distance Corr': [dist_corr.get(f, 0) for f in features],
        })
        comparison_df['Non-linearity'] = comparison_df['Distance Corr'] - comparison_df['Pearson |r|']

        st.dataframe(comparison_df, width='stretch')

        fig = go.Figure(data=[
            go.Bar(name='|Pearson|', x=features, y=comparison_df['Pearson |r|'], marker_color='steelblue'),
            go.Bar(name='Distance Corr', x=features, y=comparison_df['Distance Corr'], marker_color='coral')
        ])
        fig.update_layout(barmode='group', title='Pearson vs Distance Correlation',
                         template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')

        st.info("💡 Large difference suggests non-linear relationships!")

    if 'mutual_info' in st.session_state.analysis_results:
        st.subheader("🔮 Mutual Information")
        mi = st.session_state.analysis_results['mutual_info']

        fig = go.Figure(data=[
            go.Bar(x=list(mi.keys()), y=list(mi.values()), marker_color='teal')
        ])
        fig.update_layout(title='Mutual Information Scores', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')

    if 'polynomial' in st.session_state.analysis_results:
        st.subheader("📈 Polynomial Regression Results")
        poly_results = st.session_state.analysis_results['polynomial']

        poly_df = pd.DataFrame([
            {'Degree': deg, 'R²': vals['r2'], 'RMSE': vals['rmse']}
            for deg, vals in poly_results.items()
        ])
        st.dataframe(poly_df, width='stretch')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=poly_df['Degree'], y=poly_df['R²'], mode='lines+markers', name='R²'))
        fig.update_layout(title='R² vs Polynomial Degree', template=PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, width='stretch')


# =============================================================================
# TIME SERIES TAB
# =============================================================================
def render_timeseries_tab():
    """Render time series analysis tab - delegates to module version"""
    # Always use the module version (which has CCF and Moving Average)
    _render_timeseries_tab_module()
    return

    # The code below is deprecated and unreachable
    st.header("⏱️ Time Series Analysis")
    st.caption("ACF/PACF plots, ADF stationarity test, seasonal decomposition, rolling statistics, and ARIMA modeling")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select at least one column.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    ts = TimeSeriesAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("#### Axis Selection")

        # X-axis selection
        x_options = ['Index (Row Number)'] + all_numeric
        x_col = st.selectbox("X-axis (Time/Index)", x_options,
                            help="Select a column for X-axis (typically time/date) or use row index")

        # Y-axis selection
        selected_col = st.selectbox("Y-axis (Value)", features)

        max_lag = st.slider("Max Lag", 5, 50, 20)

    # Plot the time series
    st.subheader("📈 Time Series Plot")
    series = df[selected_col].dropna()

    fig = go.Figure()

    if x_col == 'Index (Row Number)':
        # Use row index as X-axis
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=selected_col))
        fig.update_layout(xaxis_title='Index', yaxis_title=selected_col)
    else:
        # Use selected column as X-axis
        x_data = df[x_col].loc[series.index]  # Match indices with non-null Y values
        fig.add_trace(go.Scatter(x=x_data, y=series, mode='lines', name=selected_col))
        fig.update_layout(xaxis_title=x_col, yaxis_title=selected_col)

    fig.update_layout(title=f'Time Series: {selected_col}', template=PLOTLY_TEMPLATE, height=400)
    st.plotly_chart(fig, width='stretch')

    with col2:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("📊 ACF", width='stretch'):
                # Correct API: acf_analysis(column, lags)
                results = ts.acf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['acf'] = results

        with col_b:
            if st.button("📈 PACF", width='stretch'):
                # Correct API: pacf_analysis(column, lags)
                results = ts.pacf_analysis(selected_col, max_lag)
                st.session_state.analysis_results['pacf'] = results

        with col_c:
            if st.button("🔬 Stationarity", width='stretch'):
                # Correct API: stationarity_test([columns])
                results = ts.stationarity_test([selected_col])
                st.session_state.analysis_results['adf'] = results.get(selected_col, {})

        with col_d:
            default_win = min(30, max(1, len(series)//5))
            window = st.number_input("Rolling window (samples)", min_value=1, max_value=max(1, len(series)), value=default_win, step=1)
            if st.button("🔄 Rolling Stats", width='stretch'):
                # Correct API: rolling_statistics(column, window)
                results = ts.rolling_statistics(selected_col, int(window))
                st.session_state.analysis_results['rolling'] = results

    st.markdown("---")

    if 'adf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['adf']

        if 'error' in results:
            st.error(results['error'])
        else:
            st.subheader("🔬 Augmented Dickey-Fuller Test")

            col1, col2, col3 = st.columns(3)
            col1.metric("ADF Statistic", f"{results.get('adf_statistic', 0):.4f}")
            col2.metric("p-value", f"{results.get('p_value', 0):.4f}")

            if results.get('is_stationary', False):
                col3.success("✅ Stationary")
            else:
                col3.warning("⚠️ Non-stationary")

    if 'acf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['acf']
        acf_values = results.get('acf', [])
        conf_upper = results.get('conf_int_upper', 0)

        if len(acf_values) > 0:
            st.subheader("📊 ACF")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='steelblue'))
            fig.add_hline(y=conf_upper, line_dash='dash', line_color='red')
            fig.add_hline(y=-conf_upper, line_dash='dash', line_color='red')
            fig.update_layout(title='Autocorrelation Function', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, width='stretch')

    if 'pacf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pacf']
        pacf_values = results.get('pacf', [])
        conf_upper = results.get('conf_int_upper', 0)

        if len(pacf_values) > 0:
            st.subheader("📈 PACF")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='teal'))
            fig.add_hline(y=conf_upper, line_dash='dash', line_color='red')
            fig.add_hline(y=-conf_upper, line_dash='dash', line_color='red')
            fig.update_layout(title='Partial Autocorrelation Function', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, width='stretch')

    if 'rolling' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['rolling']

        if 'error' not in results:
            st.subheader("🔄 Rolling Statistics")

            original = results.get('original', [])
            rolling_mean = results.get('rolling_mean', [])
            rolling_std = results.get('rolling_std', [])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=original, mode='lines', name='Original', opacity=0.7))
            fig.add_trace(go.Scatter(y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='red')))
            fig.add_trace(go.Scatter(y=rolling_std, mode='lines', name='Rolling Std', line=dict(color='green')))
            fig.update_layout(title='Rolling Statistics', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, width='stretch')


# =============================================================================
# CROSS-CORRELATION (CCF) TAB
# =============================================================================
def render_ccf_tab():
    """Render cross-correlation analysis tab"""
    st.header("📊 Cross-Correlation Analysis (CCF)")
    st.caption("Find lead-lag relationships between two time series")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select at least two columns for cross-correlation.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    
    if len(features) < 2:
        st.warning("⚠️ Need at least 2 feature columns for cross-correlation.")
        return

    ts = TimeSeriesAnalysis(df)

    col1, col2 = st.columns(2)
    
    with col1:
        ccf_series1 = st.selectbox("First Time Series (X)", features, key="ccf_tab_series1")
    with col2:
        other_features = [f for f in features if f != ccf_series1]
        ccf_series2 = st.selectbox("Second Time Series (Y)", 
                                   other_features if other_features else features, 
                                   key="ccf_tab_series2")
    
    ccf_max_lag = st.slider("Maximum Lag", 5, 100, 30, key="ccf_tab_lag",
                           help="Compute correlations from -lag to +lag")
    
    st.info("""📖 **Interpretation:**
- **Positive lag**: X leads Y (X changes first, Y follows)
- **Negative lag**: Y leads X (Y changes first, X follows)
- **Best lag**: The lag with strongest (absolute) correlation""")
    
    if st.button("📊 Compute Cross-Correlation", key="compute_ccf", type="primary"):
        with st.spinner("Computing cross-correlation function..."):
            results = ts.cross_correlation(ccf_series1, ccf_series2, ccf_max_lag)
            st.session_state.analysis_results['ccf'] = results
            
            if 'error' in results:
                st.error(f"CCF failed: {results['error']}")

    # Display CCF results
    if 'ccf' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ccf']
        if 'error' not in results:
            lags = results.get('lags', [])
            correlations = results.get('correlations', [])
            best_lag = results.get('best_lag', 0)
            best_corr = results.get('best_correlation', 0)
            col1_name = results.get('series1', ccf_series1)
            col2_name = results.get('series2', ccf_series2)
            
            # Metrics
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Best Lag", f"{best_lag}")
            with met_col2:
                st.metric("Best Correlation", f"{best_corr:.4f}")
            with met_col3:
                if best_lag > 0:
                    st.metric("Interpretation", f"{col1_name} leads")
                elif best_lag < 0:
                    st.metric("Interpretation", f"{col2_name} leads")
                else:
                    st.metric("Interpretation", "Simultaneous")
            
            # CCF Plot - line plot (better for sinusoidal patterns)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lags, y=correlations,
                mode='lines+markers',
                line=dict(color='steelblue', width=2),
                marker=dict(size=4, color=['red' if l == best_lag else 'steelblue' for l in lags]),
                name='CCF'
            ))
            fig.add_hline(y=0, line_dash="solid", line_color="black")
            
            # Add significance bounds (approximate 95% CI)
            n = len(df)
            sig_bound = 1.96 / np.sqrt(n)
            fig.add_hline(y=sig_bound, line_dash="dash", line_color="red", 
                         annotation_text="95% CI")
            fig.add_hline(y=-sig_bound, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f'Cross-Correlation: {col1_name} vs {col2_name}',
                xaxis_title='Lag',
                yaxis_title='Correlation',
                template=PLOTLY_TEMPLATE,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export CCF data
            st.markdown("---")
            ccf_df = pd.DataFrame({
                'Lag': lags,
                'Correlation': correlations
            })
            csv_ccf = ccf_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CCF Data (CSV)",
                data=csv_ccf,
                file_name=f"ccf_{col1_name}_vs_{col2_name}.csv",
                mime="text/csv"
            )


# =============================================================================
# MOVING AVERAGE TAB
# =============================================================================
def render_moving_average_tab():
    """Render moving average / smoothing tab"""
    st.header("📈 Moving Average & Smoothing")
    st.caption("Apply moving average filters and export smoothed data")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select at least one column.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    all_cols = df.columns.tolist()

    ts = TimeSeriesAnalysis(df)

    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        # Time column selection for X-axis
        time_options = all_cols
        default_time_idx = 0
        for i, col in enumerate(time_options):
            col_lower = col.lower()
            if col_lower in ['time', 't', 'timestamp', 'date', 'datetime', 'x']:
                default_time_idx = i
                break
        time_col = st.selectbox("X-axis (Time)", time_options, index=default_time_idx, key="ma_time_col")
        
        # Y column to smooth (exclude time column)
        y_options = [f for f in features if f != time_col]
        if not y_options:
            y_options = features
        ma_column = st.selectbox("Column to Smooth", y_options, key="ma_tab_col")
    
    with col_right:
        st.markdown("**Window Sizes**")
        # Simple text input for custom windows - user can type directly
        window_input = st.text_input(
            "Enter window sizes (comma-separated)", 
            value="5, 10, 20, 50",
            key="ma_windows_input",
            help="Enter window sizes separated by commas, e.g. '5, 10, 20, 50, 100'"
        )
        
        # Parse windows
        try:
            windows = [int(w.strip()) for w in window_input.split(',') if w.strip().isdigit()]
            windows = [w for w in windows if w >= 2]  # Filter valid windows
        except:
            windows = [5, 10, 20]
        
        if not windows:
            st.warning("Enter valid window sizes (integers >= 2)")
            windows = [5, 10, 20]
        
        st.caption(f"Using windows: {windows}")
    
    if st.button("📈 Compute Moving Averages", key="compute_ma_tab", type="primary"):
        with st.spinner("Computing moving averages..."):
            results = ts.moving_average(ma_column, windows)
            # Store time column data for plotting
            series = df[ma_column].dropna()
            results['time_data'] = df[time_col].loc[series.index].values.tolist()
            results['time_col'] = time_col
            st.session_state.analysis_results['ma_tab'] = results
            
            if 'error' in results:
                st.error(f"Moving Average failed: {results['error']}")

    # Display MA results
    if 'ma_tab' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ma_tab']
        if 'error' not in results:
            original = results.get('original', [])
            ma_dict = results.get('moving_averages', {})
            column_name = results.get('column', ma_column)
            # Get the time data for proper x-axis
            x_data = results.get('time_data', results.get('index', list(range(len(original)))))
            x_label = results.get('time_col', 'Index')
            
            if not ma_dict:
                st.warning("No moving averages computed.")
            else:
                # Plot - Original signal with higher opacity
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_data, y=original, mode='lines', name='Original',
                    line=dict(color='lightgray', width=1), opacity=0.5
                ))
                
                # Sort windows for consistent coloring (smaller = lighter)
                sorted_windows = sorted(ma_dict.keys())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                
                for i, window in enumerate(sorted_windows):
                    ma_data = ma_dict[window]
                    values = ma_data.get('values', []) if isinstance(ma_data, dict) else ma_data
                    # Convert to numpy array and handle NaN properly
                    values_arr = np.array(values, dtype=float)
                    fig.add_trace(go.Scatter(
                        x=x_data, y=values_arr, mode='lines', 
                        name=f'MA({window})',
                        line=dict(color=colors[i % len(colors)], width=2),
                        connectgaps=False  # Don't connect across NaN values
                    ))
                
                fig.update_layout(
                    title=f'Moving Averages: {column_name}',
                    xaxis_title=x_label,
                    yaxis_title=column_name,
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📊 Smoothing Statistics")
                stats_data = []
                for window, ma_data in ma_dict.items():
                    values = ma_data.get('values', []) if isinstance(ma_data, dict) else ma_data
                    valid_values = [v for v in values if v is not None and not np.isnan(v)]
                    if valid_values:
                        stats_data.append({
                            'Window': window,
                            'Mean': np.mean(valid_values),
                            'Std': np.std(valid_values),
                            'Min': np.min(valid_values),
                            'Max': np.max(valid_values),
                            'Valid Points': len(valid_values)
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Export
                st.markdown("---")
                st.subheader("📥 Export Smoothed Data")
                
                export_df = pd.DataFrame({'Original': original})
                for window, ma_data in ma_dict.items():
                    values = ma_data.get('values', []) if isinstance(ma_data, dict) else ma_data
                    export_df[f'MA_{window}'] = values
                
                csv_ma = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Moving Averages (CSV)",
                    data=csv_ma,
                    file_name=f"moving_averages_{column_name}.csv",
                    mime="text/csv",
                    key="download_ma_tab"
                )


# =============================================================================
# CAUSALITY TAB
# =============================================================================
def render_causality_tab():
    """Render causality analysis tab - delegates to module version"""
    _render_causality_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🔗 Causality Analysis (Granger Causality)")
    st.caption("Test predictive causality with Granger causality tests and lead-lag correlation analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("⚠️ Please select feature and target columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col

    causality = CausalityAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        max_lag = st.slider("Max Lag", 1, 20, 10, key='causality_lag')
        selected_feature = st.selectbox("Test Feature", features)

    with col2:
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔬 Granger Causality", width='stretch'):
                with st.spinner("Testing..."):
                    # Correct API: granger_causality([features], target, max_lag)
                    results = causality.granger_causality([selected_feature], target, max_lag)
                    st.session_state.analysis_results['granger'] = results.get(selected_feature, {})
                    st.session_state.analysis_results['granger_feature'] = selected_feature

        with col_b:
            if st.button("⏱️ Lead-Lag Analysis", width='stretch'):
                with st.spinner("Computing..."):
                    # Correct API: lead_lag_analysis([features], target, max_lag)
                    results = causality.lead_lag_analysis([selected_feature], target, max_lag)
                    st.session_state.analysis_results['lead_lag'] = results.get(selected_feature, {})
                    st.session_state.analysis_results['lead_lag_feature'] = selected_feature

    st.markdown("---")

    if 'granger' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['granger']
        feat = st.session_state.analysis_results.get('granger_feature', 'Feature')

        if 'error' in results:
            st.error(results['error'])
        elif results:
            st.subheader("🔬 Granger Causality Test")
            st.write(f"Testing: Does **{feat}** Granger-cause **{target}**?")

            granger_df = pd.DataFrame([
                {
                    'Lag': lag,
                    'p-value': data.get('ssr_ftest_pvalue', 0),
                    'Significant': '✅ Yes' if data.get('is_significant', False) else '❌ No'
                }
                for lag, data in results.items() if isinstance(data, dict)
            ])
            st.dataframe(granger_df, width='stretch')

            # Plot p-values
            lags = [lag for lag in results.keys() if isinstance(results[lag], dict)]
            pvals = [results[lag].get('ssr_ftest_pvalue', 0) for lag in lags]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lags, y=pvals, mode='lines+markers'))
            fig.add_hline(y=0.05, line_dash='dash', line_color='red', annotation_text='p=0.05')
            fig.update_layout(title='Granger Causality p-values', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, width='stretch')

    if 'lead_lag' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['lead_lag']
        feat = st.session_state.analysis_results.get('lead_lag_feature', 'Feature')

        if results:
            st.subheader("⏱️ Lead-Lag Analysis")

            col1, col2 = st.columns(2)
            col1.metric("Best Lag", results.get('best_lag', 0))
            col2.metric("Max Correlation", f"{results.get('best_correlation', 0):.4f}")

            best_lag = results.get('best_lag', 0)
            if best_lag < 0:
                st.info(f"📊 **{feat}** leads **{target}** by {abs(best_lag)} periods")
            elif best_lag > 0:
                st.info(f"📊 **{target}** leads **{feat}** by {best_lag} periods")
            else:
                st.info(f"📊 **{feat}** and **{target}** move together")

            # Plot correlations
            lags = results.get('lags', [])
            corrs = results.get('correlations', [])

            fig = go.Figure()
            fig.add_trace(go.Bar(x=lags, y=corrs, marker_color='steelblue'))
            fig.add_vline(x=best_lag, line_dash='dash', line_color='red')
            fig.update_layout(title='Cross-correlation at Different Lags', template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, width='stretch')


# =============================================================================
# VISUALIZATION TAB
# =============================================================================
def render_visualization_tab():
    """Render visualization tab - delegates to module version"""
    _render_visualization_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("📈 Visualization & Plots")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    col1, col2 = st.columns([1, 2])

    with col1:
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Scatter Matrix", "Correlation Heatmap", "Box Plots",
             "Distribution Plots", "3D Scatter", "Parallel Coordinates",
             "Linear Regression Plot (with Statistics)"]
        )

        if plot_type == "Correlation Heatmap":
            corr_method = st.selectbox("Method", ['pearson', 'spearman', 'kendall'])

        if plot_type == "3D Scatter" and len(features) >= 3:
            x_3d = st.selectbox("X axis", features, index=0)
            y_3d = st.selectbox("Y axis", features, index=min(1, len(features)-1))
            z_3d = st.selectbox("Z axis", features, index=min(2, len(features)-1))

        if plot_type == "Linear Regression Plot (with Statistics)":
            st.markdown("**Select variables for regression:**")
            x_reg = st.selectbox("X variable (independent)", features, index=0, key="reg_x")
            available_y = [f for f in features if f != x_reg]
            if st.session_state.target_col and st.session_state.target_col not in available_y:
                available_y.append(st.session_state.target_col)
            if available_y:
                y_reg = st.selectbox("Y variable (dependent)", available_y, index=0, key="reg_y")
            else:
                y_reg = st.selectbox("Y variable (dependent)", features, index=min(1, len(features)-1), key="reg_y")
            show_ci = st.checkbox("Show 95% Confidence Interval", value=True, key="reg_ci")

    if st.button("📊 Generate Plot", width='stretch'):
        with st.spinner("Creating visualization..."):

            if plot_type == "Scatter Matrix":
                fig = px.scatter_matrix(
                    df[features[:5]],
                    title="Scatter Matrix (Interactive!)",
                    template=PLOTLY_TEMPLATE
                )
                fig.update_traces(diagonal_visible=True)
                fig.update_layout(height=800)
                st.plotly_chart(fig, width='stretch')

            elif plot_type == "Correlation Heatmap":
                corr = df[features].corr(method=corr_method)
                fig = px.imshow(
                    corr,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f'{corr_method.capitalize()} Correlation Heatmap'
                )
                fig.update_layout(height=600, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, width='stretch')

            elif plot_type == "Box Plots":
                box_data = df[features].melt(var_name='Feature', value_name='Value')
                fig = px.box(box_data, x='Feature', y='Value',
                           title='Box Plots', template=PLOTLY_TEMPLATE, points='outliers')
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

            elif plot_type == "Distribution Plots":
                for col in features[:4]:
                    fig = px.histogram(df, x=col, marginal='box',
                                      title=f'Distribution of {col}', template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')

            elif plot_type == "3D Scatter":
                if len(features) >= 3:
                    fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                                       title=f'3D Scatter', template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Need at least 3 features")

            elif plot_type == "Parallel Coordinates":
                fig = px.parallel_coordinates(
                    df[features],
                    title='Parallel Coordinates',
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

            elif plot_type == "Linear Regression Plot (with Statistics)":
                # Import scipy for regression statistics
                from scipy import stats as scipy_stats

                # Get data
                x_data = df[x_reg].dropna()
                y_data = df[y_reg].dropna()

                # Align indices
                common_idx = x_data.index.intersection(y_data.index)
                x_data = x_data.loc[common_idx].values
                y_data = y_data.loc[common_idx].values

                if len(x_data) < 3:
                    st.error("Need at least 3 data points for regression.")
                else:
                    # Perform linear regression using scipy.stats.linregress
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
                    r_squared = r_value ** 2

                    # Create regression line
                    x_line = np.array([x_data.min(), x_data.max()])
                    y_line = slope * x_line + intercept

                    # Create figure
                    fig = go.Figure()

                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_data,
                        mode='markers',
                        name='Data Points',
                        marker=dict(color='steelblue', size=8, opacity=0.7)
                    ))

                    # Regression line
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        name=f'Regression Line',
                        line=dict(color='red', width=2)
                    ))

                    # Add confidence interval if requested
                    if show_ci:
                        n = len(x_data)
                        x_mean = np.mean(x_data)
                        ss_x = np.sum((x_data - x_mean) ** 2)
                        y_pred = slope * x_data + intercept
                        residuals = y_data - y_pred
                        mse = np.sum(residuals ** 2) / (n - 2)
                        se = np.sqrt(mse)

                        # For confidence interval
                        t_val = scipy_stats.t.ppf(0.975, n - 2)

                        # Calculate CI at many points for smooth band
                        x_ci = np.linspace(x_data.min(), x_data.max(), 100)
                        y_ci = slope * x_ci + intercept
                        se_fit = se * np.sqrt(1/n + (x_ci - x_mean)**2 / ss_x)
                        ci_upper = y_ci + t_val * se_fit
                        ci_lower = y_ci - t_val * se_fit

                        # Add CI band
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([x_ci, x_ci[::-1]]),
                            y=np.concatenate([ci_upper, ci_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                            name='95% Confidence Interval',
                            hoverinfo='skip'
                        ))

                    # Format equation and stats
                    sign = '+' if intercept >= 0 else '-'
                    eq_str = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"

                    # Update layout with stats annotation
                    fig.update_layout(
                        title=dict(
                            text=f'Linear Regression: {y_reg} vs {x_reg}<br>' +
                                 f'<span style="font-size:14px">{eq_str} | R² = {r_squared:.4f} | p = {p_value:.2e}</span>',
                            font=dict(size=16)
                        ),
                        xaxis_title=f'{x_reg} (Independent Variable)',
                        yaxis_title=f'{y_reg} (Dependent Variable)',
                        template=PLOTLY_TEMPLATE,
                        height=550,
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Display statistics in a clear table
                    st.markdown("### 📊 Regression Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Slope (m)", f"{slope:.6f}")
                    col2.metric("Intercept (b)", f"{intercept:.6f}")
                    col3.metric("R² (R-squared)", f"{r_squared:.4f}")
                    col4.metric("p-value", f"{p_value:.2e}")

                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Correlation (r)", f"{r_value:.4f}")
                    col6.metric("Std Error (slope)", f"{std_err:.6f}")
                    col7.metric("N (samples)", f"{len(x_data)}")
                    col8.metric("Degrees of Freedom", f"{len(x_data) - 2}")

                    # Interpretation
                    st.markdown("### 📖 Interpretation")

                    interpretation = []

                    # Slope interpretation
                    if slope > 0:
                        interpretation.append(f"- **Positive relationship**: For each 1-unit increase in {x_reg}, {y_reg} increases by {slope:.4f} units.")
                    else:
                        interpretation.append(f"- **Negative relationship**: For each 1-unit increase in {x_reg}, {y_reg} decreases by {abs(slope):.4f} units.")

                    # R² interpretation
                    if r_squared >= 0.9:
                        interpretation.append(f"- **Excellent fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance in {y_reg} is explained by {x_reg}.")
                    elif r_squared >= 0.7:
                        interpretation.append(f"- **Good fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance is explained.")
                    elif r_squared >= 0.5:
                        interpretation.append(f"- **Moderate fit**: R² = {r_squared:.4f} means {r_squared*100:.1f}% of variance is explained.")
                    else:
                        interpretation.append(f"- **Weak fit**: R² = {r_squared:.4f} means only {r_squared*100:.1f}% of variance is explained. Consider non-linear models.")

                    # p-value interpretation
                    if p_value < 0.001:
                        interpretation.append(f"- **Highly significant**: p = {p_value:.2e} (p < 0.001). Very strong evidence of a relationship.")
                    elif p_value < 0.01:
                        interpretation.append(f"- **Very significant**: p = {p_value:.4f} (p < 0.01). Strong evidence of a relationship.")
                    elif p_value < 0.05:
                        interpretation.append(f"- **Significant**: p = {p_value:.4f} (p < 0.05). Moderate evidence of a relationship.")
                    else:
                        interpretation.append(f"- **Not significant**: p = {p_value:.4f} (p ≥ 0.05). Insufficient evidence of a linear relationship.")

                    for line in interpretation:
                        st.markdown(line)


# =============================================================================
# CLUSTERING TAB (NEW)
# =============================================================================
def render_clustering_tab():
    """Render clustering analysis tab - delegates to module version"""
    _render_clustering_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🎯 Clustering Analysis")
    st.caption("K-Means, Hierarchical (Agglomerative), DBSCAN, and Gaussian Mixture Model (GMM) clustering")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    st.subheader("Clustering Methods Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox("Method", ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"])

    with col2:
        if method == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            n_init = st.slider("Initializations", 5, 20, 10)
        elif method == "Hierarchical":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        elif method == "DBSCAN":
            eps = st.slider("Eps", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 20, 5)
        elif method == "Gaussian Mixture":
            n_clusters = st.slider("Number of Components", 2, 10, 3)
            cov_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])

    with col3:
        if st.button("🎯 Run Clustering", width='stretch'):
            with st.spinner("Clustering..."):
                try:
                    if method == "K-Means":
                        results = ml.kmeans_clustering(features, n_clusters=n_clusters, n_init=n_init)
                    elif method == "Hierarchical":
                        results = ml.hierarchical_clustering(features, n_clusters=n_clusters, linkage_method=linkage)
                    elif method == "DBSCAN":
                        results = ml.dbscan_clustering(features, eps=eps, min_samples=min_samples)
                    elif method == "Gaussian Mixture":
                        results = ml.gaussian_mixture_model(features, n_components=n_clusters, covariance_type=cov_type)

                    st.session_state.analysis_results['clustering'] = results
                    st.success("✅ Clustering complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if 'clustering' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['clustering']

        if 'error' not in results:
            st.subheader("📊 Clustering Results")

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{results.get('silhouette_score', 0):.4f}")
            col2.metric("Davies-Bouldin", f"{results.get('davies_bouldin_score', 0):.4f}")
            col3.metric("Calinski-Harabasz", f"{results.get('calinski_harabasz_score', 0):.4f}")

            # Cluster visualization (prefer model's 2D projection if available)
            clusters = results.get('clusters', [])
            X_vis = results.get('X_vis')

            if X_vis is not None and hasattr(X_vis, 'shape') and X_vis.shape[1] >= 2:
                fig = px.scatter(x=X_vis[:, 0], y=X_vis[:, 1], color=clusters,
                                 title=f'{method} Clustering Results (2D projection)',
                                 labels={'x': 'Component 1', 'y': 'Component 2'},
                                 template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            elif len(features) >= 2:
                # Fall back to raw feature scatter - use cleaned data matching cluster length
                df_clean = df[features].dropna()
                n_clusters_len = len(clusters) if hasattr(clusters, '__len__') else 0
                if n_clusters_len > 0 and n_clusters_len <= len(df_clean):
                    x_data = df_clean.iloc[:n_clusters_len][features[0]]
                    y_data = df_clean.iloc[:n_clusters_len][features[1] if len(features) > 1 else features[0]]
                else:
                    x_data = df_clean[features[0]]
                    y_data = df_clean[features[1] if len(features) > 1 else features[0]]

                fig = px.scatter(
                    x=x_data, y=y_data,
                    color=clusters[:len(x_data)] if len(clusters) > len(x_data) else clusters,
                    title=f'{method} Clustering Results (feature space)',
                    labels={f'x': features[0], f'y': features[1] if len(features) > 1 else features[0]},
                    template=PLOTLY_TEMPLATE
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

            # Export clustering results
            st.subheader("📥 Export Clustering Results")
            df_clean = df[features].dropna()
            # Ensure cluster array length matches the cleaned dataframe
            n_clusters_len = len(clusters) if hasattr(clusters, '__len__') else 0
            if n_clusters_len > 0 and n_clusters_len <= len(df_clean):
                df_cluster_results = df_clean.iloc[:n_clusters_len].copy()
                df_cluster_results['Cluster'] = clusters
            else:
                # Fallback: if lengths don't match, use what we have
                df_cluster_results = df_clean.copy()
                df_cluster_results['Cluster'] = np.resize(clusters, len(df_clean)) if n_clusters_len > 0 else -1

            csv_clusters = df_cluster_results.to_csv(index=True)
            st.download_button(
                label="📥 Download Clustering Results (CSV)",
                data=csv_clusters,
                file_name="clustering_results.csv",
                mime="text/csv"
            )


# =============================================================================
# ANOMALY DETECTION TAB (NEW)
# =============================================================================
def render_anomaly_tab():
    """Render anomaly detection tab - delegates to module version"""
    _render_anomaly_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🚨 Anomaly Detection")
    st.caption("Detect outliers using Isolation Forest, LOF (Local Outlier Factor), and MCD (Minimum Covariance Determinant)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox("Method", ["Isolation Forest", "Local Outlier Factor", "Minimum Covariance"])
        contamination = st.slider("Contamination (% anomalies)", 0.01, 0.5, 0.1)

    with col2:
        if method == "Isolation Forest":
            n_estimators = st.slider("N Estimators", 50, 500, 100)

    with col3:
        if st.button("🚨 Detect Anomalies", width='stretch'):
            with st.spinner("Detecting anomalies..."):
                try:
                    if method == "Isolation Forest":
                        results = ml.isolation_forest_anomaly(features, contamination=contamination, n_estimators=n_estimators)
                    elif method == "Local Outlier Factor":
                        results = ml.local_outlier_factor(features, contamination=contamination)
                    elif method == "Minimum Covariance":
                        results = ml.minimum_covariance_determinant(features, contamination=contamination)

                    st.session_state.analysis_results['anomaly'] = results
                    st.success("✅ Detection complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if 'anomaly' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['anomaly']

        if 'error' not in results:
            anomaly_labels = results.get('anomaly_labels', [])
            n_anomalies = sum(1 for x in anomaly_labels if x == -1)
            n_normal = sum(1 for x in anomaly_labels if x == 1)
            pct_anomalies = (n_anomalies / len(anomaly_labels) * 100) if len(anomaly_labels) > 0 else 0

            st.subheader("📊 Anomaly Detection Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Anomalies Found", n_anomalies)
            col2.metric("🟢 Normal Points", n_normal)
            col3.metric("Anomaly Rate", f"{pct_anomalies:.1f}%")

            # Visualization
            if len(features) >= 2:
                # Use the same rows the model saw: drop rows with any NaN in selected features
                X = df[features].dropna().reset_index(drop=True)

                if X.shape[0] == 0:
                    st.warning("No complete rows available for plotting (NA values present).")
                else:
                    x_vals = X[features[0]].values
                    y_vals = X[features[1] if len(features) > 1 else features[0]].values

                    # Prefer predictions key from model results, fall back to anomaly_labels
                    preds = None
                    for key in ('predictions', 'anomaly_labels', 'preds'):
                        if key in results:
                            preds = results[key]
                            break

                    if preds is None or len(preds) != len(X):
                        labels = ['Unknown'] * len(X)
                    else:
                        try:
                            preds_list = list(preds)
                            labels = ['🔴 Anomaly' if int(p) == -1 else '🟢 Normal' for p in preds_list[:len(X)]]
                        except Exception:
                            labels = ['Unknown'] * len(X)

                    df_plot = pd.DataFrame({
                        features[0]: x_vals,
                        features[1]: y_vals,
                        'Status': labels
                    })

                    # Add original index for reference
                    df_plot['Original_Index'] = X.index

                    # Create figure with anomalies more prominent
                    fig = go.Figure()

                    # Plot normal points first (smaller, less prominent)
                    normal_mask = df_plot['Status'] == '🟢 Normal'
                    if normal_mask.any():
                        fig.add_trace(go.Scatter(
                            x=df_plot.loc[normal_mask, features[0]],
                            y=df_plot.loc[normal_mask, features[1]],
                            mode='markers',
                            name='Normal',
                            marker=dict(size=6, color='green', opacity=0.4),
                            hovertemplate=f'{features[0]}: %{{x:.3f}}<br>{features[1]}: %{{y:.3f}}<extra>Normal</extra>'
                        ))

                    # Plot anomalies on top (larger, more prominent)
                    anomaly_mask = df_plot['Status'] == '🔴 Anomaly'
                    if anomaly_mask.any():
                        fig.add_trace(go.Scatter(
                            x=df_plot.loc[anomaly_mask, features[0]],
                            y=df_plot.loc[anomaly_mask, features[1]],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(size=12, color='red', symbol='x', line=dict(width=2, color='darkred')),
                            hovertemplate=f'{features[0]}: %{{x:.3f}}<br>{features[1]}: %{{y:.3f}}<extra>⚠️ ANOMALY</extra>'
                        ))

                    fig.update_layout(
                        title=f'{method} Anomaly Detection Results',
                        xaxis_title=features[0],
                        yaxis_title=features[1],
                        template=PLOTLY_TEMPLATE,
                        height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Show anomaly details table
                    st.subheader("🔍 Anomaly Details")

                    # Create full results dataframe
                    df_results = X.copy()
                    df_results['Status'] = labels
                    df_results['Anomaly_Score'] = results.get('scores', [0] * len(X))

                    # Show anomalies table
                    df_anomalies = df_results[df_results['Status'] == '🔴 Anomaly'].copy()
                    if len(df_anomalies) > 0:
                        st.write(f"**Showing {len(df_anomalies)} anomalies:**")
                        st.dataframe(df_anomalies, width='stretch')
                    else:
                        st.info("No anomalies detected with current settings.")

                    # Export functionality
                    st.subheader("📥 Export Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Export all results
                        csv_all = df_results.to_csv(index=True)
                        st.download_button(
                            label="📥 Download All Results (CSV)",
                            data=csv_all,
                            file_name="anomaly_detection_all.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Export only anomalies
                        if len(df_anomalies) > 0:
                            csv_anomalies = df_anomalies.to_csv(index=True)
                            st.download_button(
                                label="📥 Download Anomalies Only (CSV)",
                                data=csv_anomalies,
                                file_name="anomaly_detection_anomalies.csv",
                                mime="text/csv"
                            )


# =============================================================================
# ADVANCED STATISTICAL TESTS TAB (FULL VERSION - Restored)
# =============================================================================
def render_statistical_tests_tab():
    """Render advanced statistical tests tab - delegates to module version"""
    _render_statistical_tests_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    
    # Fallback inline version
    st.header("🧪 Statistical Hypothesis Tests")
    st.caption("t-tests, ANOVA (One-Way, Two-Way, Repeated-Measures), Post-Hoc, Chi-square, Normality, Correlation")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    stats = StatisticalAnalysis(df)

    st.subheader("Hypothesis Tests")

    test_type = st.selectbox(
        "Test Type",
        ["Compare 2 Groups", "Compare 3+ Groups (ANOVA)", "Two-Way ANOVA", "Repeated-Measures ANOVA",
         "Post-Hoc Tests", "Chi-Square", "Normality", "Correlation"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if test_type == "Compare 2 Groups":
            col1_test = st.selectbox("Column 1", features, key='col1_test')
            col2_test = st.selectbox("Column 2", [f for f in features if f != col1_test], key='col2_test')
            test_subtype = st.radio("Test", ["Independent t-test", "Paired t-test", "Mann-Whitney U"])

            if st.button("🧪 Run Test", use_container_width=True):
                st.info(f"⏳ Running {test_subtype}...")
                with st.spinner(f"Running {test_subtype}..."):
                    try:
                        if test_subtype == "Independent t-test":
                            results = stats.ttest_independent(col1_test, col2_test)
                        elif test_subtype == "Paired t-test":
                            results = stats.ttest_paired(col1_test, col2_test)
                        else:
                            results = stats.mann_whitney_u(col1_test, col2_test)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'ttest'
                        st.success(f"✅ {test_subtype} completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Compare 3+ Groups (ANOVA)":
            st.markdown("**One-Way ANOVA**")
            st.caption("Compare means across 3 or more independent groups")

            anova_cols = st.multiselect(
                "Select Groups (3+ columns)",
                features,
                default=features[:3] if len(features) >= 3 else features,
                key='anova_cols'
            )

            if st.button("🧪 Run One-Way ANOVA", use_container_width=True):
                if len(anova_cols) >= 3:
                    st.info("⏳ Starting One-Way ANOVA calculation...")
                    with st.spinner("Running One-Way ANOVA..."):
                        try:
                            results = stats.anova_oneway(anova_cols)
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'oneway'
                            st.success("✅ One-Way ANOVA completed!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Select at least 3 groups for ANOVA")

        elif test_type == "Two-Way ANOVA":
            st.markdown("**Two-Way Factorial ANOVA**")
            st.caption("Test main effects and interaction of two factors on a continuous outcome.")

            st.info("""📊 **Data Structure Required:**
- **Dependent Variable**: The numeric measurement (e.g., 'score', 'response_time')
- **Factor 1**: First categorical grouping (e.g., 'treatment': control/drug_A/drug_B)
- **Factor 2**: Second categorical grouping (e.g., 'gender': male/female)

📁 Example file: `test_data/twoway_anova_data.csv`""")

            all_cols = list(df.columns)

            data_col = st.selectbox("📈 Dependent Variable (numeric)", features, key='twoway_data')
            factor1 = st.selectbox("🏷️ Factor 1 (categorical)", [c for c in all_cols if c != data_col], key='twoway_f1')
            factor2 = st.selectbox("🏷️ Factor 2 (categorical)", [c for c in all_cols if c not in [data_col, factor1]], key='twoway_f2')

            if data_col and factor1 and factor2:
                groups = df.groupby([factor1, factor2])[data_col].count()
                st.caption(f"Groups: {len(groups)} combinations")

            if st.button("🧪 Run Two-Way ANOVA", use_container_width=True):
                st.info("⏳ Starting Two-Way ANOVA...")
                with st.spinner("Running Two-Way ANOVA..."):
                    try:
                        results = stats.anova_twoway(data_col, factor1, factor2)
                        if 'error' in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'twoway'
                            st.success("✅ Two-Way ANOVA completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Repeated-Measures ANOVA":
            st.markdown("**Repeated-Measures (Within-Subjects) ANOVA**")
            st.caption("Compare conditions when same subjects are measured multiple times.")

            st.info("""📊 **Data Structure Required (Long Format):**
- **Dependent Variable**: The numeric measurement (e.g., 'score')
- **Subject ID Column**: Unique identifier for each subject
- **Within-Subjects Factor**: The condition/time point column

📁 Example file: `test_data/repeated_measures_anova_data.csv`""")

            all_cols = list(df.columns)

            data_col = st.selectbox("📈 Dependent Variable", features, key='rm_data')
            subject_col = st.selectbox("👤 Subject/ID Column", [c for c in all_cols if c != data_col], key='rm_subject')
            within_factor = st.selectbox("⏱️ Within-Subjects Factor", [c for c in all_cols if c not in [data_col, subject_col]], key='rm_within')

            if data_col and subject_col and within_factor:
                n_subjects = df[subject_col].nunique()
                n_conditions = df[within_factor].nunique()
                st.caption(f"Found: {n_subjects} subjects, {n_conditions} conditions")

            if st.button("🧪 Run Repeated-Measures ANOVA", use_container_width=True):
                st.info("⏳ Starting Repeated-Measures ANOVA...")
                with st.spinner("Running Repeated-Measures ANOVA..."):
                    try:
                        results = stats.anova_repeated_measures(data_col, subject_col, within_factor)
                        if 'error' in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'repeated'
                            st.success("✅ Repeated-Measures ANOVA completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Post-Hoc Tests":
            st.markdown("**Post-Hoc Pairwise Comparisons**")
            st.caption("Follow-up tests after significant ANOVA")

            all_cols = list(df.columns)

            data_col = st.selectbox("Dependent Variable (numeric)", features, key='posthoc_data')
            group_col = st.selectbox("Grouping Variable", [c for c in all_cols if c != data_col], key='posthoc_group')

            posthoc_type = st.radio("Post-Hoc Method", ["Tukey's HSD", "Bonferroni Correction"], key='posthoc_type')

            if st.button("🧪 Run Post-Hoc Tests", use_container_width=True):
                st.info("⏳ Computing pairwise comparisons...")
                with st.spinner("Running Post-Hoc tests..."):
                    try:
                        if posthoc_type == "Tukey's HSD":
                            results = stats.posthoc_tukey(data_col, group_col)
                        else:
                            results = stats.posthoc_bonferroni(data_col, group_col)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'posthoc'
                        st.success(f"✅ {posthoc_type} completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Chi-Square":
            st.markdown("**Chi-Square Test of Independence**")
            st.caption("Tests association between two categorical variables")

            all_cols = list(df.columns)

            cat_col1 = st.selectbox("First Categorical Variable", all_cols, key='chi_col1')
            cat_col2 = st.selectbox("Second Categorical Variable", [c for c in all_cols if c != cat_col1], key='chi_col2')

            if cat_col1 and cat_col2:
                st.markdown("**Contingency Table Preview:**")
                contingency = pd.crosstab(df[cat_col1], df[cat_col2])
                st.dataframe(contingency, use_container_width=True)

            if st.button("🧪 Run Chi-Square Test", use_container_width=True):
                with st.spinner("Computing Chi-Square test..."):
                    try:
                        results = stats.chi_square_test(cat_col1, cat_col2)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'chi_square'
                        st.success("✅ Chi-Square test completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Normality":
            st.markdown("**Normality Tests**")
            st.caption("Test if data follows a normal distribution (Shapiro-Wilk, D'Agostino-Pearson)")

            norm_col = st.selectbox("Select Column to Test", features, key='norm_col')
            norm_test_type = st.radio("Test Type", ["Shapiro-Wilk", "D'Agostino-Pearson", "Both"], key='norm_test_type')

            if norm_col:
                data = df[norm_col].dropna()
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data, name='Data', nbinsx=30))
                fig.update_layout(title=f'Distribution of {norm_col}', template=PLOTLY_TEMPLATE, height=300)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🧪 Run Normality Test", use_container_width=True):
                with st.spinner("Testing normality..."):
                    try:
                        from scipy import stats as scipy_stats
                        data = df[norm_col].dropna().values

                        results = {'test': 'Normality Tests', 'column': norm_col}

                        if norm_test_type in ["Shapiro-Wilk", "Both"]:
                            if len(data) <= 5000:
                                stat, p_val = scipy_stats.shapiro(data)
                                results['shapiro_statistic'] = float(stat)
                                results['shapiro_p_value'] = float(p_val)
                                results['shapiro_normal'] = p_val > 0.05
                            else:
                                results['shapiro_warning'] = 'Sample too large for Shapiro-Wilk (n>5000)'

                        if norm_test_type in ["D'Agostino-Pearson", "Both"]:
                            if len(data) >= 20:
                                stat, p_val = scipy_stats.normaltest(data)
                                results['dagostino_statistic'] = float(stat)
                                results['dagostino_p_value'] = float(p_val)
                                results['dagostino_normal'] = p_val > 0.05
                            else:
                                results['dagostino_warning'] = 'Sample too small (n<20)'

                        results['skewness'] = float(scipy_stats.skew(data))
                        results['kurtosis'] = float(scipy_stats.kurtosis(data))
                        results['n_samples'] = len(data)

                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'normality'
                        st.success("✅ Normality test completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        elif test_type == "Correlation":
            st.markdown("**Correlation Significance Tests**")
            st.caption("Test if correlation between two variables is statistically significant")

            corr_col1 = st.selectbox("First Variable", features, key='corr_col1')
            corr_col2 = st.selectbox("Second Variable", [f for f in features if f != corr_col1], key='corr_col2')
            corr_method = st.radio("Correlation Method", ["Pearson", "Spearman", "Kendall"], key='corr_method')

            if corr_col1 and corr_col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[corr_col1], y=df[corr_col2], mode='markers', opacity=0.6))
                fig.update_layout(title=f'{corr_col1} vs {corr_col2}', xaxis_title=corr_col1, yaxis_title=corr_col2,
                                  template=PLOTLY_TEMPLATE, height=300)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("🧪 Run Correlation Test", use_container_width=True):
                st.info("⏳ Computing correlation...")
                with st.spinner("Computing correlation..."):
                    try:
                        from scipy import stats as scipy_stats
                        data1 = df[corr_col1].dropna()
                        data2 = df[corr_col2].dropna()

                        common_idx = data1.index.intersection(data2.index)
                        data1 = data1.loc[common_idx].values
                        data2 = data2.loc[common_idx].values

                        if len(data1) < 3:
                            st.error("Error: Need at least 3 data points")
                            return

                        if corr_method == "Pearson":
                            r, p_val = scipy_stats.pearsonr(data1, data2)
                        elif corr_method == "Spearman":
                            r, p_val = scipy_stats.spearmanr(data1, data2)
                        else:
                            r, p_val = scipy_stats.kendalltau(data1, data2)

                        abs_r = abs(r)
                        if abs_r < 0.1:
                            strength = "negligible"
                        elif abs_r < 0.3:
                            strength = "weak"
                        elif abs_r < 0.5:
                            strength = "moderate"
                        elif abs_r < 0.7:
                            strength = "strong"
                        else:
                            strength = "very strong"

                        direction = "positive" if r > 0 else "negative"

                        results = {
                            'test': f'{corr_method} Correlation',
                            'column1': corr_col1,
                            'column2': corr_col2,
                            'statistic': float(r),
                            'p_value': float(p_val),
                            'significant': p_val < 0.05,
                            'strength': strength,
                            'direction': direction,
                            'n_samples': len(data1),
                            'interpretation': f'{strength.capitalize()} {direction} correlation (r={r:.3f}, p={p_val:.4f})'
                        }

                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'correlation'
                        st.success(f"✅ {corr_method} Correlation completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # =========================================================================
    # RESULTS DISPLAY
    # =========================================================================
    with col2:
        if 'hypothesis_test' in st.session_state.analysis_results:
            st.subheader("Test Results")
            test_results = st.session_state.analysis_results['hypothesis_test']
            anova_type = st.session_state.analysis_results.get('anova_type', None)

            if 'error' in test_results and test_results['error']:
                st.error(f"Error: {test_results['error']}")

            elif anova_type == 'twoway':
                st.markdown(f"**{test_results.get('test', 'Two-Way ANOVA')}**")
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{test_results.get('r_squared', 0):.4f}")
                col2.metric("Adj. R²", f"{test_results.get('adj_r_squared', 0):.4f}")

                st.markdown("**Effects:**")
                effects = test_results.get('effects', {})
                effects_data = []
                for effect_name, effect_info in effects.items():
                    effects_data.append({
                        'Effect': effect_info.get('name', effect_name),
                        'Sum of Squares': f"{effect_info.get('sum_sq', 0):.4f}",
                        'df': effect_info.get('df', 0),
                        'F': f"{effect_info.get('F', 0):.4f}",
                        'p-value': f"{effect_info.get('p_value', 0):.4f}",
                        'Significant': '✅' if effect_info.get('significant') else '❌'
                    })
                if effects_data:
                    st.dataframe(pd.DataFrame(effects_data), use_container_width=True)
                st.info(f"📊 {test_results.get('interpretation', '')}")

            elif anova_type == 'chi_square':
                st.markdown(f"**{test_results.get('test', 'Chi-Square Test')}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("χ² Statistic", f"{test_results.get('statistic', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('p_value', 0):.6f}")
                col3.metric("df", test_results.get('dof', 0))
                if test_results.get('significant'):
                    st.success(f"✅ {test_results.get('interpretation', 'Significant association')}")
                else:
                    st.info(f"❌ {test_results.get('interpretation', 'No significant association')}")

            elif anova_type == 'normality':
                st.markdown(f"**{test_results.get('test', 'Normality Tests')} - {test_results.get('column', '')}**")
                col1, col2 = st.columns(2)
                col1.metric("Skewness", f"{test_results.get('skewness', 0):.4f}")
                col2.metric("Kurtosis", f"{test_results.get('kurtosis', 0):.4f}")
                st.metric("N samples", test_results.get('n_samples', 0))

                if 'shapiro_statistic' in test_results:
                    st.markdown("**Shapiro-Wilk Test:**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("W Statistic", f"{test_results.get('shapiro_statistic', 0):.4f}")
                    col2.metric("p-value", f"{test_results.get('shapiro_p_value', 0):.6f}")
                    if test_results.get('shapiro_normal'):
                        col3.success("✅ Normal")
                    else:
                        col3.warning("❌ Non-normal")

                if 'dagostino_statistic' in test_results:
                    st.markdown("**D'Agostino-Pearson Test:**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("K² Statistic", f"{test_results.get('dagostino_statistic', 0):.4f}")
                    col2.metric("p-value", f"{test_results.get('dagostino_p_value', 0):.6f}")
                    if test_results.get('dagostino_normal'):
                        col3.success("✅ Normal")
                    else:
                        col3.warning("❌ Non-normal")

            elif anova_type == 'correlation':
                st.markdown(f"**{test_results.get('test', 'Correlation Test')}**")
                st.markdown(f"Testing: **{test_results.get('column1', '')}** vs **{test_results.get('column2', '')}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Correlation (r)", f"{test_results.get('statistic', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('p_value', 0):.6f}")
                col3.metric("N samples", test_results.get('n_samples', 0))
                col1, col2 = st.columns(2)
                col1.metric("Strength", test_results.get('strength', '').capitalize())
                col2.metric("Direction", test_results.get('direction', '').capitalize())
                if test_results.get('significant'):
                    st.success(f"✅ {test_results.get('interpretation', 'Significant correlation')}")
                else:
                    st.info(f"❌ Correlation not statistically significant (p ≥ 0.05)")

            elif anova_type == 'repeated':
                st.markdown(f"**{test_results.get('test', 'Repeated-Measures ANOVA')}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("F-statistic", f"{test_results.get('F', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('p_value', 0):.4f}")
                col3.metric("η² (effect size)", f"{test_results.get('partial_eta_squared', 0):.4f}")
                col1, col2 = st.columns(2)
                col1.metric("N subjects", test_results.get('n_subjects', 0))
                col2.metric("N conditions", test_results.get('n_conditions', 0))
                if test_results.get('sphericity_concern'):
                    st.warning("⚠️ Sphericity may be violated")
                if test_results.get('p_value', 1) < 0.05:
                    st.success(f"✅ {test_results.get('interpretation', 'Significant')}")
                else:
                    st.info(f"❌ {test_results.get('interpretation', 'Not significant')}")

            elif anova_type == 'posthoc':
                st.markdown(f"**{test_results.get('test', 'Post-Hoc Tests')}**")
                col1, col2 = st.columns(2)
                col1.metric("Number of Groups", test_results.get('n_groups', 0))
                col2.metric("Comparisons Made", test_results.get('n_comparisons', 0))

                st.markdown("**Group Means:**")
                means_df = pd.DataFrame({
                    'Group': list(test_results.get('group_means', {}).keys()),
                    'Mean': list(test_results.get('group_means', {}).values()),
                    'N': list(test_results.get('group_sizes', {}).values())
                })
                st.dataframe(means_df, use_container_width=True)

                st.markdown("**Pairwise Comparisons:**")
                comparisons = test_results.get('comparisons', [])
                if comparisons:
                    comp_df = pd.DataFrame(comparisons)
                    display_cols = ['group1', 'group2', 'mean_diff', 'p_value', 'significant']
                    if 'p_adjusted' in comp_df.columns:
                        display_cols = ['group1', 'group2', 'mean_diff', 'p_adjusted', 'significant']
                    comp_df_display = comp_df[display_cols].copy()
                    comp_df_display['significant'] = comp_df_display['significant'].map({True: '✅', False: '❌'})
                    st.dataframe(comp_df_display, use_container_width=True)
                st.info(f"📊 {test_results.get('interpretation', '')}")

            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Statistic", f"{test_results.get('statistic', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('p_value', 0):.4f}")
                if test_results.get('p_value', 1) < 0.05:
                    col3.success("✅ Significant (p < 0.05)")
                else:
                    col3.info("❌ Not Significant (p ≥ 0.05)")
                if 'interpretation' in test_results:
                    st.info(f"📊 {test_results['interpretation']}")


# =============================================================================
# FOURIER & WAVELET TAB (NEW)
# =============================================================================
def render_signal_analysis_tab():
    """Render Signal Analysis tab - delegates to module version"""
    _render_signal_analysis_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("🔊 Signal Processing: FFT, PSD & Wavelet Analysis")
    st.caption("FFT (Fast Fourier Transform), PSD (Power Spectral Density), CWT (Continuous Wavelet), DWT (Discrete Wavelet)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select a column.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    ts = TimeSeriesAnalysis(df)

    col1, col2 = st.columns([1, 3])

    with col1:
        # Exclude 'time' column from signal analysis (it's used for sampling rate detection)
        signal_features = [f for f in features if f.lower() != 'time']
        if not signal_features:
            signal_features = features
        selected_col = st.selectbox("Select Time Series Column", signal_features)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["FFT (Fourier)", "Power Spectral Density", "Continuous Wavelet", "Discrete Wavelet",
             "Coherence Analysis", "Cross-Wavelet Transform", "Wavelet Coherence"]
        )
        
        # Second column for bivariate analyses
        if analysis_type in ["Coherence Analysis", "Cross-Wavelet Transform", "Wavelet Coherence"]:
            other_cols = [c for c in signal_features if c != selected_col]
            if other_cols:
                selected_col2 = st.selectbox("Select Second Column", other_cols, key="signal_col2")
            else:
                st.warning("Need at least 2 columns for bivariate analysis")
                selected_col2 = None
        else:
            selected_col2 = None

        # Auto-detect sampling rate from data
        if 'time' in df.columns:
            time_diff = df['time'].diff().dropna()
            if len(time_diff) > 0:
                avg_dt = time_diff.mean()
                sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 1.0
                st.success(f"✅ Sampling rate: **{sampling_rate:.1f} Hz** (auto-detected from 'time' column)")
            else:
                sampling_rate = 1.0
                st.warning("⚠️ Could not detect sampling rate from 'time' column. Using 1.0 Hz default.")
        else:
            # Calculate from number of samples assuming 1 second duration
            n_samples = len(df[selected_col].dropna())
            sampling_rate = float(n_samples)
            st.info(f"📊 No 'time' column found. Assuming {n_samples} samples over 1 second → {sampling_rate:.1f} Hz")

        wavelet_type = st.selectbox("Wavelet Type", ["morl", "mexh", "gaus1", "gaus2", "cgau1"], index=0, help="Select the wavelet function for CWT.")
        cwt_scales = st.slider("CWT Scales (max)", 16, 256, 64, help="Maximum number of scales for CWT.")
        y_scale = st.selectbox("CWT Y-axis scale", ["log", "linear"], index=0, help="Y-axis scale for wavelet power plot.")
        significance_level = st.slider("Significance level", 0.80, 0.999, 0.95, step=0.01, help="Significance threshold for Torrence & Compo plot.")
        show_coi = st.checkbox("Show COI (Cone of Influence)", value=True, help="Display Cone of Influence on wavelet plot.")

    with col2:
        # Data quality check
        n_samples = len(df[selected_col].dropna())
        nyquist_freq = sampling_rate / 2.0

        if n_samples < 100:
            st.warning(f"⚠️ Only {n_samples} samples - may not be enough for reliable frequency analysis. Consider using test_data/signal_analysis_sample.csv")

        st.info(f"📊 {n_samples} samples at **{sampling_rate:.1f} Hz** → Nyquist: {nyquist_freq:.1f} Hz (max detectable frequency)")

        # Clear cache button
        if st.button("🗑️ Clear Cached Results", help="Clear all previous analysis results"):
            st.session_state.analysis_results = {}
            st.success("✅ Cache cleared! Run analysis again.")
            st.rerun()

        # Only show the relevant button for the selected analysis type
        if analysis_type == "FFT (Fourier)":
            if st.button("🔍 FFT Analysis", width='stretch'):
                with st.spinner("Computing FFT from loaded data..."):
                    # Use the actual data from the selected column
                    results = ts.fourier_transform(selected_col, sampling_rate=float(sampling_rate))
                    st.session_state.analysis_results['fft'] = results
                    st.success(f"✅ FFT computed: {n_samples} samples at **{sampling_rate:.1f} Hz** → Dominant: {results.get('dominant_frequency', 0):.2f} Hz")
        elif analysis_type == "Power Spectral Density":
            if st.button("📊 PSD Analysis", width='stretch'):
                with st.spinner("Computing PSD..."):
                    # Use the actual data from the selected column
                    results = ts.power_spectral_density(selected_col, sampling_rate=float(sampling_rate))
                    st.session_state.analysis_results['psd'] = results
                    st.success(f"✅ PSD computed on {len(df[selected_col].dropna())} samples from column '{selected_col}' at {sampling_rate} Hz")
        elif analysis_type == "Continuous Wavelet":
            if st.button("🌊 CWT Analysis", width='stretch'):
                with st.spinner("Computing Continuous Wavelet Transform..."):
                    results = ts.continuous_wavelet_transform(selected_col, scales=None, wavelet=wavelet_type, sampling_rate=float(sampling_rate))
                    if 'error' in results:
                        st.error(f"CWT failed: {results['error']}")
                    else:
                        st.session_state.analysis_results['cwt'] = results
                        st.session_state.analysis_results['cwt_options'] = {
                            'y_scale': y_scale,
                            'significance_level': significance_level,
                            'show_coi': show_coi,
                            'wavelet_type': wavelet_type
                        }
                        power = results.get('power', np.array([]))
                        st.success(f"✅ CWT computed: {power.shape[1]} time points × {power.shape[0]} scales using '{wavelet_type}' wavelet")
        elif analysis_type == "Discrete Wavelet":
            dwt_wavelet_type = st.selectbox("Wavelet Type (Discrete)", ["db4", "db8", "sym4", "coif1", "haar"], index=0, help="Select discrete wavelet for DWT. Common choices: db4 (Daubechies 4), haar (simplest)")
            level = st.slider("Decomposition Level", 1, 5, 3, help="Level of wavelet decomposition.")
            if st.button("🌀 DWT Analysis", width='stretch'):
                with st.spinner("Computing DWT..."):
                    # Use the actual data from the selected column with discrete wavelet
                    results = ts.discrete_wavelet_transform(selected_col, wavelet=dwt_wavelet_type, level=level)
                    st.session_state.analysis_results['dwt'] = results
                    st.session_state.analysis_results['dwt_wavelet'] = dwt_wavelet_type
                    st.success(f"✅ DWT computed on {len(df[selected_col].dropna())} samples from column '{selected_col}'")
        
        # Bivariate analyses (require second column)
        elif analysis_type == "Coherence Analysis":
            if selected_col2:
                if st.button("📊 Coherence Analysis", width='stretch'):
                    with st.spinner("Computing coherence..."):
                        results = ts.coherence_analysis(selected_col, selected_col2, sampling_rate=float(sampling_rate))
                        if 'error' in results:
                            st.error(f"Coherence failed: {results['error']}")
                        else:
                            st.session_state.analysis_results['coherence'] = results
                            st.success(f"✅ Coherence computed between '{selected_col}' and '{selected_col2}'")
            else:
                st.warning("Select a second column for coherence analysis")
        
        elif analysis_type == "Cross-Wavelet Transform":
            if selected_col2:
                xwt_wavelet = st.selectbox("XWT Wavelet", ["cmor1.5-1.0", "morl", "mexh"], index=0, 
                                           help="Complex Morlet recommended for phase analysis")
                if st.button("🌊 Cross-Wavelet Transform", width='stretch'):
                    with st.spinner("Computing Cross-Wavelet Transform..."):
                        results = ts.cross_wavelet_transform(selected_col, selected_col2, 
                                                              wavelet=xwt_wavelet, sampling_rate=float(sampling_rate))
                        if 'error' in results:
                            st.error(f"XWT failed: {results['error']}")
                        else:
                            st.session_state.analysis_results['xwt'] = results
                            st.session_state.analysis_results['xwt_options'] = {
                                'y_scale': y_scale,
                                'significance_level': significance_level,
                                'show_coi': show_coi,
                                'wavelet_type': xwt_wavelet,
                                'col1': selected_col,
                                'col2': selected_col2
                            }
                            st.success(f"✅ XWT computed between '{selected_col}' and '{selected_col2}'")
            else:
                st.warning("Select a second column for cross-wavelet transform")
        
        elif analysis_type == "Wavelet Coherence":
            if selected_col2:
                wtc_wavelet = st.selectbox("WTC Wavelet", ["cmor1.5-1.0", "morl", "mexh"], index=0,
                                           help="Complex Morlet recommended for phase analysis")
                smooth_factor = st.slider("Smoothing Factor", 3, 15, 5, help="Smoothing window for coherence estimation")
                if st.button("🔗 Wavelet Coherence", width='stretch'):
                    with st.spinner("Computing Wavelet Coherence..."):
                        results = ts.wavelet_coherence(selected_col, selected_col2, 
                                                        wavelet=wtc_wavelet, sampling_rate=float(sampling_rate),
                                                        smooth_factor=smooth_factor)
                        if 'error' in results:
                            st.error(f"WTC failed: {results['error']}")
                        else:
                            st.session_state.analysis_results['wtc'] = results
                            st.session_state.analysis_results['wtc_options'] = {
                                'y_scale': y_scale,
                                'significance_level': significance_level,
                                'show_coi': show_coi,
                                'wavelet_type': wtc_wavelet,
                                'col1': selected_col,
                                'col2': selected_col2
                            }
                            st.success(f"✅ Wavelet Coherence computed between '{selected_col}' and '{selected_col2}'")
            else:
                st.warning("Select a second column for wavelet coherence")

    st.markdown("---")
    # ...existing code...

    # Display results
    # Combined FFT and PSD panel plot
    fft_res = st.session_state.analysis_results.get('fft')
    psd_res = st.session_state.analysis_results.get('psd')
    if fft_res and psd_res and ('error' not in fft_res) and ('error' not in psd_res):
        st.subheader("🔍 FFT & 📊 Power Spectral Density (Combined)")
        col1, col2 = st.columns(2)
        col1.metric("Dominant Frequency (FFT)", f"{fft_res.get('dominant_frequency', 0):.4f}")
        col2.metric("Dominant Frequency (PSD)", f"{psd_res.get('dominant_frequency', 0):.4f}")

        # Prepare subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("FFT Magnitude Spectrum", "Power Spectral Density"))
        # FFT panel
        frequencies = fft_res.get('frequencies', [])
        magnitude = fft_res.get('magnitude', [])
        if len(frequencies) > 0 and len(magnitude) > 0:
            fig.add_trace(
                go.Scatter(x=frequencies[:len(frequencies)//2], y=magnitude[:len(magnitude)//2],
                           mode='lines', fill='tozeroy', name='FFT'),
                row=1, col=1
            )
        # PSD panel
        psd_freq = psd_res.get('frequencies', [])
        psd_vals = psd_res.get('power_spectral_density', [])
        if len(psd_freq) > 0 and len(psd_vals) > 0:
            fig.add_trace(
                go.Scatter(x=psd_freq, y=psd_vals, mode='lines', fill='tozeroy', name='PSD', line=dict(color='orange')),
                row=2, col=1
            )
        fig.update_layout(height=700, template=PLOTLY_TEMPLATE)
        fig.update_xaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        st.plotly_chart(fig, width='stretch')

    else:
        # Fallback: show FFT or PSD individually if only one is present
        if fft_res and ('error' not in fft_res):
            st.subheader("🔍 FFT Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Dominant Frequency (Hz)", f"{fft_res.get('dominant_frequency', 0):.2f}")
            col2.metric("Peak Power", f"{fft_res.get('peak_power', 0):.2e}")

            # Plot FFT spectrum
            frequencies = fft_res.get('positive_frequencies', [])
            magnitude = fft_res.get('magnitude', [])

            if len(frequencies) > 0 and len(magnitude) > 0:
                # Use only positive frequencies for cleaner plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frequencies,
                    y=magnitude,
                    mode='lines',
                    fill='tozeroy',
                    name='FFT Magnitude',
                    line=dict(color='steelblue')
                ))
                fig.update_layout(
                    title='FFT Magnitude Spectrum (Positive Frequencies)',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Magnitude',
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')

                # Show top frequencies
                st.write("**Top 5 Dominant Frequencies:**")
                top_freqs = fft_res.get('dominant_frequencies', [])
                top_powers = fft_res.get('dominant_powers', [])
                for i, (f, p) in enumerate(zip(top_freqs[:5], top_powers[:5]), 1):
                    st.write(f"{i}. {f:.2f} Hz - Power: {p:.2e}")
        if psd_res and ('error' not in psd_res):
            st.subheader("📊 Power Spectral Density")
            col1, col2 = st.columns(2)
            col1.metric("Dominant Frequency", f"{psd_res.get('dominant_frequency', 0):.4f}")
            col2.metric("Total Power", f"{psd_res.get('total_power', 0):.4f}")
            psd_freq = psd_res.get('frequencies', [])
            psd_vals = psd_res.get('power_spectral_density', [])
            if len(psd_freq) > 0 and len(psd_vals) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=psd_freq, y=psd_vals, mode='lines', fill='tozeroy', line=dict(color='orange')))
                fig.update_layout(title='Power Spectral Density', xaxis_title='Frequency',
                                yaxis_title='Power', template=PLOTLY_TEMPLATE, height=400)
                st.plotly_chart(fig, width='stretch')

    # Export FFT/PSD results
    if fft_res or psd_res:
        st.subheader("📥 Export Spectral Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            if fft_res and 'error' not in fft_res:
                fft_df = pd.DataFrame({
                    'Frequency_Hz': fft_res.get('positive_frequencies', []),
                    'Magnitude': fft_res.get('magnitude', []),
                    'Power': fft_res.get('power', [])
                })
                csv_fft = fft_df.to_csv(index=False)
                st.download_button(
                    label="📥 FFT Results (CSV)",
                    data=csv_fft,
                    file_name="fft_results.csv",
                    mime="text/csv"
                )

        with col2:
            if psd_res and 'error' not in psd_res:
                psd_df = pd.DataFrame({
                    'Frequency_Hz': psd_res.get('frequencies', []),
                    'Power_Spectral_Density': psd_res.get('power_spectral_density', [])
                })
                csv_psd = psd_df.to_csv(index=False)
                st.download_button(
                    label="📥 PSD Results (CSV)",
                    data=csv_psd,
                    file_name="psd_results.csv",
                    mime="text/csv"
                )

    if 'cwt' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['cwt']
        if 'error' not in results:
            st.subheader("🌊 Continuous Wavelet Transform")
            st.info("Time-frequency analysis showing power at each frequency over time")
            try:
                cwt_opts = st.session_state.analysis_results.get('cwt_options', {})
                y_scale_opt = cwt_opts.get('y_scale', 'log')
                signif_opt = cwt_opts.get('significance_level', 0.95)
                show_coi_opt = cwt_opts.get('show_coi', True)
                wavelet_type_opt = cwt_opts.get('wavelet_type', 'morl')

                # Create and display CWT plot
                fig = ts.plot_wavelet_torrence(
                    results,
                    selected_col,
                    y_scale=y_scale_opt,
                    significance_level=signif_opt,
                    show_coi=show_coi_opt,
                    wavelet=wavelet_type_opt
                )
                if fig:
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)

                # Export CWT results
                st.subheader("📥 Export CWT Results")
                col1, col2 = st.columns(2)
                with col1:
                    cwt_summary_df = pd.DataFrame({
                        'Scale': results.get('scales', []),
                        'Period': results.get('periods', []),
                        'Global_Power': np.mean(results.get('power', np.array([[0]])), axis=1)
                    })
                    csv_cwt = cwt_summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CWT Summary (CSV)",
                        data=csv_cwt,
                        file_name="cwt_summary.csv",
                        mime="text/csv"
                    )
                with col2:
                    scale_avg_power = np.mean(results.get('power', np.array([[0]])), axis=0)
                    cwt_time_df = pd.DataFrame({
                        'Time': results.get('time', []),
                        'Scale_Averaged_Power': scale_avg_power
                    })
                    csv_cwt_time = cwt_time_df.to_csv(index=False)
                    st.download_button(
                        label="📥 CWT Time Series (CSV)",
                        data=csv_cwt_time,
                        file_name="cwt_time_series.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"CWT plotting failed: {str(e)}")

    if 'dwt' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['dwt']
        if 'error' not in results:
            st.subheader("🌀 Discrete Wavelet Transform")
            try:
                fig = ts.plot_discrete_wavelet(results, selected_col)
                if fig:
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
                else:
                    st.info("DWT data available but no plot generated")

                # Export DWT results
                st.subheader("📥 Export DWT Results")
                coefficients = results.get('coefficients', [])
                if coefficients:
                    # Create summary of DWT decomposition
                    dwt_summary = []
                    for c in coefficients:
                        dwt_summary.append({
                            'Level': c['level'],
                            'Detail_Length': c['detail_length'],
                            'Approx_Length': c['approximation_length'],
                            'Detail_RMS': np.sqrt(np.mean(np.array(c['detail'])**2)),
                            'Detail_Max': np.max(np.abs(c['detail'])),
                            'Detail_Energy': np.sum(np.array(c['detail'])**2)
                        })
                    dwt_df = pd.DataFrame(dwt_summary)
                    csv_dwt = dwt_df.to_csv(index=False)
                    st.download_button(
                        label="📥 DWT Summary (CSV)",
                        data=csv_dwt,
                        file_name="dwt_summary.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"DWT plotting failed: {str(e)}")
    
    # =========================================================================
    # Coherence Analysis Display
    # =========================================================================
    if 'coherence' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['coherence']
        if 'error' not in results:
            st.subheader("📊 Magnitude-Squared Coherence")
            st.info("Coherence measures linear correlation as a function of frequency (0 = no correlation, 1 = perfect)")
            
            try:
                frequencies = results.get('frequencies', [])
                coherence = results.get('coherence', [])
                phase = results.get('phase', [])
                
                if len(frequencies) > 0 and len(coherence) > 0:
                    # Create two-panel plot
                    from plotly.subplots import make_subplots
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       subplot_titles=('Coherence', 'Phase'),
                                       vertical_spacing=0.1)
                    
                    # Coherence plot
                    fig.add_trace(go.Scatter(x=frequencies, y=coherence, mode='lines',
                                            name='Coherence', fill='tozeroy',
                                            line=dict(color='steelblue', width=2)), row=1, col=1)
                    fig.add_hline(y=0.5, line_dash='dash', line_color='red', 
                                 annotation_text='0.5 threshold', row=1, col=1)
                    
                    # Phase plot
                    if len(phase) > 0:
                        fig.add_trace(go.Scatter(x=frequencies, y=np.degrees(phase), mode='lines',
                                                name='Phase (degrees)', line=dict(color='orange', width=2)), 
                                     row=2, col=1)
                    
                    fig.update_layout(
                        title='Magnitude-Squared Coherence Analysis',
                        height=500,
                        template=PLOTLY_TEMPLATE,
                        showlegend=True
                    )
                    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
                    fig.update_yaxes(title_text='Coherence', range=[0, 1], row=1, col=1)
                    fig.update_yaxes(title_text='Phase (°)', row=2, col=1)
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Export
                    coh_df = pd.DataFrame({
                        'Frequency_Hz': frequencies,
                        'Coherence': coherence,
                        'Phase_rad': phase if len(phase) > 0 else [np.nan]*len(frequencies)
                    })
                    st.download_button(
                        "📥 Download Coherence (CSV)",
                        coh_df.to_csv(index=False),
                        "coherence_results.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Coherence plotting failed: {str(e)}")
    
    # =========================================================================
    # Cross-Wavelet Transform Display (Torrence & Compo style)
    # =========================================================================
    if 'xwt' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['xwt']
        if 'error' not in results:
            st.subheader("🌊 Cross-Wavelet Transform (XWT)")
            opts = st.session_state.analysis_results.get('xwt_options', {})
            st.info(f"Common power and relative phase between {opts.get('col1', 'signal1')} and {opts.get('col2', 'signal2')}")
            
            try:
                power = results.get('power', np.array([[]]))
                phase = results.get('phase', np.array([[]]))
                periods = results.get('periods', [])
                time = results.get('time', list(range(power.shape[1] if power.ndim > 1 else 0)))
                coi = results.get('coi', [])
                
                if power.size > 0 and len(periods) > 0:
                    y_scale_opt = opts.get('y_scale', 'log')
                    show_coi_opt = opts.get('show_coi', True)
                    
                    # Create XWT power plot (Torrence & Compo style)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Use log2 of periods for y-axis
                    log2_periods = np.log2(periods)
                    
                    # Plot power
                    im = ax.contourf(time, log2_periods, np.log2(power + 1e-10), 
                                     levels=50, cmap='jet', extend='both')
                    
                    # Add phase arrows (subsample for clarity)
                    if phase.size > 0:
                        arrow_step_x = max(1, len(time) // 20)
                        arrow_step_y = max(1, len(periods) // 10)
                        for i in range(0, len(time), arrow_step_x):
                            for j in range(0, len(periods), arrow_step_y):
                                if i < phase.shape[1] and j < phase.shape[0]:
                                    dx = np.cos(phase[j, i]) * 0.3
                                    dy = np.sin(phase[j, i]) * 0.3
                                    ax.annotate('', xy=(time[i]+dx, log2_periods[j]+dy),
                                               xytext=(time[i], log2_periods[j]),
                                               arrowprops=dict(arrowstyle='->', color='white', lw=0.5))
                    
                    # Add COI
                    if show_coi_opt and len(coi) > 0:
                        coi_log2 = np.log2(np.array(coi) + 1e-10)
                        ax.fill_between(time, coi_log2, np.max(log2_periods), 
                                        color='white', alpha=0.3, hatch='/')
                    
                    # Formatting
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Period')
                    ax.set_title(f"Cross-Wavelet Power: {opts.get('col1', 'sig1')} × {opts.get('col2', 'sig2')}")
                    
                    # Set y-ticks to actual period values
                    yticks = np.arange(np.floor(np.min(log2_periods)), np.ceil(np.max(log2_periods))+1)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([f'{2**y:.1f}' for y in yticks])
                    ax.invert_yaxis()
                    
                    plt.colorbar(im, ax=ax, label='log₂(Power)')
                    plt.tight_layout()
                    
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
                    
                    st.caption("**Arrows**: Phase relationship (→ in-phase, ← anti-phase, ↑ signal1 leads, ↓ signal2 leads). **Hatched area**: Cone of Influence (edge effects)")
                    
            except Exception as e:
                st.error(f"XWT plotting failed: {str(e)}")
    
    # =========================================================================
    # Wavelet Coherence Display (Torrence & Compo style)
    # =========================================================================
    if 'wtc' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['wtc']
        if 'error' not in results:
            st.subheader("🔗 Wavelet Coherence (WTC)")
            opts = st.session_state.analysis_results.get('wtc_options', {})
            st.info(f"Time-frequency coherence between {opts.get('col1', 'signal1')} and {opts.get('col2', 'signal2')} (normalized 0-1)")
            
            try:
                coherence = results.get('coherence', np.array([[]]))
                phase = results.get('phase', np.array([[]]))
                periods = results.get('periods', [])
                time = results.get('time', list(range(coherence.shape[1] if coherence.ndim > 1 else 0)))
                coi = results.get('coi', [])
                significance = results.get('significance', None)
                
                if coherence.size > 0 and len(periods) > 0:
                    y_scale_opt = opts.get('y_scale', 'log')
                    show_coi_opt = opts.get('show_coi', True)
                    signif_opt = opts.get('significance_level', 0.95)
                    
                    # Create WTC plot (Torrence & Compo style)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Use log2 of periods for y-axis
                    log2_periods = np.log2(periods)
                    
                    # Plot coherence (0-1 scale)
                    im = ax.contourf(time, log2_periods, coherence, 
                                     levels=np.linspace(0, 1, 21), cmap='RdYlBu_r', extend='both')
                    
                    # Add significance contour
                    if significance is not None:
                        ax.contour(time, log2_periods, significance, levels=[signif_opt],
                                  colors='black', linewidths=1.5, linestyles='solid')
                    
                    # Add phase arrows (only where coherence is high)
                    if phase.size > 0:
                        arrow_step_x = max(1, len(time) // 20)
                        arrow_step_y = max(1, len(periods) // 10)
                        for i in range(0, len(time), arrow_step_x):
                            for j in range(0, len(periods), arrow_step_y):
                                if i < phase.shape[1] and j < phase.shape[0]:
                                    if i < coherence.shape[1] and j < coherence.shape[0] and coherence[j, i] > 0.5:
                                        dx = np.cos(phase[j, i]) * 0.3
                                        dy = np.sin(phase[j, i]) * 0.3
                                        ax.annotate('', xy=(time[i]+dx, log2_periods[j]+dy),
                                                   xytext=(time[i], log2_periods[j]),
                                                   arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
                    
                    # Add COI
                    if show_coi_opt and len(coi) > 0:
                        coi_log2 = np.log2(np.array(coi) + 1e-10)
                        ax.fill_between(time, coi_log2, np.max(log2_periods), 
                                        color='white', alpha=0.3, hatch='/')
                    
                    # Formatting
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Period')
                    ax.set_title(f"Wavelet Coherence: {opts.get('col1', 'sig1')} × {opts.get('col2', 'sig2')}")
                    
                    # Set y-ticks to actual period values
                    yticks = np.arange(np.floor(np.min(log2_periods)), np.ceil(np.max(log2_periods))+1)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([f'{2**y:.1f}' for y in yticks])
                    ax.invert_yaxis()
                    
                    cbar = plt.colorbar(im, ax=ax, label='Coherence')
                    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                    plt.tight_layout()
                    
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
                    
                    st.caption("**Arrows**: Phase relationship (shown only where coherence > 0.5). **Black contour**: Significance boundary. **Hatched area**: COI")
                    
            except Exception as e:
                st.error(f"WTC plotting failed: {str(e)}")


# =============================================================================
# DIMENSIONALITY REDUCTION TAB (NEW)
# =============================================================================
def render_dimreduction_tab():
    """Render dimensionality reduction tab - delegates to module version"""
    _render_dimreduction_tab_module()
    return
    # DEPRECATED: Below code is unreachable
    st.header("📉 Dimensionality Reduction")
    st.caption("PCA, SVD, t-SNE (t-distributed Stochastic Neighbor Embedding), UMAP, and ICA (Independent Component Analysis)")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols

    ml = MLModels(df)

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Reduction Method",
            ["PCA", "t-SNE", "UMAP", "SVD", "ICA"]
        )

    with col2:
        max_comp = max(2, min(10, len(features)))
        if max_comp <= 2:
            max_comp = 3  # Ensure slider min < max
        n_components = st.slider("Components", 2, max_comp, 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        if method == "PCA":
            if st.button("🔬 PCA", width='stretch'):
                with st.spinner("Computing PCA..."):
                    results = ml.pca_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['pca_new'] = results

    with col2:
        if method == "t-SNE":
            if st.button("📊 t-SNE", width='stretch'):
                with st.spinner("Computing t-SNE..."):
                    results = ml.tsne_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['tsne'] = results

    with col3:
        if method == "UMAP":
            if st.button("🔷 UMAP", width='stretch'):
                with st.spinner("Computing UMAP..."):
                    results = ml.umap_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['umap'] = results
    # Add SVD and ICA buttons below
    col4, col5 = st.columns(2)
    with col4:
        if method == "SVD":
            if st.button("📐 SVD", width='stretch'):
                with st.spinner("Computing SVD..."):
                    results = ml.svd_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['svd'] = results

    with col5:
        if method == "ICA":
            if st.button("🔀 ICA", width='stretch'):
                with st.spinner("Computing ICA..."):
                    results = ml.ica_analysis(features, n_components=n_components)
                    st.session_state.analysis_results['ica'] = results

    st.markdown("---")

    if 'pca_new' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['pca_new']
        if 'error' not in results:
            st.subheader("🔬 PCA Results with Feature Vectors")
            total_var = results.get('total_variance_explained', 0)
            st.metric("Variance Explained (Total)", f"{total_var*100:.1f}%")

            explained_var = results.get('explained_variance', [])
            if len(explained_var) > 0:
                fig = px.bar(x=[f'PC{i+1}' for i in range(len(explained_var))], y=explained_var,
                            title='Explained Variance per Component', template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, width='stretch')

            # Enhanced Cartesian biplot with feature vectors
            transformed = results.get('transformed_data')
            components = results.get('components')
            feature_names = results.get('feature_names', [])

            if (transformed is not None and hasattr(transformed, 'shape') and transformed.shape[1] >= 2
                and components is not None and feature_names):

                # Create biplot with vectors
                # Note: PCA components are (n_components, n_features), need to transpose for biplot
                # which expects (n_features, n_components)
                try:
                    components_T = components.T  # Transpose: (n_features, n_components)
                    fig_biplot, vector_info = create_pca_biplot_with_vectors(
                        transformed[:, :2],
                        components_T[:, :2] if components_T.shape[1] >= 2 else components_T,
                        explained_var,
                        feature_names,
                        scale_factor=3.0
                    )
                    st.plotly_chart(fig_biplot, width='stretch')

                    # Display insights
                    st.markdown("### 📊 Vector Interpretation Guide")
                    insights = generate_pca_insights(vector_info, explained_var, total_var)
                    st.markdown(insights)

                    # Display detailed vector interpretation
                    with st.expander("🔍 Detailed Vector Analysis"):
                        vector_interp = interpret_vectors(vector_info, feature_names)

                        st.markdown("#### PC Drivers")
                        st.markdown(vector_interp['pc1_drivers'])
                        st.markdown(vector_interp['pc2_drivers'])

                        st.markdown("#### Feature Correlations (Based on Vector Angles)")
                        for corr in vector_interp['correlations']:
                            st.markdown(f"- {corr}")

                        st.markdown("#### Feature Importance")
                        st.markdown(vector_interp['feature_importance'])

                        st.markdown("#### How to Read Vectors")
                        st.markdown("""
                        - **Vector direction**: Shows how feature aligns with PC1/PC2
                        - **Vector length**: Magnitude of contribution (longer = stronger)
                        - **Parallel vectors**: Features are correlated
                        - **Perpendicular vectors**: Features are independent
                        - **Opposite vectors**: Features are negatively correlated
                        """)

                except Exception as e:
                    st.error(f"Error creating PCA biplot: {str(e)}")
                    # Fallback to simple scatter
                    fig_pc = px.scatter(x=transformed[:, 0], y=transformed[:, 1],
                                        title='PCA Projection (PC1 vs PC2)',
                                        labels={'x': 'PC1', 'y': 'PC2'},
                                        template=PLOTLY_TEMPLATE)
                    fig_pc.update_layout(height=500)
                    st.plotly_chart(fig_pc, width='stretch')

    if 'tsne' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['tsne']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("📊 t-SNE Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='t-SNE Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

    if 'umap' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['umap']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("🔷 UMAP Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='UMAP Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

    if 'svd' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['svd']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("📐 SVD Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='SVD Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')

    if 'ica' in st.session_state.analysis_results:
        results = st.session_state.analysis_results['ica']
        if 'error' not in results and 'transformed_data' in results:
            st.subheader("🔀 ICA Visualization")
            data = results['transformed_data']
            if data.shape[1] >= 2:
                fig = px.scatter(x=data[:, 0], y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
                               title='ICA Projection', template=PLOTLY_TEMPLATE)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')


# =============================================================================
# v4.0 NEW FEATURE TABS
# =============================================================================

def render_effect_sizes_tab():
    """Render the effect sizes analysis tab"""
    st.header("📏 Effect Size Analysis")
    st.caption("Calculate standardized effect sizes with confidence intervals for scientific reporting")
    
    df = st.session_state.df
    features = st.session_state.feature_cols
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for effect size calculation.")
        return
    
    calc = EffectSizes(df)
    
    # Initialize session state for effect size results
    if 'effect_size_results' not in st.session_state:
        st.session_state.effect_size_results = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Two-Group Effect Sizes")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        with config_col1:
            group1_col = st.selectbox("Group 1 Column", numeric_cols, key="es_group1")
        with config_col2:
            group2_col = st.selectbox("Group 2 Column", [c for c in numeric_cols if c != group1_col], key="es_group2")
        with config_col3:
            confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01, key="es_conf")
        
        # Calculate all effect sizes at once
        if st.button("📊 Calculate All Effect Sizes", key="calc_all_es", type="primary"):
            with st.spinner("Computing effect sizes..."):
                try:
                    results = []
                    
                    # Cohen's d
                    cohens_result = calc.cohens_d(group1_col, group2_col, confidence_level=confidence)
                    results.append({
                        'Measure': "Cohen's d",
                        'Value': cohens_result['cohens_d'],
                        'CI_Lower': cohens_result['ci_lower'],
                        'CI_Upper': cohens_result['ci_upper'],
                        'Interpretation': cohens_result['interpretation']
                    })
                    
                    # Hedges' g
                    hedges_result = calc.hedges_g(group1_col, group2_col, confidence_level=confidence)
                    results.append({
                        'Measure': "Hedges' g",
                        'Value': hedges_result['hedges_g'],
                        'CI_Lower': hedges_result['ci_lower'],
                        'CI_Upper': hedges_result['ci_upper'],
                        'Interpretation': hedges_result['interpretation']
                    })
                    
                    # Glass's Delta
                    glass_result = calc.glass_delta(group1_col, group2_col)
                    results.append({
                        'Measure': "Glass's Δ",
                        'Value': glass_result['glass_delta'],
                        'CI_Lower': glass_result.get('ci_lower', glass_result['glass_delta'] - 0.2),
                        'CI_Upper': glass_result.get('ci_upper', glass_result['glass_delta'] + 0.2),
                        'Interpretation': glass_result.get('interpretation', 'N/A')
                    })
                    
                    results_df = pd.DataFrame(results)
                    st.session_state.effect_size_results = results
                    
                    st.success("✅ Effect sizes calculated successfully!")
                    
                    # Display results table
                    st.dataframe(
                        results_df.style.format({
                            'Value': '{:.4f}',
                            'CI_Lower': '{:.4f}',
                            'CI_Upper': '{:.4f}'
                        }),
                        width='stretch',
                        hide_index=True
                    )
                    
                    # Create Forest Plot visualization
                    fig = go.Figure()
                    
                    colors = {'Small': '#3498db', 'Medium': '#f39c12', 'Large': '#e74c3c', 'Negligible': '#95a5a6'}
                    
                    for i, row in results_df.iterrows():
                        color = colors.get(row['Interpretation'], '#3498db')
                        
                        # Error bars (CI)
                        fig.add_trace(go.Scatter(
                            x=[row['CI_Lower'], row['CI_Upper']],
                            y=[row['Measure'], row['Measure']],
                            mode='lines',
                            line=dict(color=color, width=4),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Point estimate
                        fig.add_trace(go.Scatter(
                            x=[row['Value']],
                            y=[row['Measure']],
                            mode='markers',
                            marker=dict(color=color, size=15, symbol='diamond'),
                            name=f"{row['Measure']}: {row['Value']:.3f} ({row['Interpretation']})",
                            hovertemplate=f"<b>{row['Measure']}</b><br>Value: {row['Value']:.4f}<br>95% CI: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]<br>Interpretation: {row['Interpretation']}<extra></extra>"
                        ))
                    
                    # Add reference lines for effect size thresholds
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
                    fig.add_vline(x=0.2, line_dash="dot", line_color="gray", annotation_text="Small", annotation_position="top")
                    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="Medium", annotation_position="top")
                    fig.add_vline(x=0.8, line_dash="dot", line_color="gray", annotation_text="Large", annotation_position="top")
                    fig.add_vline(x=-0.2, line_dash="dot", line_color="gray")
                    fig.add_vline(x=-0.5, line_dash="dot", line_color="gray")
                    fig.add_vline(x=-0.8, line_dash="dot", line_color="gray")
                    
                    fig.update_layout(
                        title=f'Effect Size Forest Plot: {group1_col} vs {group2_col}',
                        xaxis_title='Effect Size',
                        yaxis_title='Measure',
                        template=PLOTLY_TEMPLATE,
                        height=350,
                        showlegend=True,
                        legend=dict(yanchor="bottom", y=-0.3, xanchor="center", x=0.5, orientation="h")
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Distribution comparison plot
                    st.markdown("---")
                    st.subheader("📈 Distribution Comparison")
                    
                    fig_dist = go.Figure()
                    
                    # Group 1 histogram
                    fig_dist.add_trace(go.Histogram(
                        x=df[group1_col].dropna(),
                        name=group1_col,
                        opacity=0.7,
                        nbinsx=30
                    ))
                    
                    # Group 2 histogram
                    fig_dist.add_trace(go.Histogram(
                        x=df[group2_col].dropna(),
                        name=group2_col,
                        opacity=0.7,
                        nbinsx=30
                    ))
                    
                    # Add mean lines
                    mean1 = df[group1_col].mean()
                    mean2 = df[group2_col].mean()
                    
                    fig_dist.add_vline(x=mean1, line_dash="dash", line_color="blue", 
                                      annotation_text=f"{group1_col} mean: {mean1:.2f}")
                    fig_dist.add_vline(x=mean2, line_dash="dash", line_color="red",
                                      annotation_text=f"{group2_col} mean: {mean2:.2f}")
                    
                    fig_dist.update_layout(
                        title='Distribution Comparison',
                        xaxis_title='Value',
                        yaxis_title='Count',
                        barmode='overlay',
                        template=PLOTLY_TEMPLATE,
                        height=300
                    )
                    
                    st.plotly_chart(fig_dist, width='stretch')
                    
                    # Descriptive statistics
                    st.markdown("---")
                    st.subheader("📋 Descriptive Statistics")
                    
                    desc_stats = pd.DataFrame({
                        'Statistic': ['N', 'Mean', 'SD', 'Median', 'Min', 'Max'],
                        group1_col: [
                            int(df[group1_col].count()),
                            f"{df[group1_col].mean():.4f}",
                            f"{df[group1_col].std():.4f}",
                            f"{df[group1_col].median():.4f}",
                            f"{df[group1_col].min():.4f}",
                            f"{df[group1_col].max():.4f}"
                        ],
                        group2_col: [
                            int(df[group2_col].count()),
                            f"{df[group2_col].mean():.4f}",
                            f"{df[group2_col].std():.4f}",
                            f"{df[group2_col].median():.4f}",
                            f"{df[group2_col].min():.4f}",
                            f"{df[group2_col].max():.4f}"
                        ]
                    })
                    
                    st.dataframe(desc_stats, width='stretch', hide_index=True)
                    
                    # Download buttons
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    dl_col1, dl_col2 = st.columns(2)
                    
                    with dl_col1:
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Effect Sizes (CSV)",
                            csv_results,
                            "effect_sizes.csv",
                            "text/csv",
                            key="download_es_csv"
                        )
                    
                    with dl_col2:
                        # Full report JSON
                        full_report = {
                            'comparison': f'{group1_col} vs {group2_col}',
                            'confidence_level': confidence,
                            'effect_sizes': results,
                            'descriptive_stats': {
                                group1_col: {
                                    'n': int(df[group1_col].count()),
                                    'mean': float(df[group1_col].mean()),
                                    'sd': float(df[group1_col].std()),
                                    'median': float(df[group1_col].median())
                                },
                                group2_col: {
                                    'n': int(df[group2_col].count()),
                                    'mean': float(df[group2_col].mean()),
                                    'sd': float(df[group2_col].std()),
                                    'median': float(df[group2_col].median())
                                }
                            }
                        }
                        json_report = json.dumps(full_report, indent=2)
                        st.download_button(
                            "📥 Download Full Report (JSON)",
                            json_report,
                            "effect_size_report.json",
                            "application/json",
                            key="download_es_json"
                        )
                    
                    st.session_state.analysis_results['effect_size'] = results
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("📚 Interpretation Guide")
        st.markdown("""
        | Effect Size | Small | Medium | Large |
        |-------------|-------|--------|-------|
        | Cohen's d | 0.2 | 0.5 | 0.8 |
        | Hedges' g | 0.2 | 0.5 | 0.8 |
        | Eta-squared | 0.01 | 0.06 | 0.14 |
        | Cramér's V | 0.1 | 0.3 | 0.5 |
        """)
        
        st.markdown("---")
        st.subheader("🔄 Effect Size Conversions")
        
        conv_type = st.selectbox("Conversion", ["r to d", "d to r"], key="es_conv_type")
        value = st.number_input("Value to convert", value=0.5, key="es_conv_val")
        
        if st.button("Convert", key="es_convert"):
            if conv_type == "r to d":
                result = calc.r_to_d(value)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(f"r = {value:.3f} → d = {result['cohens_d']:.4f}")
                    st.write(f"r² = {result['r_squared']:.4f}")
            else:
                result = calc.d_to_r(value)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(f"d = {value:.3f} → r = {result['r']:.4f}")
                    st.write(f"r² = {result['r_squared']:.4f}")
        
        st.markdown("---")
        st.info("""
        **💡 Tips:**
        - Cohen's d uses pooled SD
        - Hedges' g corrects for small samples
        - Glass's Δ uses control group SD only
        - All plots are downloadable (📷 icon)
        """)


def render_model_validation_tab():
    """Render the model validation tab"""
    st.header("✅ Model Validation")
    st.caption("Rigorous cross-validation, calibration analysis, and diagnostic tools")
    
    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    if not features or not target:
        st.warning("Please select feature columns and a target column in the Data tab.")
        return
    
    validator = ModelValidation(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cross-Validation")
        
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        task_type = st.selectbox("Task Type", ["Regression", "Classification"], key="mv_task")
        
        if task_type == "Regression":
            model_options = {"Linear Regression": LinearRegression(), 
                           "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)}
        else:
            model_options = {"Logistic Regression": LogisticRegression(max_iter=1000),
                           "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)}
        
        model_name = st.selectbox("Model", list(model_options.keys()), key="mv_model")
        cv_folds = st.slider("CV Folds", 3, 10, 5, key="mv_cv")
        
        if st.button("🔄 Run Cross-Validation", key="run_cv"):
            with st.spinner("Running cross-validation..."):
                try:
                    result = validator.cross_validate(
                        model_options[model_name],
                        features,
                        target,
                        cv=cv_folds
                    )
                    
                    st.success("Cross-validation complete!")
                    
                    # Display scores
                    scores_df = pd.DataFrame({
                        'Metric': list(result['mean_scores'].keys()),
                        'Mean': [f"{v:.4f}" for v in result['mean_scores'].values()],
                        'Std': [f"{v:.4f}" for v in result['std_scores'].values()]
                    })
                    st.dataframe(scores_df, width='stretch')
                    
                    st.session_state.analysis_results['cv_results'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("Learning Curve Analysis")
        
        if st.button("📈 Generate Learning Curve", key="run_lc"):
            with st.spinner("Analyzing learning curve..."):
                try:
                    result = validator.learning_curve_analysis(
                        model_options[model_name],
                        features,
                        target,
                        cv=cv_folds
                    )
                    
                    # Plot learning curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result['train_sizes'],
                        y=result['train_scores_mean'],
                        mode='lines+markers',
                        name='Training Score',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=result['train_sizes'],
                        y=result['test_scores_mean'],
                        mode='lines+markers',
                        name='Validation Score',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title='Learning Curve',
                        xaxis_title='Training Set Size',
                        yaxis_title='Score',
                        template=PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Residual Diagnostics")
        st.caption("For regression models only")
        
        if task_type == "Regression":
            if st.button("🔬 Run Residual Diagnostics", key="run_resid"):
                with st.spinner("Analyzing residuals..."):
                    try:
                        result = validator.residual_diagnostics(
                            model_options[model_name],
                            features,
                            target
                        )
                        
                        # Normality test
                        st.write(f"**Normality Test (Shapiro-Wilk)**")
                        st.write(f"p-value: {result['normality_test']['p_value']:.4f}")
                        if result['normality_test']['p_value'] > 0.05:
                            st.success("✅ Residuals appear normally distributed")
                        else:
                            st.warning("⚠️ Residuals may not be normally distributed")
                        
                        # Durbin-Watson
                        st.write(f"**Durbin-Watson Statistic**: {result['durbin_watson']:.4f}")
                        if 1.5 < result['durbin_watson'] < 2.5:
                            st.success("✅ No significant autocorrelation")
                        else:
                            st.warning("⚠️ Possible autocorrelation in residuals")
                        
                        # Q-Q Plot
                        residuals = result['standardized_residuals']
                        from scipy import stats
                        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
                        sorted_residuals = np.sort(residuals)
                        
                        fig = px.scatter(x=theoretical_quantiles, y=sorted_residuals[:len(theoretical_quantiles)],
                                        title='Q-Q Plot (Residuals)', template=PLOTLY_TEMPLATE)
                        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', 
                                                name='Reference', line=dict(dash='dash', color='red')))
                        st.plotly_chart(fig, width='stretch')
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Residual diagnostics are for regression models. Select regression task type.")


def render_data_quality_tab():
    """Render the data quality analysis tab"""
    st.header("🔍 Data Quality Analysis")
    st.caption("Missing data analysis, imputation, and quality assessment")
    
    df = st.session_state.df
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    dqa = DataQuality(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Missing Data Summary")
        
        if st.button("Analyze Missing Data", key="analyze_missing"):
            try:
                summary = dqa.missing_data_summary()
                st.dataframe(summary, width='stretch')
                
                # Visualize
                if summary['missing_pct'].sum() > 0:
                    fig = px.bar(summary, x=summary.index, y='missing_pct',
                               title='Missing Data Percentage by Column',
                               template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.success("✅ No missing data detected!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("🔧 Imputation")
        
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        if cols_with_missing:
            imp_cols = st.multiselect("Columns to Impute", cols_with_missing, key="imp_cols")
            imp_method = st.selectbox("Method", ["mean", "median", "mode", "knn"], key="imp_method")
            
            if imp_method == "knn":
                n_neighbors = st.slider("K Neighbors", 3, 15, 5, key="imp_knn")
            
            if st.button("Apply Imputation", key="apply_imp"):
                if imp_cols:
                    try:
                        kwargs = {'n_neighbors': n_neighbors} if imp_method == "knn" else {}
                        result = dqa.impute_missing(imp_cols, method=imp_method, **kwargs)
                        st.session_state.df = result['imputed_data']
                        st.success(f"✅ Imputed {len(imp_cols)} column(s) using {imp_method}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Select columns to impute.")
        else:
            st.success("✅ No columns with missing data!")
    
    with col2:
        st.subheader("🎯 Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_cols = st.multiselect("Columns to Check", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols, key="outlier_cols")
        outlier_method = st.selectbox("Detection Method", ["iqr", "zscore", "mad"], key="outlier_method")
        
        if st.button("Detect Outliers", key="detect_outliers"):
            if outlier_cols:
                try:
                    result = dqa.detect_outliers(outlier_cols, method=outlier_method)
                    
                    st.write(f"**Total outliers detected:** {result['total_outliers']}")
                    
                    # Per-column breakdown
                    outlier_df = pd.DataFrame({
                        'Column': list(result['outlier_counts'].keys()),
                        'Outlier Count': list(result['outlier_counts'].values())
                    })
                    st.dataframe(outlier_df, width='stretch')
                    
                    st.session_state.analysis_results['outliers'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("📋 Quality Report")
        
        if st.button("Generate Full Quality Report", key="gen_quality"):
            with st.spinner("Analyzing data quality..."):
                try:
                    report = dqa.generate_quality_report()
                    
                    st.write("**Data Shape:**", f"{report['n_rows']} rows × {report['n_cols']} columns")
                    st.write("**Memory Usage:**", f"{report['memory_usage']:.2f} MB")
                    st.write("**Duplicate Rows:**", report['n_duplicates'])
                    st.write("**Total Missing Values:**", report['total_missing'])
                    
                    st.session_state.analysis_results['quality_report'] = report
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_feature_selection_tab():
    """Render the feature selection tab"""
    st.header("🎯 Feature Selection")
    st.caption("Multiple methods for optimal feature subset selection")
    
    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    if not features or not target:
        st.warning("Please select feature columns and a target column in the Data tab.")
        return
    
    selector = FeatureSelection(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selection Methods")
        
        method = st.selectbox("Method", [
            "Statistical (Mutual Information)",
            "Statistical (F-test)",
            "Recursive Feature Elimination (RFE)",
            "Lasso Selection",
            "Permutation Importance"
        ], key="fs_method")
        
        n_features = st.slider("Number of Features to Select", 1, len(features), min(5, len(features)), key="fs_n")
        
        if st.button("🔍 Run Feature Selection", key="run_fs"):
            with st.spinner("Selecting features..."):
                try:
                    if "Statistical (Mutual Information)" in method:
                        result = selector.statistical_selection(features, target, method='mutual_info', k=n_features)
                    elif "Statistical (F-test)" in method:
                        result = selector.statistical_selection(features, target, method='f_regression', k=n_features)
                    elif "RFE" in method:
                        result = selector.recursive_feature_elimination(features, target, n_features=n_features)
                    elif "Lasso" in method:
                        result = selector.lasso_selection(features, target)
                    elif "Permutation" in method:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        result = selector.permutation_selection(model, features, target)
                    
                    st.success("Feature selection complete!")
                    st.write("**Selected Features:**")
                    for i, feat in enumerate(result['selected_features'], 1):
                        st.write(f"{i}. {feat}")
                    
                    st.session_state.analysis_results['feature_selection'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Ensemble Selection")
        st.caption("Combines multiple methods for robust selection")
        
        if st.button("🔄 Run Ensemble Selection", key="run_ensemble_fs"):
            with st.spinner("Running ensemble feature selection..."):
                try:
                    result = selector.ensemble_selection(features, target)
                    
                    st.success("Ensemble selection complete!")
                    st.write("**Consensus Features (selected by multiple methods):**")
                    for feat in result['consensus_features']:
                        st.write(f"✅ {feat}")
                    
                    if 'feature_votes' in result:
                        st.write("**Feature Votes:**")
                        votes_df = pd.DataFrame({
                            'Feature': list(result['feature_votes'].keys()),
                            'Votes': list(result['feature_votes'].values())
                        }).sort_values('Votes', ascending=False)
                        st.dataframe(votes_df, width='stretch')
                    
                    st.session_state.analysis_results['ensemble_fs'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("Feature Ranking Visualization")
        
        if 'feature_selection' in st.session_state.analysis_results:
            result = st.session_state.analysis_results['feature_selection']
            if 'scores' in result:
                fig = px.bar(x=list(result['scores'].keys()), 
                           y=list(result['scores'].values()),
                           title='Feature Importance Scores',
                           template=PLOTLY_TEMPLATE)
                fig.update_layout(xaxis_title='Feature', yaxis_title='Score')
                st.plotly_chart(fig, width='stretch')


def render_survival_tab():
    """Render the survival analysis tab"""
    st.header("⏳ Survival Analysis")
    st.caption("Time-to-event analysis with censoring")
    
    if not SURVIVAL_AVAILABLE:
        st.warning("⚠️ Survival analysis requires the `lifelines` package.")
        st.code("pip install data-toolkit[survival]", language="bash")
        return
    
    df = st.session_state.df
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns (time and event) for survival analysis.")
        return
    
    surv = SurvivalAnalysis(df)
    
    # Configuration panel at top
    st.subheader("⚙️ Configuration")
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        time_col = st.selectbox("Time Column", numeric_cols, key="surv_time")
    with config_col2:
        event_col = st.selectbox("Event Column (1=event, 0=censored)", 
                                [c for c in numeric_cols if c != time_col], key="surv_event")
    with config_col3:
        group_col = st.selectbox("Group Column (optional)", [None] + all_cols, key="surv_group")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Kaplan-Meier Analysis")
        
        if st.button("📊 Run Kaplan-Meier", key="run_km"):
            with st.spinner("Computing Kaplan-Meier curves..."):
                try:
                    result = surv.kaplan_meier(time_col, event_col, 
                                              group_col=group_col if group_col else None)
                    
                    # Check for error
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✅ Kaplan-Meier analysis complete!")
                        
                        # Display metrics
                        if 'median_survival' in result and result['median_survival'] is not None:
                            st.metric("Median Survival Time", f"{result['median_survival']:.2f}")
                        
                        if 'n_observations' in result:
                            met_col1, met_col2, met_col3 = st.columns(3)
                            with met_col1:
                                st.metric("Total Observations", result['n_observations'])
                            with met_col2:
                                st.metric("Events", result.get('n_events', 'N/A'))
                            with met_col3:
                                st.metric("Censored", result.get('n_censored', 'N/A'))
                        
                        # Create Kaplan-Meier survival curve plot
                        fig = go.Figure()
                        
                        if 'groups' in result:
                            # Multiple groups
                            colors = px.colors.qualitative.Set1
                            for i, (group_name, group_data) in enumerate(result['groups'].items()):
                                if 'survival_function' in group_data:
                                    times = list(group_data['survival_function'].keys())
                                    survival = list(group_data['survival_function'].values())
                                    
                                    fig.add_trace(go.Scatter(
                                        x=times, y=survival,
                                        mode='lines+markers',
                                        name=f'{group_name} (n={group_data.get("n_observations", "?")})',
                                        line=dict(color=colors[i % len(colors)], width=2),
                                        marker=dict(size=4)
                                    ))
                        else:
                            # Single group
                            if 'survival_function' in result:
                                sf = result['survival_function']
                                if 'KM_estimate' in sf:
                                    times = list(sf['KM_estimate'].keys())
                                    survival = list(sf['KM_estimate'].values())
                                else:
                                    times = list(sf.keys())
                                    survival = list(sf.values())
                                
                                fig.add_trace(go.Scatter(
                                    x=times, y=survival,
                                    mode='lines+markers',
                                    name='Survival Function',
                                    line=dict(color='blue', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(0, 100, 200, 0.1)'
                                ))
                                
                                # Add confidence interval if available
                                if 'confidence_interval_lower' in result and 'confidence_interval_upper' in result:
                                    ci_lower = list(result['confidence_interval_lower'].values())
                                    ci_upper = list(result['confidence_interval_upper'].values())
                                    
                                    fig.add_trace(go.Scatter(
                                        x=times + times[::-1],
                                        y=ci_upper + ci_lower[::-1],
                                        fill='toself',
                                        fillcolor='rgba(0, 100, 200, 0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        showlegend=True,
                                        name='95% CI'
                                    ))
                        
                        fig.update_layout(
                            title='Kaplan-Meier Survival Curve',
                            xaxis_title='Time',
                            yaxis_title='Survival Probability',
                            yaxis=dict(range=[0, 1.05]),
                            template=PLOTLY_TEMPLATE,
                            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Survival at specific time points
                        if 'survival_at' in result:
                            st.write("**Survival Probability at Key Time Points:**")
                            surv_at_df = pd.DataFrame([
                                {'Time': t, 'Survival Probability': f"{p:.3f}"}
                                for t, p in result['survival_at'].items()
                            ])
                            st.dataframe(surv_at_df, width='stretch', hide_index=True)
                        
                        # Download results
                        results_json = json.dumps(result, indent=2, default=str)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            results_json,
                            "kaplan_meier_results.json",
                            "application/json",
                            key="download_km_json"
                        )
                        
                        st.session_state.analysis_results['km_result'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.subheader("📊 Log-Rank Test")
        
        # Get available groups if group column is selected
        available_groups = []
        if group_col:
            available_groups = df[group_col].dropna().unique().tolist()
        
        if group_col and len(available_groups) > 2:
            selected_groups = st.multiselect(
                "Select 2 groups to compare",
                available_groups,
                default=available_groups[:2] if len(available_groups) >= 2 else available_groups,
                max_selections=2,
                key="logrank_groups"
            )
        else:
            selected_groups = available_groups
        
        if st.button("🔬 Run Log-Rank Test", key="run_logrank"):
            if group_col:
                with st.spinner("Computing log-rank test..."):
                    try:
                        # Filter data to only include selected groups
                        if len(available_groups) > 2 and len(selected_groups) == 2:
                            df_filtered = df[df[group_col].isin(selected_groups)].copy()
                            surv_filtered = SurvivalAnalysis(df_filtered)
                            result = surv_filtered.log_rank_test(time_col, event_col, group_col)
                        else:
                            result = surv.log_rank_test(time_col, event_col, group_col)
                        
                        # Check for error in result
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        elif 'test_statistic' not in result:
                            st.error(f"Error: Unexpected result format - {result}")
                        else:
                            st.success("✅ Log-Rank Test Complete!")
                            
                            # Display results in a nice format
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.metric("Test Statistic", f"{result['test_statistic']:.4f}")
                            with res_col2:
                                st.metric("p-value", f"{result['p_value']:.6f}")
                            
                            # Significance interpretation
                            if result.get('is_significant', result['p_value'] < 0.05):
                                st.success(f"✅ **Significant difference** between {result.get('group_1', 'Group 1')} and {result.get('group_2', 'Group 2')} (p < 0.05)")
                            else:
                                st.info(f"ℹ️ **No significant difference** between groups (p ≥ 0.05)")
                            
                            # Create a simple bar chart comparing groups
                            groups = df[group_col].unique()
                            if len(groups) == 2:
                                group_stats = []
                                for g in groups:
                                    mask = df[group_col] == g
                                    group_stats.append({
                                        'Group': str(g),
                                        'N': mask.sum(),
                                        'Events': df.loc[mask, event_col].sum(),
                                        'Median Time': df.loc[mask, time_col].median()
                                    })
                                
                                stats_df = pd.DataFrame(group_stats)
                                
                                fig = go.Figure(data=[
                                    go.Bar(name='Events', x=stats_df['Group'], y=stats_df['Events']),
                                    go.Bar(name='Median Time', x=stats_df['Group'], y=stats_df['Median Time'])
                                ])
                                fig.update_layout(
                                    title='Group Comparison',
                                    barmode='group',
                                    template=PLOTLY_TEMPLATE
                                )
                                st.plotly_chart(fig, width='stretch')
                            
                            st.session_state.analysis_results['logrank_result'] = result
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("⚠️ Select a group column for log-rank test.")
    
    with col2:
        st.subheader("🔬 Cox Proportional Hazards")
        
        cox_covariates = st.multiselect("Covariates", 
                                        [c for c in numeric_cols if c not in [time_col, event_col]],
                                        key="cox_covars")
        
        if st.button("📊 Run Cox Regression", key="run_cox"):
            if cox_covariates:
                with st.spinner("Fitting Cox model..."):
                    try:
                        result = surv.cox_regression(time_col, event_col, cox_covariates)
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.success("✅ Cox regression complete!")
                            
                            # Model performance metrics
                            met_col1, met_col2 = st.columns(2)
                            with met_col1:
                                if 'concordance_index' in result:
                                    st.metric("C-statistic (Concordance)", 
                                             f"{result['concordance_index']:.4f}")
                            with met_col2:
                                if 'aic_partial' in result:
                                    st.metric("AIC (Partial)", f"{result['aic_partial']:.2f}")
                            
                            # Hazard Ratios visualization
                            if 'hazard_ratios' in result:
                                st.write("**Hazard Ratios (with 95% CI):**")
                                
                                hr_data = []
                                for covar, hr_info in result['hazard_ratios'].items():
                                    hr_data.append({
                                        'Covariate': covar,
                                        'HR': hr_info['hazard_ratio'],
                                        'CI_Lower': hr_info['ci_lower'],
                                        'CI_Upper': hr_info['ci_upper'],
                                        'p-value': hr_info['p_value'],
                                        'Significant': '✓' if hr_info['significant'] else ''
                                    })
                                
                                hr_df = pd.DataFrame(hr_data)
                                st.dataframe(hr_df, width='stretch', hide_index=True)
                                
                                # Forest plot for hazard ratios
                                fig = go.Figure()
                                
                                for i, row in hr_df.iterrows():
                                    color = 'red' if row['HR'] > 1 else 'green'
                                    
                                    # CI error bars
                                    fig.add_trace(go.Scatter(
                                        x=[row['CI_Lower'], row['CI_Upper']],
                                        y=[row['Covariate'], row['Covariate']],
                                        mode='lines',
                                        line=dict(color=color, width=3),
                                        showlegend=False
                                    ))
                                    
                                    # Point estimate
                                    fig.add_trace(go.Scatter(
                                        x=[row['HR']],
                                        y=[row['Covariate']],
                                        mode='markers',
                                        marker=dict(color=color, size=12, symbol='diamond'),
                                        name=f"{row['Covariate']}: {row['HR']:.2f}",
                                        showlegend=False
                                    ))
                                
                                # Reference line at HR=1
                                fig.add_vline(x=1, line_dash="dash", line_color="gray")
                                
                                fig.update_layout(
                                    title='Forest Plot: Hazard Ratios',
                                    xaxis_title='Hazard Ratio (log scale)',
                                    yaxis_title='Covariate',
                                    xaxis_type='log',
                                    template=PLOTLY_TEMPLATE,
                                    height=max(300, len(hr_data) * 50)
                                )
                                
                                st.plotly_chart(fig, width='stretch')
                                
                                # Download hazard ratios
                                csv_hr = hr_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Hazard Ratios (CSV)",
                                    csv_hr,
                                    "hazard_ratios.csv",
                                    "text/csv",
                                    key="download_hr_csv"
                                )
                            
                            st.session_state.analysis_results['cox_result'] = result
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("⚠️ Select at least one covariate.")
        
        # Display saved results
        st.markdown("---")
        st.subheader("📋 Interpretation Guide")
        st.markdown("""
        | Metric | Interpretation |
        |--------|----------------|
        | **Hazard Ratio (HR)** | HR > 1: ↑ risk, HR < 1: ↓ risk |
        | **C-statistic** | 0.5 = random, 1.0 = perfect |
        | **Log-Rank p-value** | < 0.05 = significant difference |
        | **Median Survival** | Time at 50% survival |
        """)


def render_interpretability_tab():
    """Render the model interpretability tab"""
    st.header("🔮 Model Interpretability")
    st.caption("Explain model predictions with SHAP, LIME, and partial dependence")
    
    if not INTERPRETABILITY_AVAILABLE:
        st.warning("⚠️ Interpretability requires `shap` and `lime` packages.")
        st.code("pip install data-toolkit[interpretability]", language="bash")
        return
    
    df = st.session_state.df
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    if not features or not target:
        st.warning("Please select feature columns and a target column in the Data tab.")
        return
    
    # Train a simple model for interpretation
    from sklearn.ensemble import RandomForestRegressor
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 SHAP Analysis")
        st.caption("SHapley Additive exPlanations - Feature importance")
        
        n_samples = st.slider("Sample Size", 50, 200, 100, key="shap_samples")
        
        if st.button("🔍 Run SHAP Analysis", key="run_shap"):
            with st.spinner("Computing SHAP values..."):
                try:
                    # Train model
                    X = df[features].dropna()
                    y = df.loc[X.index, target]
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    
                    # Create interpreter
                    interp = ModelInterpretability(model, df, features)
                    result = interp.shap_analysis(n_samples=min(n_samples, len(X)))
                    
                    st.success("SHAP analysis complete!")
                    
                    if 'feature_importance' in result:
                        fig = px.bar(x=list(result['feature_importance'].keys()),
                                   y=list(result['feature_importance'].values()),
                                   title='SHAP Feature Importance',
                                   template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width='stretch')
                    
                    st.session_state.analysis_results['shap'] = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("🔬 Permutation Importance")
        
        if st.button("📊 Compute Permutation Importance", key="run_perm_imp"):
            with st.spinner("Computing permutation importance..."):
                try:
                    X = df[features].dropna()
                    y = df.loc[X.index, target]
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    
                    interp = ModelInterpretability(model, df, features)
                    result = interp.permutation_feature_importance(n_repeats=10)
                    
                    st.success("Permutation importance complete!")
                    
                    if 'importances_mean' in result:
                        imp_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': result['importances_mean']
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(imp_df, x='Feature', y='Importance',
                                   title='Permutation Feature Importance',
                                   template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_advanced_ts_tab():
    """Render the advanced time series tab"""
    st.header("📈 Advanced Time Series")
    st.caption("Prophet forecasting, changepoint detection, and more")
    
    if not ADVANCED_TS_AVAILABLE:
        st.warning("⚠️ Advanced time series requires additional packages.")
        st.code("pip install data-toolkit[timeseries]", language="bash")
        return
    
    df = st.session_state.df
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    ats = AdvancedTimeSeries(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Prophet Forecasting")
        
        date_col = st.selectbox("Date Column", all_cols, key="prophet_date")
        value_col = st.selectbox("Value Column", numeric_cols, key="prophet_value")
        periods = st.slider("Forecast Periods", 7, 365, 30, key="prophet_periods")
        
        if st.button("📈 Generate Forecast", key="run_prophet"):
            with st.spinner("Fitting Prophet model..."):
                try:
                    result = ats.prophet_forecast(date_col, value_col, periods=periods)
                    
                    st.success("Forecast complete!")
                    
                    if 'forecast' in result:
                        forecast_df = result['forecast']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                                               mode='lines', name='Forecast'))
                        if 'yhat_lower' in forecast_df.columns:
                            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                                                   mode='lines', name='Upper Bound', line=dict(dash='dash')))
                            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                                                   mode='lines', name='Lower Bound', line=dict(dash='dash')))
                        fig.update_layout(title='Prophet Forecast', template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("📍 Changepoint Detection")
        
        cp_col = st.selectbox("Column to Analyze", numeric_cols, key="cp_col")
        cp_method = st.selectbox("Detection Method", ["pelt", "binseg", "window"], key="cp_method")
        
        if st.button("🔍 Detect Changepoints", key="run_cp"):
            with st.spinner("Detecting changepoints..."):
                try:
                    result = ats.detect_changepoints(cp_col, method=cp_method)
                    
                    st.success(f"Found {result['n_changepoints']} changepoint(s)")
                    
                    # Plot with changepoints
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df[cp_col], mode='lines', name='Signal'))
                    
                    for cp in result['changepoints']:
                        fig.add_vline(x=cp, line_dash="dash", line_color="red")
                    
                    fig.update_layout(title='Signal with Changepoints', template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_domain_specific_tab():
    """Render the domain-specific analysis tab"""
    st.header("🔬 Domain-Specific Analysis")
    st.caption("Environmental science, clinical research, and ecology tools")
    
    df = st.session_state.df
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not DOMAIN_SPECIFIC_AVAILABLE:
        st.warning("⚠️ Some domain-specific features require additional packages.")
        st.code("pip install data-toolkit[ecology]", language="bash")
    
    domain = DomainSpecificAnalysis(df) if DOMAIN_SPECIFIC_AVAILABLE else None
    
    tab1, tab2, tab3 = st.tabs(["🌿 Environmental", "🏥 Clinical", "🦎 Ecology"])
    
    with tab1:
        st.subheader("Environmental Science")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mann-Kendall Trend Test**")
            mk_col = st.selectbox("Column", numeric_cols, key="mk_col")
            
            if st.button("🔍 Run Mann-Kendall Test", key="run_mk"):
                if domain:
                    try:
                        result = domain.mann_kendall_test(mk_col)
                        
                        # Display results
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Trend", result['trend'])
                            st.metric("Kendall's τ", f"{result['tau']:.4f}")
                        with res_col2:
                            st.metric("p-value", f"{result['p_value']:.6f}")
                            if result['p_value'] < 0.05:
                                st.success("✅ Significant trend")
                            else:
                                st.info("No significant trend")
                        
                        # Create trend visualization
                        data = df[mk_col].dropna().reset_index(drop=True)
                        fig = go.Figure()
                        
                        # Original data
                        fig.add_trace(go.Scatter(
                            x=list(range(len(data))),
                            y=data.values,
                            mode='lines+markers',
                            name='Data',
                            line=dict(color='blue', width=1),
                            marker=dict(size=4)
                        ))
                        
                        # Add trend line (using Sen's slope if available)
                        try:
                            sens_result = domain.sens_slope(mk_col)
                            slope = sens_result['slope']
                            intercept = sens_result['intercept']
                            x_trend = np.array([0, len(data)-1])
                            y_trend = intercept + slope * x_trend
                            
                            fig.add_trace(go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode='lines',
                                name=f'Trend (slope={slope:.4f})',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                        except:
                            pass
                        
                        # Color based on trend direction
                        trend_color = 'green' if result['trend'] == 'increasing' else ('red' if result['trend'] == 'decreasing' else 'gray')
                        
                        fig.update_layout(
                            title=f"Mann-Kendall Trend Analysis: {mk_col}<br><sub>Trend: {result['trend']} (τ={result['tau']:.3f}, p={result['p_value']:.4f})</sub>",
                            xaxis_title='Observation Index',
                            yaxis_title=mk_col,
                            template=PLOTLY_TEMPLATE,
                            height=400
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Download results
                        results_json = json.dumps(result, indent=2, default=str)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            results_json,
                            "mann_kendall_results.json",
                            "application/json",
                            key="download_mk_json"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            st.write("**Sen's Slope Estimator**")
            sens_col = st.selectbox("Column", numeric_cols, key="sens_col")
            
            if st.button("📈 Calculate Sen's Slope", key="run_sens"):
                if domain:
                    try:
                        result = domain.sens_slope(sens_col)
                        
                        # Display results
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Sen's Slope", f"{result['slope']:.6f}")
                        with res_col2:
                            st.metric("Intercept", f"{result['intercept']:.4f}")
                        
                        # Create visualization
                        data = df[sens_col].dropna().reset_index(drop=True)
                        x_vals = np.arange(len(data))
                        
                        fig = go.Figure()
                        
                        # Original data
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=data.values,
                            mode='markers',
                            name='Data Points',
                            marker=dict(color='blue', size=6, opacity=0.6)
                        ))
                        
                        # Sen's slope trend line
                        y_trend = result['intercept'] + result['slope'] * x_vals
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_trend,
                            mode='lines',
                            name=f"Sen's Slope: {result['slope']:.4f}",
                            line=dict(color='red', width=3)
                        ))
                        
                        # Add confidence interval if available
                        if 'ci_lower' in result and 'ci_upper' in result:
                            y_lower = result['intercept'] + result['ci_lower'] * x_vals
                            y_upper = result['intercept'] + result['ci_upper'] * x_vals
                            
                            fig.add_trace(go.Scatter(
                                x=np.concatenate([x_vals, x_vals[::-1]]),
                                y=np.concatenate([y_upper, y_lower[::-1]]),
                                fill='toself',
                                fillcolor='rgba(255, 0, 0, 0.1)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% CI',
                                showlegend=True
                            ))
                        
                        # Calculate change per unit
                        total_change = result['slope'] * len(data)
                        pct_change = (total_change / data.iloc[0]) * 100 if data.iloc[0] != 0 else 0
                        
                        fig.update_layout(
                            title=f"Sen's Slope Estimation: {sens_col}<br><sub>Slope: {result['slope']:.4f} per unit (Total change: {total_change:.2f}, {pct_change:.1f}%)</sub>",
                            xaxis_title='Observation Index',
                            yaxis_title=sens_col,
                            template=PLOTLY_TEMPLATE,
                            height=400
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Additional statistics
                        st.write("**Trend Statistics:**")
                        stats_df = pd.DataFrame({
                            'Metric': ['Slope (per observation)', 'Intercept', 'Total Change', 'Percent Change'],
                            'Value': [f"{result['slope']:.6f}", f"{result['intercept']:.4f}", 
                                      f"{total_change:.4f}", f"{pct_change:.2f}%"]
                        })
                        st.dataframe(stats_df, width='stretch', hide_index=True)
                        
                        # Download results
                        results_json = json.dumps(result, indent=2, default=str)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            results_json,
                            "sens_slope_results.json",
                            "application/json",
                            key="download_sens_json"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Clinical Research")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bland-Altman Method Comparison**")
            method1 = st.selectbox("Method 1", numeric_cols, key="ba_m1")
            method2 = st.selectbox("Method 2", [c for c in numeric_cols if c != method1], key="ba_m2")
            
            if st.button("📊 Bland-Altman Analysis", key="run_ba"):
                if domain:
                    try:
                        result = domain.bland_altman(method1, method2)
                        st.write(f"**Bias (Mean Difference):** {result['mean_difference']:.4f}")
                        st.write(f"**Lower LoA:** {result['lower_loa']:.4f}")
                        st.write(f"**Upper LoA:** {result['upper_loa']:.4f}")
                        
                        # Bland-Altman plot
                        mean_vals = (df[method1] + df[method2]) / 2
                        diff_vals = df[method1] - df[method2]
                        
                        fig = px.scatter(x=mean_vals, y=diff_vals,
                                       title='Bland-Altman Plot',
                                       template=PLOTLY_TEMPLATE)
                        fig.add_hline(y=result['mean_difference'], line_dash='solid', line_color='blue')
                        fig.add_hline(y=result['upper_loa'], line_dash='dash', line_color='red')
                        fig.add_hline(y=result['lower_loa'], line_dash='dash', line_color='red')
                        fig.update_layout(xaxis_title='Mean of Methods', yaxis_title='Difference')
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            st.write("**Cohen's Kappa (Inter-rater Agreement)**")
            rater1 = st.selectbox("Rater 1", numeric_cols, key="kappa_r1")
            rater2 = st.selectbox("Rater 2", [c for c in numeric_cols if c != rater1], key="kappa_r2")
            
            if st.button("📊 Calculate Kappa", key="run_kappa"):
                if domain:
                    try:
                        result = domain.cohens_kappa(rater1, rater2)
                        
                        # Display metrics
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("Cohen's κ", f"{result['kappa']:.4f}")
                        with met_col2:
                            st.metric("Observed Agreement", f"{result['observed_agreement']:.1%}")
                        with met_col3:
                            st.metric("Expected Agreement", f"{result['expected_agreement']:.1%}")
                        
                        # Interpretation with color coding
                        kappa = result['kappa']
                        if kappa >= 0.81:
                            st.success(f"✅ {result['interpretation']} (κ = {kappa:.3f})")
                        elif kappa >= 0.61:
                            st.success(f"👍 {result['interpretation']} (κ = {kappa:.3f})")
                        elif kappa >= 0.41:
                            st.info(f"📊 {result['interpretation']} (κ = {kappa:.3f})")
                        elif kappa >= 0.21:
                            st.warning(f"⚠️ {result['interpretation']} (κ = {kappa:.3f})")
                        else:
                            st.error(f"❌ {result['interpretation']} (κ = {kappa:.3f})")
                        
                        st.write(f"**95% CI:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
                        st.write(f"**N:** {result['n_observations']} observations, {result['n_categories']} categories")
                        
                        # Create confusion matrix for visualization
                        r1_data = df[rater1].dropna()
                        r2_data = df[rater2].loc[r1_data.index].dropna()
                        r1_data = r1_data.loc[r2_data.index]
                        confusion = pd.crosstab(r1_data, r2_data, margins=True, margins_name='Total')
                        
                        # Confusion Matrix Heatmap
                        st.write("**Agreement Matrix (Confusion Matrix):**")
                        conf_no_margins = confusion.iloc[:-1, :-1]
                        fig_conf = px.imshow(
                            conf_no_margins.values,
                            x=conf_no_margins.columns.astype(str),
                            y=conf_no_margins.index.astype(str),
                            labels=dict(x=f'{rater2}', y=f'{rater1}', color='Count'),
                            title='Inter-Rater Agreement Matrix',
                            color_continuous_scale='Blues',
                            text_auto=True,
                            template=PLOTLY_TEMPLATE
                        )
                        fig_conf.update_layout(height=400)
                        st.plotly_chart(fig_conf, width='stretch')
                        
                        # Kappa scale visualization
                        st.write("**Kappa Interpretation Scale:**")
                        scale_labels = ['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Almost Perfect']
                        scale_thresholds = [0, 0.20, 0.40, 0.60, 0.80, 1.0]
                        scale_colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
                        
                        fig_scale = go.Figure()
                        
                        # Add colored bars for each category
                        for i in range(len(scale_labels)):
                            fig_scale.add_trace(go.Bar(
                                x=[scale_thresholds[i+1] - scale_thresholds[i] if i < len(scale_thresholds)-1 else 0.2],
                                y=['Kappa Scale'],
                                orientation='h',
                                marker_color=scale_colors[i],
                                name=scale_labels[i],
                                text=scale_labels[i],
                                textposition='inside',
                                base=scale_thresholds[i],
                                hoverinfo='name',
                                showlegend=False
                            ))
                        
                        # Add marker for current kappa value
                        fig_scale.add_trace(go.Scatter(
                            x=[max(0, min(1, kappa))],
                            y=['Kappa Scale'],
                            mode='markers+text',
                            marker=dict(size=20, color='black', symbol='diamond'),
                            text=[f'κ={kappa:.2f}'],
                            textposition='top center',
                            name='Your κ',
                            showlegend=False
                        ))
                        
                        fig_scale.update_layout(
                            title='Where Your Kappa Falls on the Agreement Scale',
                            xaxis=dict(range=[-0.1, 1.1], title='Kappa Value', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                            yaxis=dict(visible=False),
                            height=200,
                            barmode='stack',
                            template=PLOTLY_TEMPLATE
                        )
                        st.plotly_chart(fig_scale, width='stretch')
                        
                        # Download results
                        results_json = json.dumps(result, indent=2, default=str)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            results_json,
                            "cohens_kappa_results.json",
                            "application/json",
                            key="download_kappa_json"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("Ecology")
        
        st.write("**Shannon Diversity Index**")
        species_cols = st.multiselect("Species/Abundance Columns", numeric_cols, key="shannon_cols")
        
        if st.button("🌿 Calculate Shannon Diversity", key="run_shannon"):
            if domain and species_cols:
                try:
                    result = domain.shannon_diversity(species_cols)
                    
                    # Display results with metrics
                    met_col1, met_col2, met_col3 = st.columns(3)
                    with met_col1:
                        st.metric("Shannon H'", f"{result['diversity_index']:.4f}")
                    with met_col2:
                        st.metric("Evenness (J')", f"{result['evenness']:.4f}")
                    with met_col3:
                        st.metric("Richness (S)", result['richness'])
                    
                    # Interpretation
                    if result['diversity_index'] < 1.0:
                        st.info("📊 Low diversity - community dominated by few species")
                    elif result['diversity_index'] < 2.0:
                        st.info("📊 Moderate diversity")
                    else:
                        st.success("📊 High diversity - balanced community")
                    
                    # Get abundance data for visualization
                    abundances = df[species_cols].sum()
                    total = abundances.sum()
                    proportions = abundances / total
                    
                    # Create visualizations
                    fig_col1, fig_col2 = st.columns(2)
                    
                    with fig_col1:
                        # Pie chart of species proportions
                        fig_pie = px.pie(
                            values=proportions.values,
                            names=proportions.index,
                            title='Species Composition',
                            template=PLOTLY_TEMPLATE
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, width='stretch')
                    
                    with fig_col2:
                        # Bar chart of abundances
                        fig_bar = px.bar(
                            x=abundances.index,
                            y=abundances.values,
                            title='Species Abundances',
                            template=PLOTLY_TEMPLATE,
                            color=abundances.values,
                            color_continuous_scale='Greens'
                        )
                        fig_bar.update_layout(
                            xaxis_title='Species',
                            yaxis_title='Abundance',
                            showlegend=False
                        )
                        st.plotly_chart(fig_bar, width='stretch')
                    
                    # Rank-Abundance curve (Whittaker plot)
                    sorted_props = proportions.sort_values(ascending=False)
                    ranks = list(range(1, len(sorted_props) + 1))
                    
                    fig_rank = go.Figure()
                    fig_rank.add_trace(go.Scatter(
                        x=ranks,
                        y=sorted_props.values,
                        mode='lines+markers',
                        name='Rank-Abundance',
                        marker=dict(size=10, color='green'),
                        line=dict(width=2)
                    ))
                    
                    fig_rank.update_layout(
                        title='Rank-Abundance Curve (Whittaker Plot)',
                        xaxis_title='Species Rank',
                        yaxis_title='Relative Abundance',
                        yaxis_type='log',
                        template=PLOTLY_TEMPLATE,
                        height=350
                    )
                    st.plotly_chart(fig_rank, width='stretch')
                    
                    # Hill Diversity Profile (q vs effective number of species)
                    st.write("**Hill Diversity Profile:**")
                    st.caption("Shows how diversity changes with q parameter (weight given to common vs rare species)")
                    
                    # Calculate Hill numbers for different q values
                    q_values = np.linspace(0, 3, 31)
                    hill_numbers = []
                    
                    for q in q_values:
                        if q == 1:
                            # Limit as q->1 is exp(Shannon)
                            H = -np.sum(proportions * np.log(proportions + 1e-10))
                            hill_numbers.append(np.exp(H))
                        else:
                            # General Hill number formula: (sum(p^q))^(1/(1-q))
                            hill_q = np.sum(proportions ** q) ** (1 / (1 - q))
                            hill_numbers.append(hill_q)
                    
                    fig_hill = go.Figure()
                    fig_hill.add_trace(go.Scatter(
                        x=q_values,
                        y=hill_numbers,
                        mode='lines',
                        name='Hill Diversity Profile',
                        line=dict(width=3, color='darkgreen'),
                        fill='tozeroy',
                        fillcolor='rgba(0, 100, 0, 0.1)'
                    ))
                    
                    # Mark key points
                    key_q = [0, 1, 2]
                    key_names = ['Richness (q=0)', 'Shannon (q=1)', 'Simpson (q=2)']
                    key_values = [hill_numbers[0], hill_numbers[10], hill_numbers[20]]  # q=0, 1, 2
                    
                    fig_hill.add_trace(go.Scatter(
                        x=key_q,
                        y=key_values,
                        mode='markers+text',
                        marker=dict(size=12, color='red', symbol='diamond'),
                        text=key_names,
                        textposition='top center',
                        name='Key Indices',
                        showlegend=False
                    ))
                    
                    fig_hill.update_layout(
                        title='Hill Diversity Profile (Effective Number of Species vs q)',
                        xaxis_title='Order q (0=richness, 1=Shannon, 2=Simpson)',
                        yaxis_title='Effective Number of Species (qD)',
                        template=PLOTLY_TEMPLATE,
                        height=400,
                        annotations=[
                            dict(x=0.5, y=key_values[0]*0.9, text="← More weight to rare species", 
                                 showarrow=False, font=dict(size=10)),
                            dict(x=2.5, y=key_values[2]*1.1, text="More weight to common species →", 
                                 showarrow=False, font=dict(size=10))
                        ]
                    )
                    st.plotly_chart(fig_hill, width='stretch')
                    
                    # Diversity metrics table
                    st.write("**Diversity Metrics Summary:**")
                    metrics_df = pd.DataFrame({
                        'Metric': ["Shannon Index (H')", "Max Possible H'", "Evenness (J')", 
                                   "Species Richness (S)", "Hill q=0 (Richness)", 
                                   "Hill q=1 (exp Shannon)", "Hill q=2 (inv Simpson)", "Total Individuals"],
                        'Value': [f"{result['diversity_index']:.4f}", 
                                  f"{np.log(result['richness']):.4f}",
                                  f"{result['evenness']:.4f}",
                                  str(result['richness']),
                                  f"{hill_numbers[0]:.2f}",
                                  f"{hill_numbers[10]:.2f}",
                                  f"{hill_numbers[20]:.2f}",
                                  str(int(total))],
                        'Description': [
                            "Measure of diversity considering richness & evenness",
                            "ln(S) - diversity if all species equally abundant",
                            "H'/H'max - how evenly individuals distributed",
                            "Number of different species",
                            "All species counted equally (same as S)",
                            "Effective number of equally-common species",
                            "Weights common species more heavily",
                            "Sum of all individuals across species"
                        ]
                    })
                    st.dataframe(metrics_df, width='stretch', hide_index=True)
                    
                    # Download results
                    full_result = {
                        **result,
                        'species_abundances': abundances.to_dict(),
                        'species_proportions': proportions.to_dict(),
                        'total_individuals': int(total),
                        'hill_profile': {'q': q_values.tolist(), 'qD': hill_numbers},
                        'hill_q0': hill_numbers[0],
                        'hill_q1': hill_numbers[10],
                        'hill_q2': hill_numbers[20]
                    }
                    results_json = json.dumps(full_result, indent=2, default=str)
                    st.download_button(
                        "📥 Download Results (JSON)",
                        results_json,
                        "shannon_diversity_results.json",
                        "application/json",
                        key="download_shannon_json"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            elif not species_cols:
                st.warning("Select at least one species/abundance column.")


def render_report_tab():
    """Render the report generation tab"""
    st.header("📄 Report Generator")
    st.caption("Generate publication-ready analysis reports")
    
    df = st.session_state.df
    
    if df is None:
        st.info("📂 Please load data first in the Data tab.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Report Settings")
        
        report_title = st.text_input("Report Title", "Data Analysis Report", key="report_title")
        author = st.text_input("Author", "", key="report_author")
        include_data_summary = st.checkbox("Include Data Summary", True, key="inc_summary")
        include_results = st.checkbox("Include Analysis Results", True, key="inc_results")
        
        output_format = st.selectbox("Output Format", ["HTML", "Markdown"], key="report_format")
    
    with col2:
        st.subheader("Generate Report")
        
        if st.button("📄 Generate Report", key="gen_report"):
            try:
                report = ReportGenerator(title=report_title)
                
                if author:
                    report.add_section("Author", author, level=3)
                
                if include_data_summary:
                    report.add_data_provenance(df)
                
                if include_results and st.session_state.analysis_results:
                    report.add_section("Analysis Results", "Summary of performed analyses:")
                    
                    for key, value in st.session_state.analysis_results.items():
                        if isinstance(value, dict):
                            report.add_statistics_table(value, title=key.replace('_', ' ').title())
                
                if output_format == "HTML":
                    content = report.generate_html()
                    st.download_button(
                        "📥 Download HTML Report",
                        content,
                        file_name="analysis_report.html",
                        mime="text/html"
                    )
                else:
                    content = report.generate_markdown()
                    st.download_button(
                        "📥 Download Markdown Report",
                        content,
                        file_name="analysis_report.md",
                        mime="text/markdown"
                    )
                
                st.success("Report generated successfully!")
                
                with st.expander("Preview Report"):
                    if output_format == "HTML":
                        st.components.v1.html(content, height=400, scrolling=True)
                    else:
                        st.markdown(content)
                        
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point"""

    init_session_state()
    render_tutorial_sidebar()

    st.title("📊 Advanced Data Analysis Toolkit")
    st.caption("Version 4.0 - Scientific Research Edition with Interactive Plotly Charts")

    # =========================================================================
    # LEVEL 1: Main Category Tabs (8 groups - expanded for v4.0)
    # =========================================================================
    main_tabs = st.tabs([
        "📁 Data",
        "📊 Statistics",
        "🔊 Signal Processing",
        "⏱️ Time Series",
        "🤖 Machine Learning",
        "🔬 Scientific Tools",
        "📈 Visualization",
        "📄 Reports"
    ])

    # =========================================================================
    # 📁 DATA GROUP
    # =========================================================================
    with main_tabs[0]:
        data_subtabs = st.tabs([
            "📂 Data Loading",
            "🔍 Data Quality"
        ])
        with data_subtabs[0]:
            render_data_tab()
        with data_subtabs[1]:
            render_data_quality_tab()

    # =========================================================================
    # 📊 STATISTICS GROUP (5 subtabs - added Effect Sizes)
    # =========================================================================
    with main_tabs[1]:
        st.markdown("#### 📊 Statistics Group")
        st.caption("Descriptive statistics, hypothesis testing, effect sizes, and uncertainty quantification")

        stats_subtabs = st.tabs([
            "📊 Descriptive Statistics",
            "🧪 Hypothesis Tests",
            "📏 Effect Sizes",
            "📈 Bayesian Inference",
            "🎲 Uncertainty Analysis"
        ])

        with stats_subtabs[0]:
            render_statistical_tab()
        with stats_subtabs[1]:
            render_statistical_tests_tab()
        with stats_subtabs[2]:
            render_effect_sizes_tab()
        with stats_subtabs[3]:
            render_bayesian_tab()
        with stats_subtabs[4]:
            render_uncertainty_tab()

    # =========================================================================
    # 🔊 SIGNAL PROCESSING GROUP
    # =========================================================================
    with main_tabs[2]:
        st.markdown("#### 🔊 Signal Processing Group")
        st.caption("Frequency analysis using FFT, PSD, and Wavelet transforms (CWT/DWT)")
        render_signal_analysis_tab()

    # =========================================================================
    # ⏱️ TIME SERIES GROUP (5 subtabs - added CCF and Moving Average)
    # =========================================================================
    with main_tabs[3]:
        st.markdown("#### ⏱️ Time Series Group")
        st.caption("Temporal pattern analysis, stationarity testing, and causal relationships")

        ts_subtabs = st.tabs([
            "⏱️ Time Series Analysis",
            "📊 Cross-Correlation (CCF)",
            "📈 Moving Average",
            "🔗 Causality (Granger)",
            "🔮 Advanced (Prophet/DTW)"
        ])

        with ts_subtabs[0]:
            render_timeseries_tab()
        with ts_subtabs[1]:
            render_ccf_tab()
        with ts_subtabs[2]:
            render_moving_average_tab()
        with ts_subtabs[3]:
            render_causality_tab()
        with ts_subtabs[4]:
            render_advanced_ts_tab()

    # =========================================================================
    # 🤖 MACHINE LEARNING GROUP (reorganized into nested groups for better UX)
    # =========================================================================
    with main_tabs[4]:
        st.markdown("#### 🤖 Machine Learning Group")
        st.caption("Supervised learning, unsupervised learning, feature engineering, and computer vision")
        
        # Group ML subtabs into 3 logical categories for better navigation
        ml_category = st.radio(
            "Select ML Category:",
            ["🎯 Supervised Learning", "🔍 Unsupervised Learning", "🖼️ Computer Vision"],
            horizontal=True,
            key="ml_category"
        )
        
        st.markdown("---")
        
        if ml_category == "🎯 Supervised Learning":
            # Supervised learning: Regression, Classification, Validation, Feature Selection, Interpretability
            ml_super_tabs = st.tabs([
                "🤖 Regression/Classification",
                "✅ Model Validation",
                "🎯 Feature Selection",
                "🔮 Interpretability",
                "🔀 Non-Linear Methods"
            ])
            
            with ml_super_tabs[0]:
                render_ml_tab()
            with ml_super_tabs[1]:
                render_model_validation_tab()
            with ml_super_tabs[2]:
                render_feature_selection_tab()
            with ml_super_tabs[3]:
                render_interpretability_tab()
            with ml_super_tabs[4]:
                render_nonlinear_tab()
        
        elif ml_category == "🔍 Unsupervised Learning":
            # Unsupervised: PCA, Clustering, Anomaly, Dim Reduction
            ml_unsup_tabs = st.tabs([
                "🔬 PCA",
                "🎯 Clustering",
                "🚨 Anomaly Detection",
                "📉 Dimensionality Reduction"
            ])
            
            with ml_unsup_tabs[0]:
                render_pca_tab()
            with ml_unsup_tabs[1]:
                render_clustering_tab()
            with ml_unsup_tabs[2]:
                render_anomaly_tab()
            with ml_unsup_tabs[3]:
                render_dimreduction_tab()
        
        else:  # Computer Vision
            # Image-based: Image Recognition, Biomass Segmentation
            ml_vision_tabs = st.tabs([
                "🖼️ Image Recognition",
                "🌿 Biomass Segmentation"
            ])
            
            with ml_vision_tabs[0]:
                render_image_tab()
            with ml_vision_tabs[1]:
                render_biomass_tab()

    # =========================================================================
    # 🔬 SCIENTIFIC TOOLS GROUP (v4.0 - NEW)
    # =========================================================================
    with main_tabs[5]:
        st.markdown("#### 🔬 Scientific Tools (v4.0)")
        st.caption("Survival analysis, domain-specific methods for environmental, clinical, and ecology research")

        sci_subtabs = st.tabs([
            "⏳ Survival Analysis",
            "🌿 Domain-Specific"
        ])

        with sci_subtabs[0]:
            render_survival_tab()
        with sci_subtabs[1]:
            render_domain_specific_tab()

    # =========================================================================
    # 📈 VISUALIZATION GROUP
    # =========================================================================
    with main_tabs[6]:
        st.markdown("#### 📈 Visualization Group")
        st.caption("Interactive charts, scatter plots, distributions, and regression visualization")
        render_visualization_tab()

    # =========================================================================
    # 📄 REPORTS GROUP (v4.0 - NEW)
    # =========================================================================
    with main_tabs[7]:
        st.markdown("#### 📄 Report Generation (v4.0)")
        st.caption("Generate publication-ready HTML and Markdown reports")
        render_report_tab()

    st.markdown("---")
    st.caption("💡 All charts are **interactive**: zoom, pan, hover, download! | v4.0 Scientific Research Edition")


if __name__ == "__main__":
    main()
