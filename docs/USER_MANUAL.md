# 📘 Advanced Data Analysis Toolkit - User Manual

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Tab: Data Loading](#tab-data-loading)
4. [Tab: Statistical Analysis](#tab-statistical-analysis)
5. [Tab: Machine Learning](#tab-machine-learning)
6. [Tab: Neural Networks](#tab-neural-networks) 🧠
7. [Tab: Bayesian Analysis](#tab-bayesian-analysis)
8. [Tab: Uncertainty Analysis](#tab-uncertainty-analysis)
9. [Tab: Non-Linear Analysis](#tab-non-linear-analysis)
10. [Tab: Time Series](#tab-time-series)
11. [Tab: Causality Analysis](#tab-causality-analysis)
12. [Tab: Visualizations](#tab-visualizations)
13. [Tab: Scientific Tools](#tab-scientific-tools)
    - [Community Ecology](#community-ecology)
    - [Ordination](#ordination)
    - [Multivariate Analysis](#multivariate-analysis)
    - [Curve Fitting](#curve-fitting)
14. [Tab: Plugins](#tab-plugins)
15. [Rust Acceleration Toggle](#rust-acceleration-toggle)
16. [Tips and Best Practices](#tips-and-best-practices)
17. [Troubleshooting](#troubleshooting)
18. [New in v4.3: Scientific Research Features](#new-in-v43-scientific-research-features)

---

## Getting Started

### Launching the Application

```bash
# Option 1: Using entry point (after pip install -e .)
data-toolkit

# Option 2: Direct run
python run.py

# Option 3: Module
python -m data_toolkit.main_gui
```

### Basic Workflow

1. **Load Data**: Go to "📁 Data Loading" tab → Click "📂 Load CSV File"
2. **Select Columns**: Choose feature columns (Ctrl+Click for multiple) and target column
3. **Analyze**: Navigate to analysis tabs and click buttons to run analyses
4. **View Results**: Results appear in text panels; plots open in new windows

---

## Interface Overview

### Header Bar

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📊 Advanced Data Analysis Toolkit    [Status] ☑ 🦀 Rust Acceleration│
│                                                    v8.0 - Unified   │
└─────────────────────────────────────────────────────────────────────┘
```

- **Title**: Application name
- **Backend Status**: Shows "⚡ Backend: Rust (Fast)" or "🐍 Backend: Python"
- **Rust Toggle**: Checkbox to enable/disable Rust acceleration (if compiled)
- **Version**: Current version number

### Common Elements

| Element | Description |
|---------|-------------|
| **Feature Columns Listbox** | Multi-select list of columns to analyze |
| **Target Dropdown** | Single column selection for supervised learning |
| **Results Panel** | Text area showing analysis output |
| **Buttons** | Click to run specific analyses |

---

## Tab: Data Loading

**Purpose**: Load data files and preview your dataset.

### Components

#### 📂 Load CSV File (Button)
- Opens file dialog to select CSV or Excel files
- Supported formats: `.csv`, `.xlsx`, `.xls`
- After loading, displays file info and data preview

#### File Info Panel
Shows:
- **Filename**: Name of loaded file
- **Rows**: Number of data rows
- **Columns**: Number of columns
- **Memory**: Memory usage of dataset

#### Column Selection Panel
- **Available Columns**: List of all columns in your data
- **Ctrl+Click**: Select multiple columns as features
- **Target Dropdown**: Select one column as target variable

#### Data Preview Table
- Shows first rows of your data
- Scrollable horizontally and vertically
- Column headers displayed

#### Quick Plot Panel
- Automatically plots first numeric column vs second
- Shows linear regression line
- Updates when you change column selection

### Usage Example

1. Click "📂 Load CSV File"
2. Select `general_analysis_data.csv`
3. Observe: 500 rows, 7 columns displayed
4. Ctrl+Click to select `feature_1`, `feature_2`, `feature_3`
5. Select `target` from Target dropdown

---

## Tab: Statistical Analysis

**Purpose**: Descriptive statistics, correlations, and distribution analysis.

### Buttons

#### 📊 Descriptive Stats
**What it does**: Calculates summary statistics for selected columns.

**Output includes**:
- Count, Mean, Std, Min, Max
- 25th, 50th (median), 75th percentiles
- Skewness, Kurtosis
- Missing values count

**When to use**: First step in any analysis to understand your data.

---

#### 🔗 Correlation Matrix
**What it does**: Computes pairwise correlations between all selected features.

**Methods available**: Pearson (linear), Spearman (rank), Kendall (ordinal)

**Output**: Matrix showing correlation coefficients (-1 to +1)

**Interpretation**:
- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

---

#### 📈 Cross Correlation
**What it does**: Calculates correlation between each feature and the target.

**Output**: List of features ranked by correlation strength.

**When to use**: To identify which features are most related to your target.

---

#### ⏱️ Lag Analysis
**What it does**: Computes correlation at different time lags.

**Parameter**: Max Lag (default: 10)

**Output**: Correlation values at each lag from -max_lag to +max_lag.

**When to use**: Time series data to find delayed relationships.

---

#### 📉 Distribution Analysis
**What it does**: Analyzes the statistical distribution of each column.

**Output includes**:
- Normality test (Shapiro-Wilk p-value)
- Skewness and Kurtosis
- Histogram visualization

**Interpretation**: p-value < 0.05 suggests non-normal distribution.

---

#### 🎯 Outlier Detection
**What it does**: Identifies outliers using IQR or Z-score method.

**Methods**:
- **IQR**: Points below Q1-1.5×IQR or above Q3+1.5×IQR
- **Z-score**: Points with |z| > 3

**Output**: Count and percentage of outliers per column.

---

## Tab: Machine Learning

**Purpose**: Train regression models and perform clustering/dimensionality reduction.

### Model Selection Dropdown
Choose from:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Gradient Boosting

### Buttons

#### 🎯 Train Model
**What it does**: Trains selected model on your data.

**Output includes**:
- R² Score (coefficient of determination)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Model coefficients (for linear models)

**Interpretation**: R² closer to 1.0 = better fit.

---

#### 🔄 Cross Validation
**What it does**: Evaluates model using k-fold cross-validation.

**Parameter**: Number of folds (default: 5)

**Output**: Mean R² and standard deviation across folds.

**When to use**: To get reliable estimate of model performance.

---

#### 📊 Feature Importance
**What it does**: Ranks features by their importance to the model.

**Methods**:
- Linear models: Absolute coefficient values
- Tree models: Built-in feature importance
- Permutation importance (all models)

**Output**: Bar chart and ranked list of features.

---

#### 🎨 PCA Analysis
**What it does**: Principal Component Analysis for dimensionality reduction.

**Output includes**:
- Explained variance ratio per component
- Cumulative explained variance
- Component loadings
- Scatter plot of first 2 components

**When to use**: High-dimensional data, visualization, feature reduction.

---

#### 🔮 K-Means Clustering
**What it does**: Groups data into k clusters.

**Parameter**: Number of clusters (default: 3)

**Output**:
- Cluster assignments
- Cluster centers
- Inertia (within-cluster sum of squares)
- Visualization of clusters

---

#### 🌐 DBSCAN Clustering
**What it does**: Density-based clustering (finds clusters of arbitrary shape).

**Parameters**:
- eps: Maximum distance between points (default: 0.5)
- min_samples: Minimum points per cluster (default: 5)

**Output**: Cluster labels (-1 = noise/outlier)

**When to use**: Unknown number of clusters, non-spherical clusters.

---

## Tab: Neural Networks 🧠

**Purpose**: Deep learning models for regression, classification, time series forecasting, and anomaly detection.

> ⚠️ **Requirement**: TensorFlow must be installed (`pip install tensorflow`)

### Model Types

#### 🧠 MLP Regressor
**What it does**: Multi-Layer Perceptron for regression tasks.

**Parameters**:
- Hidden Layers: Comma-separated layer sizes (e.g., "64,32")
- Epochs: Number of training iterations (default: 100)
- Batch Size: Samples per gradient update (default: 32)
- Validation Split: Fraction for validation (default: 0.2)

**Output**:
- Training/Validation loss curves
- MSE, MAE, R² score on test set
- Model architecture summary

**When to use**: Non-linear regression, complex feature relationships.

---

#### 🧠 MLP Classifier
**What it does**: Multi-Layer Perceptron for classification tasks.

**Parameters**: Same as MLP Regressor

**Output**:
- Training/Validation loss curves
- Accuracy, Precision, Recall, F1-score
- Confusion matrix

**When to use**: Multi-class classification, pattern recognition.

---

#### 📈 LSTM Forecast
**What it does**: Long Short-Term Memory network for time series forecasting.

**Parameters**:
- LSTM Lookback: Number of past time steps to use (default: 10)
- Forecast Horizon: Steps to predict ahead (default: 5)
- Epochs: Training iterations (default: 100)
- Batch Size: Samples per update (default: 32)

**Output**:
- Actual vs Predicted comparison
- Future forecast values
- Test MSE and MAE

**When to use**: Time series with sequential patterns, stock prices, sensor data.

---

#### 🚨 Autoencoder Anomaly Detection
**What it does**: Learns normal patterns and flags anomalies via reconstruction error.

**Parameters**:
- Encoding Dimension: Bottleneck size (default: 8)
- Epochs: Training iterations (default: 100)
- Contamination: Expected anomaly rate (default: 0.05)

**Output**:
- Reconstruction error plot
- Threshold line
- Anomaly indices and count
- Anomaly percentage

**When to use**: Detecting fraud, equipment failures, unusual patterns.

### API Usage

```python
from data_toolkit import NeuralNetworkModels

nn = NeuralNetworkModels(df)

# MLP Regressor
results = nn.mlp_regressor(
    features=['col1', 'col2'],
    target='target',
    hidden_layers=[64, 32],
    epochs=100
)

# LSTM Forecast
results = nn.lstm_forecast(
    column='value',
    lookback=10,
    forecast_horizon=5
)

# Autoencoder
results = nn.autoencoder_anomaly_detection(
    features=['col1', 'col2', 'col3'],
    encoding_dim=8,
    contamination=0.05
)
```

---

## Tab: Bayesian Analysis

**Purpose**: Bayesian regression and uncertainty quantification.

### Buttons

#### 🎲 Bayesian Regression
**What it does**: Fits linear model with Bayesian inference.

**Output includes**:
- Posterior mean coefficients
- Posterior standard deviations
- 95% credible intervals
- R² score

**Interpretation**: Credible interval = 95% probability true value lies within.

---

#### 📊 Credible Intervals
**What it does**: Calculates Bayesian credible intervals for predictions.

**Parameter**: Confidence level (default: 0.95)

**Output**: Lower and upper bounds for each coefficient.

---

#### 📈 Posterior Distribution
**What it does**: Visualizes the posterior distribution of coefficients.

**Output**: Histogram/density plot for each coefficient.

**When to use**: Understanding uncertainty in parameter estimates.

---

#### ⚖️ Model Comparison
**What it does**: Compares multiple models using Bayesian criteria.

**Output**:
- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)

**Interpretation**: Lower BIC/AIC = better model (penalizes complexity).

---

#### 🔬 Prior Sensitivity
**What it does**: Tests how sensitive results are to prior assumptions.

**Output**: Coefficient estimates under different prior settings.

**When to use**: Checking robustness of Bayesian analysis.

---

## Tab: Uncertainty Analysis

**Purpose**: Quantify uncertainty in model predictions and parameters.

### Parameters
- **Confidence Level**: 0.90, 0.95, or 0.99
- **Bootstrap Samples**: Number of resamples (default: 1000)

### Buttons

#### 🔄 Bootstrap CI
**What it does**: Estimates confidence intervals via bootstrap resampling.

**How it works**:
1. Resample data with replacement
2. Fit model on each resample
3. Compute percentiles of coefficient distribution

**Output**:
- Mean coefficients
- Standard errors
- Confidence interval bounds
- Histogram of bootstrap distribution

**When to use**: Non-parametric confidence intervals, small samples.

---

#### 📊 Prediction Intervals
**What it does**: Calculates intervals for future predictions.

**Output**: For each test point, shows predicted value ± interval.

**Interpretation**: Future observations should fall within interval X% of time.

---

#### 📈 Confidence Bands
**What it does**: Shows uncertainty band around regression line.

**Output**: Plot with regression line and shaded confidence region.

---

#### 🔗 Error Propagation
**What it does**: Propagates uncertainty through calculations.

**Output**: How input uncertainties affect output uncertainty.

---

#### 🎯 Residual Analysis
**What it does**: Analyzes model residuals for patterns.

**Output includes**:
- Residual vs Fitted plot
- Q-Q plot (normality check)
- Durbin-Watson statistic (autocorrelation)
- Breusch-Pagan test (heteroscedasticity)

**Interpretation**:
- Durbin-Watson ≈ 2: No autocorrelation
- Breusch-Pagan p < 0.05: Heteroscedasticity present

---

#### 🎲 Monte Carlo
**What it does**: Monte Carlo simulation for prediction uncertainty.

**Parameter**: Number of simulations

**Output**: Distribution of predictions under parameter uncertainty.

---

## Tab: Non-Linear Analysis

**Purpose**: Detect and model non-linear relationships.

### Buttons

#### 🔮 Mutual Information
**What it does**: Measures information shared between features and target.

**Output**: MI score for each feature (higher = more information).

**Advantage over correlation**: Captures non-linear relationships.

---

#### 📊 Distance Correlation
**What it does**: Measures both linear AND non-linear dependence.

**Output**: Distance correlation coefficient (0 to 1).

**Key property**:
- Distance correlation = 0 ⟺ Independence
- Unlike Pearson, detects non-linear relationships

**Example**:
| Relationship | Pearson r | Distance Corr |
|--------------|-----------|---------------|
| y = x² | ~0 | ~0.7 |
| y = sin(x) | Low | High |

---

#### 🌊 Gaussian Process
**What it does**: Non-parametric regression with uncertainty.

**Output**:
- Predicted mean function
- Confidence band (±2σ)
- Plot showing GP fit

**When to use**: Unknown functional form, need uncertainty estimates.

---

#### 📈 Polynomial Regression
**What it does**: Fits polynomial of specified degree.

**Parameter**: Degree (1=linear, 2=quadratic, etc.)

**Output**: Coefficients, R², plot of fit.

**Warning**: High degrees can overfit.

---

#### 〰️ Spline Regression
**What it does**: Fits smooth spline curve.

**Output**: Flexible curve that follows data shape.

**When to use**: Smooth non-linear trends.

---

#### 🧠 Neural Network
**What it does**: Fits multilayer perceptron regressor.

**Output**: R² score, predictions.

**Note**: May require scaling/normalization.

---

#### 🎯 SVM Regression
**What it does**: Support Vector Machine for regression.

**Kernels**: Linear, RBF, Polynomial

**When to use**: Complex non-linear relationships.

---

## Tab: Time Series

**Purpose**: Analyze temporal patterns and dependencies.

### Parameter
- **Max Lag**: Maximum lag for ACF/PACF (default: 20)

### Buttons

#### 📊 ACF Plot
**What it does**: Plots Autocorrelation Function.

**Interpretation**:
- Shows correlation of series with lagged versions
- Significant spikes indicate dependency at that lag
- Blue shaded area = confidence bounds

**Pattern recognition**:
- Slow decay → Non-stationary or trend
- Cutoff after lag k → MA(k) process
- Seasonal spikes → Seasonality present

---

#### 📈 PACF Plot
**What it does**: Plots Partial Autocorrelation Function.

**Interpretation**:
- Direct correlation controlling for intermediate lags
- Cutoff after lag p → AR(p) process

**Using ACF + PACF together**:
| ACF | PACF | Suggests |
|-----|------|----------|
| Tails off | Cuts off at p | AR(p) |
| Cuts off at q | Tails off | MA(q) |
| Tails off | Tails off | ARMA |

---

#### 🔬 Stationarity Test
**What it does**: Augmented Dickey-Fuller test for stationarity.

**Output**:
- ADF statistic
- p-value
- Critical values (1%, 5%, 10%)

**Interpretation**:
- p < 0.05 → Reject null → Series IS stationary
- p ≥ 0.05 → Cannot reject → Series may be non-stationary

**If non-stationary**: Try differencing the data.

---

#### 🔄 Decomposition
**What it does**: Separates series into Trend + Seasonal + Residual.

**Models**:
- Additive: Y = Trend + Seasonal + Residual
- Multiplicative: Y = Trend × Seasonal × Residual

**Output**: 4-panel plot showing original, trend, seasonal, residual.

---

#### 📉 Rolling Stats
**What it does**: Calculates rolling mean and standard deviation.

**Parameter**: Window size

**Output**: Plot showing original series with rolling statistics.

**Use**: Visual check for stationarity (constant mean/variance).

---

#### 🎯 ARIMA Model
**What it does**: Fits AutoRegressive Integrated Moving Average model.

**Parameters**: p (AR order), d (differencing), q (MA order)

**Output**:
- Model summary
- Fitted values
- Residual diagnostics

**Tips for choosing p, d, q**:
- d: Number of differences for stationarity
- p: From PACF cutoff
- q: From ACF cutoff

---

## Tab: Causality Analysis

**Purpose**: Test causal relationships between variables.

### Parameter
- **Max Lag**: Maximum lag to test (default: 10)

### Buttons

#### 🔬 Granger Causality
**What it does**: Tests if one variable helps predict another.

**Null hypothesis**: X does NOT Granger-cause Y.

**Output**:
- F-statistic and p-value for each lag
- Conclusion at each lag

**Interpretation**:
- p < 0.05 → X Granger-causes Y at that lag
- Does NOT imply true causation, only predictive relationship

---

#### ⏱️ Lead-Lag Analysis
**What it does**: Finds optimal lag for maximum correlation.

**Output for each feature**:
- Correlation at each lag
- Best lag (highest |correlation|)
- Direction (feature leads or target leads)

**Interpretation**:
- Best lag < 0: Feature leads target
- Best lag > 0: Target leads feature
- Best lag = 0: Contemporaneous relationship

---

#### 📊 Correlation at Lags
**What it does**: Shows correlation structure across multiple lags.

**Output**: Table of correlations at each lag.

**Visualization**: Line plot of correlation vs lag.

---

## Tab: Visualizations

**Purpose**: Create various plots and charts.

### Buttons

#### 🎨 Scatter Matrix
**What it does**: Grid of scatter plots for all pairs of selected variables.

**Diagonal**: KDE (density) plots showing distribution of each variable.

**Off-diagonal**: Scatter plots showing pairwise relationships.

**When to use**: Exploring relationships in multivariate data.

---

#### 🔥 Correlation Heatmap
**What it does**: Color-coded matrix of correlations.

**Color scale**: Red (negative) → White (zero) → Blue (positive)

**When to use**: Quick overview of all correlations.

---

#### 📦 Box Plots
**What it does**: Box-and-whisker plots for selected columns.

**Shows**:
- Median (center line)
- IQR (box)
- Whiskers (1.5×IQR)
- Outliers (points beyond whiskers)

**When to use**: Comparing distributions, spotting outliers.

---

#### 📊 FFT Spectrum
**What it does**: Fast Fourier Transform for frequency analysis.

**Output**: Power spectrum showing dominant frequencies.

**When to use**: Finding periodic patterns in data.

---

#### 🔇 Noise Filter
**What it does**: Applies Savitzky-Golay filter to smooth data.

**Output**: Original vs filtered signal overlay.

**When to use**: Removing noise while preserving shape.

---

#### 🌐 3D Scatter
**What it does**: Three-dimensional scatter plot.

**Requires**: At least 3 columns selected.

**Controls**: Click and drag to rotate view.

---

## Tab: Scientific Tools

**Purpose**: Specialised methods for community ecology, multivariate ordination, multivariate hypothesis testing, and non-linear curve fitting.

**Location**: 🔬 Scientific Tools main tab → subtabs.

### Community Ecology

**Purpose**: Quantify and compare biological community composition.

#### Alpha Diversity
**What it does**: Calculates within-sample diversity indices.

| Index | Formula | Interpretation |
|-------|---------|----------------|
| **Shannon (H')** | $-\sum p_i \ln p_i$ | Entropy-based; higher = more diverse |
| **Simpson (1-D)** | $1 - \sum p_i^2$ | Probability two random individuals differ |
| **Inverse Simpson** | $1 / \sum p_i^2$ | Effective number of equally common species |
| **Margalef** | $(S-1) / \ln N$ | Species richness corrected for sample size |
| **Pielou's Evenness** | $H' / \ln S$ | 0–1, how evenly individuals are distributed |
| **Chao1** | Richness estimator using singletons/doubletons | Estimates true richness including unobserved species |

**When to use**: Comparing diversity across sites, treatments, or time periods.

#### Beta Diversity
**What it does**: Measures community compositional turnover between sites.

| Metric | Range | What it measures |
|--------|-------|------------------|
| **Bray-Curtis** | 0–1 | Quantitative dissimilarity (abundance data) |
| **Jaccard** | 0–1 | Presence/absence dissimilarity |
| **Sørensen** | 0–1 | Presence/absence (double-weights shared species) |
| **Morisita-Horn** | 0–1 | Abundance-based, sample-size independent |

**Output**: Distance matrix + heatmap visualization.

#### Rarefaction
**What it does**: Estimates species richness at standardised sample sizes.

**When to use**: Comparing richness across samples with unequal sampling effort. The rarefaction curve flattens when additional sampling yields few new species.

#### Species Accumulation Curves
**What it does**: Shows how total observed species richness grows as more samples are added (permutation-based mean ± SD).

#### SHE Analysis
**What it does**: Decomposes Shannon diversity into its richness (S) and evenness (E) components across cumulative samples.

---

### Ordination

**Purpose**: Reduce high-dimensional community/environmental data to 2–3 interpretable axes.

#### Available Methods

| Method | Type | Best for |
|--------|------|----------|
| **PCoA** (Principal Coordinates Analysis) | Unconstrained | Any distance metric → Euclidean embedding; preserves original distances |
| **NMDS** (Non-metric Multidimensional Scaling) | Unconstrained | Preserves rank-order of distances; robust to non-linearity |
| **CA** (Correspondence Analysis) | Unconstrained | Species × sites matrices with unimodal species response |
| **DCA** (Detrended Correspondence Analysis) | Unconstrained | CA with arch-effect removal for long ecological gradients |
| **CCA** (Canonical Correspondence Analysis) | Constrained | Relates species composition to environmental variables |
| **RDA** (Redundancy Analysis) | Constrained | Linear species response constrained by environmental variables |

#### Interpreting Ordination Plots
- Points close together = similar composition
- Axis % = proportion of variance (or inertia) explained
- **Biplot arrows** (CCA/RDA): direction and length show environmental variable effects
- **Species scores** (CA/DCA/CCA): show species optima along gradients

#### Mantel Test
**What it does**: Tests correlation between two distance matrices via permutation.

**Output**: Mantel *r* statistic (−1 to +1) and *p*-value.

**When to use**: Testing whether spatial distance predicts community dissimilarity, or whether two distance matrices are correlated.

---

### Multivariate Analysis

**Purpose**: Hypothesis testing on multivariate data — do groups differ in their multivariate composition?

#### PERMANOVA (Permutational MANOVA)
**What it does**: Tests for differences among groups using distance matrices and permutation (Anderson, 2001).

**Output**: Pseudo-F, p-value, R² (proportion of variation explained by grouping).

**Assumptions**: No distributional assumptions; only requires exchangeability under H₀.

**When to use**: The default choice for testing community composition differences among treatments or sites.

#### ANOSIM (Analysis of Similarities)
**What it does**: Rank-based test comparing between-group vs within-group dissimilarities (Clarke, 1993).

**Output**: R statistic (−1 to +1), p-value.

**Interpretation**: R = 1 = complete separation; R ≈ 0 = no difference; R < 0 = more variation within groups.

#### SIMPER (Similarity Percentages)
**What it does**: Identifies which species/variables contribute most to between-group Bray-Curtis dissimilarity.

**Output**: Per-species contribution (%), cumulative %, mean contribution ± SD for each pair of groups.

**When to use**: After a significant PERMANOVA — "which species drive the difference?"

#### MANOVA (Multivariate ANOVA)
**What it does**: Parametric multivariate test using Wilks' Lambda, Pillai's Trace, and Hotelling-Lawley Trace.

**Assumptions**: Multivariate normality, homogeneity of covariance matrices.

**When to use**: When data satisfy normality assumptions and you want classical test statistics.

#### Hotelling's T²
**What it does**: Multivariate two-sample t-test.

**When to use**: Comparing multivariate means of exactly 2 groups.

#### Discriminant Analysis (LDA / CVA)
**What it does**: Finds linear combinations that best separate groups; provides classification accuracy, confusion matrix, and ordination plot in discriminant space.

---

### Curve Fitting

**Purpose**: Fit specialised non-linear models beyond polynomial regression.

#### Available Models

| Model | Equation | Typical Use |
|-------|----------|-------------|
| **Power (Allometric)** | $y = a \cdot x^b$ | Species-area, allometric scaling |
| **Exponential (2-param)** | $y = a \cdot e^{bx}$ | Growth/decay without asymptote |
| **Exponential (3-param)** | $y = a \cdot e^{bx} + c$ | Growth/decay with baseline offset |
| **Logistic (4-param)** | $y = d + \frac{a - d}{1 + (x/c)^b}$ | Dose-response, population growth |
| **Sinusoidal** | $y = A \sin(2\pi f x + \phi) + \text{offset}$ | Seasonal/cyclical patterns |
| **Gompertz** | $y = a \cdot e^{-b \cdot e^{-cx}}$ | Asymmetric sigmoid, tumour growth |
| **RMA Regression** | Type II regression | Both variables have measurement error |
| **GLM** | $g(E[y]) = X\beta$ | Generalized linear model with link function |

#### Model Comparison
**Multi-Model Comparison** mode fits all selected models and compares by AIC, BIC, R², and RMSE. Best model is highlighted.

#### Residual Diagnostics
Each fit includes a residuals-vs-fitted plot to check for systematic patterns.

---

## Tab: Plugins

**Purpose**: Extend toolkit with custom processing functions.

### Left Panel - Plugin List

#### Loaded Plugins List
- Shows all currently loaded plugins
- Icons indicate category (🔧 preprocessing, 📊 analysis, 📈 visualization)
- Click to select and view details

#### 📁 Load File (Button)
Opens file dialog to load a `.py` plugin file.

#### 📋 New/Paste (Button)
Creates new plugin from template or paste your code.

#### ❌ Remove (Button)
Unloads selected plugin.

#### Examples Dropdown
Select built-in example plugins:
- Z-Score Normalization
- Custom Weighted Correlation
- Rolling Window Features

### Right Panel - Code Editor

#### Plugin Code Text Area
- View and edit plugin source code
- Syntax: Python

#### 💾 Load/Update Plugin (Button)
Parses code and registers the plugin.

#### 📝 Template (Button)
Inserts blank plugin template.

#### 💾 Save to File (Button)
Exports plugin code to a `.py` file.

### Execution Section

#### Parameters Frame
- Auto-generated input fields for plugin parameters
- Types: float, int, str, bool
- Dropdowns for choice parameters

#### ▶️ Run Plugin (Button)
Executes plugin on current data with selected columns.

#### 📊 Apply to Data (Button)
Replaces main dataset with plugin output (if DataFrame).

### Results Panel
Shows:
- Success/error messages
- Plugin output summary
- New columns created (if any)

---

## Rust Acceleration Toggle

### Location
Header bar, right side: **☑ 🦀 Rust Acceleration**

### States

| Checkbox | Status Display | Meaning |
|----------|----------------|---------|
| ☑ (checked) | ⚡ Backend: Rust (Fast) | Using compiled Rust code |
| ☐ (unchecked) | 🐍 Backend: Python | Using pure Python |
| ☐ (disabled) | 🐍 Backend: Python (Rust not compiled) | Rust unavailable |

### Accelerated Operations
When Rust is enabled, these run faster:
- Distance correlation (10-50x)
- Bootstrap CI (5-20x)
- Monte Carlo simulations (10-30x)
- Lead-lag correlations (3-10x)
- Rolling statistics (3-10x)
- Outlier detection (5-20x)

### When to Use Each

**Use Rust (checked)**:
- Large datasets (>10,000 rows)
- Many bootstrap iterations (>1000)
- Distance correlation on many features
- Real-time analysis needs

**Use Python (unchecked)**:
- Debugging (easier stack traces)
- Comparing results between implementations
- When Rust gives unexpected results

---

## Tips and Best Practices

### Data Preparation
1. **Check for missing values** before analysis
2. **Scale features** for neural networks and SVM
3. **Handle outliers** before fitting models
4. **Transform skewed** variables if needed

### Feature Selection
1. Start with **correlation analysis** to find relevant features
2. Use **mutual information** for non-linear relationships
3. Check **VIF** for multicollinearity
4. Use **regularization** (Ridge/Lasso) with many features

### Model Selection
1. Start with **Linear Regression** as baseline
2. Try **Random Forest** for non-linear patterns
3. Use **cross-validation** to compare models
4. Check **residuals** for model assumptions

### Time Series
1. Always **test stationarity** first
2. **Difference** non-stationary series
3. Use **ACF/PACF** to determine ARIMA orders
4. Check residuals for **remaining autocorrelation**

### Reporting
1. Include **confidence intervals**, not just point estimates
2. Report **R²** and **RMSE** together
3. Show **residual plots** to validate models
4. Note **sample size** and any data transformations

---

## Troubleshooting

### Common Issues

#### "No data loaded"
**Solution**: Go to Data Loading tab and load a CSV file first.

#### "Please select columns"
**Solution**: Ctrl+Click to select feature columns in the column list.

#### "Please select a target"
**Solution**: Choose a target column from the dropdown.

#### Empty plot window appears
**Solution**: Close the empty window; the actual plot should appear next. (This has been fixed in v8.)

#### Rust toggle is disabled
**Cause**: Rust extensions not compiled.
**Solution**:
```bash
cd rust_extensions
pip install maturin
maturin develop --release
```

#### "Module not found" errors
**Solution**: Ensure you installed the package:
```bash
pip install -e .
```

#### Plots don't appear
**Solution**: Check if matplotlib backend is set correctly. The application uses TkAgg.

#### Analysis is very slow
**Solutions**:
- Enable Rust acceleration
- Reduce bootstrap samples (e.g., 500 instead of 1000)
- Select fewer columns
- Use smaller dataset for exploration

### Getting Help

1. Check the `test_data/README.md` for expected results
2. Run tests: `pytest tests/`
3. Check Jupyter notebook for usage examples
4. Review example plugins for custom extensions

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Click | Multi-select columns |
| Ctrl+A | Select all (in text fields) |
| Escape | Close plot windows |

---

## Version History

- **v4.0**: Scientific research enhancements (NEW!)
  - Effect sizes with confidence intervals
  - Model validation (nested CV, calibration, ROC analysis)
  - Data quality analysis and multiple imputation
  - Feature selection (RFE, Boruta, SHAP, ensemble)
  - Survival analysis (Kaplan-Meier, Cox regression)
  - Model interpretability (SHAP, LIME, PDP)
  - Advanced time series (Prophet, changepoints, DTW)
  - Domain-specific tools (environmental, clinical, ecology)
  - Automated report generation
  - Multiple testing correction, VIF, robust statistics
- **v8.0**: Unified edition with Rust toggle, plugin system, white theme
- **v7.0**: Modular architecture, dark theme
- **v6.0**: Added causality analysis
- **v5.0**: Added time series module
- **v4.0**: Added Bayesian analysis
- **v3.0**: Added uncertainty quantification
- **v2.0**: Added ML models
- **v1.0**: Initial release with statistical analysis

---

## New in v4.0: Scientific Research Features

### Effect Size Calculations

Calculate standardized effect sizes with confidence intervals for proper scientific reporting:

```python
from data_toolkit.effect_sizes import EffectSizeCalculator

calc = EffectSizeCalculator(df)

# Cohen's d with 95% CI
result = calc.cohens_d('treatment', 'control')
print(f"d = {result['cohens_d']:.3f} ({result['interpretation']})")

# Hedges' g (small-sample correction)
result = calc.hedges_g('group1', 'group2')

# Cramér's V for categorical data
result = calc.cramers_v(contingency_table)
```

### Model Validation

Rigorous validation for reproducible research:

```python
from data_toolkit.model_validation import ModelValidator

validator = ModelValidator(df)

# Nested cross-validation (unbiased hyperparameter tuning)
result = validator.nested_cross_validation(
    model, param_grid, features, target
)

# Calibration analysis for classifiers
result = validator.calibration_analysis(model, features, target)

# Comprehensive residual diagnostics
result = validator.residual_diagnostics(model, features, target)
```

### Data Quality Analysis

Comprehensive missing data and quality assessment:

```python
from data_toolkit.data_quality import DataQualityAnalyzer

dqa = DataQualityAnalyzer(df)

# Little's MCAR test
result = dqa.little_mcar_test()

# Multiple imputation with Rubin's rules
result = dqa.multiple_imputation(columns, n_imputations=5)

# Comprehensive quality report
report = dqa.generate_quality_report()
```

### Feature Selection

Multiple methods for optimal feature subset:

```python
from data_toolkit.feature_selection import FeatureSelector

selector = FeatureSelector(df)

# Ensemble method (combines multiple approaches)
result = selector.ensemble_selection(features, target)
print("Consensus features:", result['consensus_features'])
```

### Survival Analysis

Time-to-event analysis (requires `lifelines`):

```python
from data_toolkit.survival_analysis import SurvivalAnalyzer

surv = SurvivalAnalyzer(df)

# Kaplan-Meier curves
result = surv.kaplan_meier('time', 'event', group_col='treatment')

# Cox proportional hazards
result = surv.cox_regression('time', 'event', covariates)
```

### Model Interpretability

Explain model predictions (requires `shap`, `lime`):

```python
from data_toolkit.interpretability import ModelInterpreter

interp = ModelInterpreter(model, df, features)

# SHAP analysis
result = interp.shap_analysis()

# LIME local explanation
result = interp.lime_explain(instance_idx=42)
```

### Advanced Time Series

Prophet, changepoints, and DTW (requires optional packages):

```python
from data_toolkit.advanced_timeseries import AdvancedTimeSeriesAnalysis

ats = AdvancedTimeSeriesAnalysis(df)

# Prophet forecast
result = ats.prophet_forecast('date', 'value', periods=30)

# Changepoint detection
result = ats.detect_changepoints('column', method='pelt')
```

### Domain-Specific Analysis

Tools for environmental science, clinical research, and ecology:

```python
from data_toolkit.domain_specific import DomainAnalysis

domain = DomainAnalysis(df)

# Environmental: Mann-Kendall trend test
result = domain.mann_kendall_test('temperature')

# Clinical: Bland-Altman method comparison
result = domain.bland_altman('method_a', 'method_b')

# Ecology: Shannon diversity
result = domain.shannon_diversity(['species_a', 'species_b'])
```

### Report Generation

Automated publication-ready reports:

```python
from data_toolkit.report_generator import ReportGenerator

report = ReportGenerator("My Analysis")
report.add_data_provenance(df, "data.csv")
report.add_statistics_table(results, "Main Results")
report.add_figure(fig, "Figure 1")
report.generate_html("report.html")
```

### Statistical Enhancements

New methods in `StatisticalAnalysis`:

```python
from data_toolkit.statistical_analysis import StatisticalAnalysis

sa = StatisticalAnalysis(df)

# Multiple testing correction
result = sa.multiple_testing_correction(p_values, method='fdr_bh')

# Variance Inflation Factor
result = sa.variance_inflation_factor(feature_cols)

# Robust statistics (resistant to outliers)
result = sa.robust_statistics('column')
```

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

---

## New in v4.3: Scientific Research Features

All features from v4.0 remain available. v4.3 adds:

### Community Ecology Module

Full community ecology analysis without external ecology packages:

```python
from data_toolkit.ecology import all_alpha_diversity, distance_matrix, rarefaction_curve

# Alpha diversity (12+ indices at once)
indices = all_alpha_diversity(abundance_array)
# → {'shannon': 2.31, 'simpson': 0.87, 'chao1': 45.2, ...}

# Beta diversity distance matrix
dm = distance_matrix(abundance_matrix, metric='bray_curtis')

# Rarefaction curve
rare = rarefaction_curve(abundance_array, n_steps=50)
```

### Ordination Module

Multivariate ordination—PCoA, NMDS, CA, DCA, CCA, RDA, Mantel test:

```python
from data_toolkit.ordination import pcoa, nmds, canonical_correspondence_analysis, mantel_test

# PCoA on a distance matrix
result = pcoa(distance_matrix, n_components=3)
# result['coordinates'], result['explained_variance'], result['eigenvalues']

# CCA with environmental constraints
result = canonical_correspondence_analysis(species_matrix, env_matrix)
# result['site_scores'], result['biplot_scores'], result['explained_inertia']

# Mantel test
result = mantel_test(dm1, dm2, method='spearman', permutations=999)
```

### Multivariate Hypothesis Testing

PERMANOVA, ANOSIM, SIMPER, MANOVA, Hotelling T², Discriminant Analysis:

```python
from data_toolkit.multivariate_analysis import permanova, anosim, simper

result = permanova(distance_matrix, group_labels, permutations=999)
# result['F_statistic'], result['p_value'], result['R2']

result = simper(abundance_matrix, group_labels, feature_names=species_names)
# Per-pair species contributions to dissimilarity
```

### Curve Fitting Module

Non-linear models—power, exponential, logistic, sinusoidal, Gompertz, RMA, GLM:

```python
from data_toolkit.curve_fitting import power_fit, logistic_fit, compare_fits

result = power_fit(x, y)
# result['parameters'], result['r_squared'], result['aic'], result['x_fit'], result['y_fit']

# Compare all models at once
comparison = compare_fits(x, y, models=['power', 'exponential', 'logistic', 'gompertz'])
```

---

*Last updated: April 2026*
