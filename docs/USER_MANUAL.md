# 📘 Advanced Data Analysis Toolkit - User Manual

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Tab: Data Loading](#tab-data-loading)
4. [Tab: Statistical Analysis](#tab-statistical-analysis)
5. [Tab: Machine Learning](#tab-machine-learning)
6. [Tab: Bayesian Analysis](#tab-bayesian-analysis)
7. [Tab: Uncertainty Analysis](#tab-uncertainty-analysis)
8. [Tab: Non-Linear Analysis](#tab-non-linear-analysis)
9. [Tab: Time Series](#tab-time-series)
10. [Tab: Causality Analysis](#tab-causality-analysis)
11. [Tab: Visualizations](#tab-visualizations)
12. [Tab: Plugins](#tab-plugins)
13. [Rust Acceleration Toggle](#rust-acceleration-toggle)
14. [Tips and Best Practices](#tips-and-best-practices)
15. [Troubleshooting](#troubleshooting)

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

- **v8.0**: Unified edition with Rust toggle, plugin system, white theme
- **v7.0**: Modular architecture, dark theme
- **v6.0**: Added causality analysis
- **v5.0**: Added time series module
- **v4.0**: Added Bayesian analysis
- **v3.0**: Added uncertainty quantification
- **v2.0**: Added ML models
- **v1.0**: Initial release with statistical analysis

---

*Last updated: December 2024*
