"""
Tutorial Sidebar module for the Data Analysis Toolkit
"""

import streamlit as st

# Import accelerated functions
try:
    from rust_accelerated import AccelerationSettings, is_rust_available
except ImportError:
    from .rust_accelerated import AccelerationSettings, is_rust_available


TUTORIALS = {
    "getting_started": """
## 🚀 Getting Started with the Advanced Data Analysis Toolkit

Welcome! This toolkit provides comprehensive data analysis capabilities organized into **9 main tabs** with **subtabs** inside each group.

---

### 📂 Tab Structure (9 Main Tabs → Subtabs)

**Click a main tab, then click a subtab inside to access specific features:**

| Main Tab | Subtabs Inside |
|----------|----------------|
| **📁 Data** | Data Loading (upload files, select columns) |
| **📊 Statistics** | Descriptive Statistics, Hypothesis Tests, Bayesian Inference, Uncertainty Analysis |
| **🔊 Signal Processing** | FFT/Wavelet (frequency and time-frequency analysis) |
| **⏱️ Time Series** | Time Series Analysis, Causality (Granger) |
| **🤖 Machine Learning** | Regression/Classification, 🧠 Neural Networks, PCA, Clustering, Anomaly Detection, Dimensionality Reduction, Non-Linear Analysis |
| **🔬 Scientific Tools** | Survival Analysis, Domain-Specific, Community Ecology, Ordination, Multivariate Tests, Curve Fitting |
| **📈 Visualization** | Interactive Plots (scatter, box, 3D, regression lines) |
| **📋 Reports** | Generate and export analysis reports |
| **🔌 Plugins** | Load and run custom analysis plugins |

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
| Confidence intervals? | 📊 Statistics → Uncertainty Analysis |
| Frequency content? | 🔊 Signal Processing → FFT/Wavelet |
| Time patterns? | ⏱️ Time Series → Analysis |
| Does X cause Y? | ⏱️ Time Series → Causality |
| Predict values? | 🤖 ML → Regression/Classification |
| Deep learning? | 🤖 ML → 🧠 Neural Networks |
| Reduce dimensions? | 🤖 ML → PCA or Dimensionality Reduction |
| Find clusters? | 🤖 ML → Clustering |
| Find outliers? | 🤖 ML → Anomaly Detection |
| Visualize data? | 📈 Visualization → Plots |
| Species diversity? | 🔬 Scientific Tools → Community Ecology |
| Ordination (PCoA, NMDS)? | 🔬 Scientific Tools → Ordination |
| Multivariate group tests? | 🔬 Scientific Tools → Multivariate Tests |
| Non-linear curve fitting? | 🔬 Scientific Tools → Curve Fitting |
| Custom analysis? | 🔌 Plugins → Load and run |

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

Anomaly detection identifies **unusual data points** that don't fit the normal pattern. This toolkit provides several robust methods, each with unique strengths and parameters. Below is a detailed guide to each method, their arguments, and what to expect from their results.

### Methods Available

| Method | Algorithm | Best For |
|--------|-----------|----------|
| **Isolation Forest** | Random tree isolation | General purpose, scalable, works well for most data |
| **LOF (Local Outlier Factor)** | Local density comparison | Detecting local anomalies, clusters, density-based outliers |
| **MCD (Minimum Covariance Determinant)** | Robust covariance | Elliptical distributions, robust to outliers |
| **One-Class SVM** | Support Vector Machine | High-dimensional data, flexible kernel choices |
| **DBSCAN** | Density-based clustering | Arbitrary shapes, noise, spatial/irregular clusters |
| **Autoencoder** | Neural network reconstruction | Complex, non-linear patterns, deep learning, high-dimensional data |

---

### Method Details & Arguments

#### **Isolation Forest**
- **How it works:** Randomly isolates points; anomalies are isolated faster.
- **Key argument:** `contamination` (float, 0-1): Expected fraction of anomalies (e.g., 0.05 = 5%).
- **What to expect:** Fast, general-purpose. Good for large datasets. Returns anomaly scores and binary labels.

#### **LOF (Local Outlier Factor)**
- **How it works:** Compares local density of a point to its neighbors.
- **Key arguments:**
    - `n_neighbors` (int): Number of neighbors to use (default: 20).
    - `contamination` (float): Expected fraction of anomalies.
- **What to expect:** Detects local anomalies, can find outliers in clusters. Sensitive to `n_neighbors`.

#### **MCD (Minimum Covariance Determinant)**
- **How it works:** Fits an ellipse to the data, robustly estimates covariance.
- **Key argument:** `contamination` (float): Expected fraction of anomalies.
- **What to expect:** Best for data with elliptical/normal structure. Robust to outliers, but less effective for non-elliptical data.

#### **One-Class SVM**
- **How it works:** Learns a boundary around the majority of data using support vectors.
- **Key arguments:**
    - `nu` (float, 0-1): Upper bound on the fraction of anomalies (default: 0.05).
    - `kernel` (str): Kernel type (`'rbf'`, `'linear'`, etc.; default: `'rbf'`).
    - `gamma` (str or float): Kernel coefficient (`'scale'` or `'auto'` or a float).
- **What to expect:** Flexible, works for high-dimensional data. Can be slow for large datasets. Sensitive to kernel and nu.

#### **DBSCAN**
- **How it works:** Groups points into dense clusters; points not in any cluster are labeled as anomalies.
- **Key arguments:**
    - `eps` (float): Maximum distance between neighbors (default: 0.5).
    - `min_samples` (int): Minimum points to form a cluster (default: 5).
- **What to expect:** Finds arbitrarily shaped clusters, labels noise as anomalies. No need to specify number of clusters. Good for spatial data.

#### **Autoencoder**
- **How it works:** Neural network learns to reconstruct input; anomalies have high reconstruction error.
- **Key arguments:**
    - `encoding_dim` (int): Size of bottleneck layer (default: 8).
    - `contamination` (float): Expected anomaly rate (default: 0.05).
    - `hidden_layers` (list): Encoder architecture (e.g., [64, 32]).
- **What to expect:** Powerful for complex, non-linear data. Needs more data (500+ samples recommended). Returns reconstruction errors, threshold, and anomaly indices.

---

### Output Interpretation
- **Normal points**: Label `1` (inliers)
- **Anomalies**: Label `-1` (outliers)
- **Anomaly scores**: Higher = more likely to be an anomaly (except for DBSCAN, where noise is labeled `-1`)
- **Scatter plot**: Anomalies shown in **red**

### Use Cases
- Fraud detection
- System monitoring
- Quality control
- Data cleaning
- Scientific discovery

---

### Choosing a Method
- **Start with Isolation Forest** for general use.
- **Try LOF** for local anomalies or clusters.
- **Use MCD** for elliptical/normal data.
- **One-Class SVM** for high-dimensional or kernel-based separation.
- **DBSCAN** for spatial/irregular clusters or when you expect noise.
- **Autoencoder** for complex, high-dimensional, or non-linear data (requires more data).

---

💡 **Tip**: Adjust the main parameter (contamination, nu, eps, etc.) to tune sensitivity. Always visualize results and check if detected anomalies make sense for your domain!
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

    "neural_networks": """
## 🧠 Neural Networks Guide

Deep learning models for regression, forecasting, and anomaly detection.

### Available Models

| Model | Use Case | Best For |
|-------|----------|----------|
| **MLP Regressor** | Predict continuous values | Non-linear regression |
| **MLP Classifier** | Classify categories | Multi-class classification |
| **LSTM Forecast** | Time series prediction | Sequential patterns |
| **Autoencoder** | Anomaly detection | Finding outliers |

---

### 🔮 MLP (Multi-Layer Perceptron)

Feedforward neural network with customizable architecture.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | [64, 32] | Neurons per layer |
| `activation` | relu | Activation function |
| `dropout_rate` | 0.2 | Regularization (0-0.5) |
| `epochs` | 100 | Training iterations |
| `batch_size` | 32 | Samples per update |

**Output Metrics (Regression):**
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination (closer to 1 is better)

**Tips:**
- Use 100+ epochs for complex patterns
- Reduce dropout for small datasets
- Add more layers for highly non-linear relationships

---

### 📈 LSTM (Long Short-Term Memory)

Specialized for time series with temporal dependencies.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 20 | Lookback window (past steps to use) |
| `forecast_horizon` | 10 | Future steps to predict |
| `lstm_units` | 64 | Neurons per LSTM layer |
| `n_lstm_layers` | 2 | Number of LSTM layers |

**When to Use:**
- Data has temporal patterns
- Sequential dependencies matter
- Need to forecast future values

**Tips:**
- sequence_length should capture one full cycle/pattern
- More data = better forecasts (100+ samples minimum)
- Use for univariate time series (single column)

---

### 🚨 Autoencoder (Anomaly Detection)

Learns to compress and reconstruct data. Anomalies have high reconstruction error.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoding_dim` | 8 | Bottleneck size (compression) |
| `contamination` | 0.05 | Expected anomaly rate (5%) |
| `hidden_layers` | [64, 32] | Encoder architecture |

**How It Works:**
1. **Encoder**: Compresses data to low dimensions
2. **Decoder**: Reconstructs original data
3. **Error**: Normal data reconstructs well, anomalies don't
4. **Threshold**: Set by contamination rate

**Output:**
- `reconstruction_errors`: Error per sample
- `threshold`: Cutoff for anomaly classification
- `anomaly_indices`: Which samples are anomalies

---

### ⚠️ Important Notes

**Data Requirements:**
- Neural networks need MORE data than traditional ML
- Minimum: 500+ samples (1000+ recommended)
- More features = need more data

**When NOT to Use:**
- Small datasets (< 200 samples) → Use Random Forest instead
- Simple linear relationships → Use Linear Regression
- Need interpretability → Use Decision Trees

**Training Tips:**
- Watch for overfitting (val_loss increasing while loss decreases)
- Start with fewer epochs, increase if needed
- Use validation split to monitor generalization

---

### 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Predictions always same value | Increase epochs, reduce dropout |
| Overfitting (val_loss >> loss) | Increase dropout, reduce layers |
| Training too slow | Reduce batch_size, fewer layers |
| Poor accuracy | More data, more epochs, tune architecture |

💡 **Tip**: Start simple (2 layers, 64 neurons) and add complexity only if needed!
""",

    "image_recognition": """
## 🖼️ Image Recognition (Streamlit Image Tab)

Step-by-step:
1) **Select data folder** — provide class subfolders or upload a `labels.csv` mapping filenames to labels (prefer relative paths).
2) **Preview & validate** — inspect a few images per class; check for missing or corrupt files.
3) **Configure** — set `image_size` (64–224), `batch_size` (16–64), `epochs`, and enable augmentation if needed.
4) **Choose model** — CNN from scratch or Transfer Learning (recommended for small datasets).
5) **Train** — use early stopping on validation loss; monitor accuracy and per-class metrics.
6) **Save** — enable "Save model automatically" or use the Save trained model button; prefer `.keras` format. If class names are missing, predictions will show numeric labels.

Troubleshooting: Verify `labels.csv` paths, re-save in `.keras` to avoid HDF5 issues, and register custom layers used by the model.
""",

    "anova_extended": """
## 🧪 Extended ANOVA & Post-Hoc Tests

### Two-Way ANOVA
Tests main effects of two factors and their interaction on a continuous outcome.
- **Main Effect A**: Does factor A affect the outcome?
- **Main Effect B**: Does factor B affect the outcome?
- **Interaction A×B**: Does the effect of A depend on B?

### Repeated-Measures ANOVA
Use when the same subjects are measured under multiple conditions.
- Accounts for within-subject correlation
- More powerful than independent-groups ANOVA
- Check sphericity assumption (Mauchly's test)

### Post-Hoc Tests
After significant ANOVA, determine which groups differ:

| Test | Use When |
|------|----------|
| **Tukey's HSD** | All pairwise comparisons, balanced groups |
| **Bonferroni** | Conservative, few comparisons |

💡 **Tip**: Run post-hoc tests only after significant omnibus ANOVA!
""",

    "probability_distributions": """
## 📊 Probability Distribution Fitting

### Extended Distribution Fitting
Fits 12+ theoretical distributions to your data:
- Normal, t-distribution, Gamma, Exponential
- Lognormal, Weibull, Laplace, Logistic
- Pareto, Cauchy, and more

### Model Selection Criteria
| Metric | Interpretation |
|--------|----------------|
| **AIC** | Lower is better (penalizes complexity) |
| **BIC** | Lower is better (stricter penalty) |
| **KS test** | p > 0.05 means good fit |

### Random Variable Analysis
- Moments: mean, variance, skewness, kurtosis
- Quantiles: median, quartiles, percentiles
- Confidence intervals for parameters

### QQ Plots
Points on diagonal = good fit to theoretical distribution.

💡 **Tip**: Compare multiple distributions using AIC before choosing one!
""",

    "spectral_extended": """
## 🔊 Extended Spectral Analysis

### Coherence Analysis
Measures linear correlation between two signals as a function of frequency.
- **Value 0-1**: Higher = stronger relationship at that frequency
- Use for finding common oscillations in paired signals

### Cross-Wavelet Transform (XWT)
Shows where two signals share common power in time-frequency space.
- Reveals coupled oscillations
- Identifies phase relationships

### Wavelet Coherence (WTC)
Localized coherence varying in both time and frequency.
- Better than standard coherence for non-stationary signals
- Shows when/where signals are correlated

### Harmonic Analysis
Least-squares fitting of sinusoidal components.
- Extracts dominant periodicities
- Provides amplitude, frequency, and phase

💡 **Tip**: Use coherence for stationary signals, wavelet coherence for non-stationary!
""",

    "arima_sarima": """
## ⏱️ ARIMA/SARIMA Forecasting

### ARIMA(p, d, q)
- **p**: Autoregressive order (use PACF cutoff)
- **d**: Differencing order (0 if stationary)
- **q**: Moving average order (use ACF cutoff)

### SARIMA(p,d,q)(P,D,Q,m)
Adds seasonal components:
- **P, D, Q**: Seasonal AR, differencing, MA orders
- **m**: Seasonal period (12=monthly, 4=quarterly)

### Auto-ARIMA
Automatic parameter selection via grid search:
- Tests multiple (p,d,q) combinations
- Selects model with lowest AIC/BIC

### Forecasting
- Point forecasts with confidence intervals
- Residual diagnostics (should be white noise)
- Use holdout set for validation

💡 **Tip**: Always check stationarity (ADF test) before ARIMA modeling!
""",

    "multivariate_ts": """
## 📈 Multivariate Time Series

### VAR (Vector Autoregression)
Models multiple time series jointly.
- Each variable depends on lagged values of all variables
- Includes Granger causality tests
- Good for forecasting interrelated variables

### VECM (Vector Error Correction)
Use when series are cointegrated (share long-run equilibrium).
- Johansen test identifies cointegration rank
- Models both short-run dynamics and long-run relationships
- Better than VAR for non-stationary but cointegrated series

### DTW (Dynamic Time Warping)
Measures similarity between time series that may be shifted/stretched.
- Handles different lengths and speeds
- Finds optimal alignment between series
- Use for pattern matching and clustering

### Granger Causality
Tests if past values of X help predict Y.
- Not true causation, but predictive causality
- Check at multiple lags

💡 **Tip**: Use VECM if Johansen test shows cointegration, VAR otherwise!
""",

    "biomass_segmentation": """
## 🌿 Biomass Segmentation (U-Net)

### Overview
Semantic segmentation for aquatic biomass using deep learning.
Classes: Background, Dreissena mussels, Cladophora algae, Other vegetation

### U-Net Architecture
- Encoder: Extracts features (downsampling)
- Decoder: Reconstructs spatial resolution (upsampling)
- Skip connections: Preserve fine details

### Training Tips
- Use augmentation for small datasets
- Dice loss works better than cross-entropy for imbalanced masks
- Start with pretrained VGG encoder for transfer learning

### Output Analysis
- Per-class coverage percentages
- Spatial distribution maps
- Biomass density estimates

💡 **Tip**: Transfer learning with VGG backbone works best for small datasets!
""",

    "ecology": """
## 🌿 Community Ecology Analysis Guide

**Location:** Main Tab: 🔬 Scientific Tools › Subtab: Community Ecology

### Overview

Analyse species × sites data with standard ecological diversity metrics,
rarefaction, accumulation curves, and SHE analysis.

---

### Alpha Diversity (Within-Sample)

Indices computed for each sample (row):

| Index | Formula Concept | Sensitive To |
|-------|-----------------|--------------|
| **Shannon H'** | −Σ pᵢ ln(pᵢ) | Rare species |
| **Simpson 1-D** | 1 − Σ pᵢ² | Dominant species |
| **Inv. Simpson** | 1 / Σ pᵢ² | Dominance |
| **Fisher's α** | S = α ln(1 + N/α) | Sample-size robust |
| **Margalef** | (S−1) / ln(N) | Richness scaled by N |
| **Chao1** | S + f₁²/(2f₂) | Undetected species |
| **ACE** | Similar to Chao1 | Rare-species-based |
| **Pielou J'** | H' / ln(S) | Evenness |

**Typical workflow:**
1. Select species columns and (optionally) a sample-label column
2. Click **Calculate Alpha Diversity**
3. Compare indices across sites in the results table

---

### Beta Diversity (Between-Sample)

| Metric | Data Type | Range |
|--------|-----------|-------|
| **Bray-Curtis** | Abundance | 0–1 |
| **Jaccard** | Presence/Absence | 0–1 |
| **Sørensen** | Presence/Absence | 0–1 |
| **Morisita-Horn** | Abundance | 0–1 |
| **Whittaker βW** | Presence/Absence | 0–∞ |

0 = identical communities, 1 = completely different.

---

### Rarefaction Curves

Rarefaction standardises richness to a common sampling effort, answering:
*"If I had only sampled N individuals, how many species would I expect?"*

- Curves that plateau indicate adequate sampling.
- Curves still rising suggest more species remain undetected.

---

### Species Accumulation

Shows cumulative species richness as more samples are added.
The curve's approach to an asymptote indicates sampling completeness.

---

### SHE Analysis

Plots **S** (richness), **H'** (Shannon), and **E** (evenness) against
cumulative N to diagnose the community abundance model:
- **Log-normal community**: H' and E increase together
- **Broken-stick**: H' increases linearly with ln(N)

💡 **Tip**: Start with alpha diversity to characterise each site, then use beta
diversity + ordination to compare sites!
""",

    "ordination": """
## 🧭 Ordination Methods Guide

**Location:** Main Tab: 🔬 Scientific Tools › Subtab: Ordination

### Overview

Ordination projects multivariate data into a low-dimensional space
for visualisation and pattern discovery.

---

### Unconstrained Methods

These methods find the axes of maximum variation in your data:

**PCoA (Principal Coordinates Analysis)**
- Input: distance matrix (Bray-Curtis, Jaccard, etc.)
- Output: Euclidean embedding preserving distances
- Eigenvalue-based → deterministic, fast
- Use when: any distance metric is appropriate

**NMDS (Non-Metric Multidimensional Scaling)**
- Input: distance matrix
- Output: configuration preserving rank-order of distances
- Iterative → check stress value
- Stress interpretation: < 0.05 excellent, < 0.1 good, < 0.2 acceptable

**CA (Correspondence Analysis)**
- Input: species × sites abundance matrix
- Assumes unimodal species response to environment
- Produces joint species + site ordination
- Eigenvalues = "inertia" (total chi-squared / N)

**DCA (Detrended Correspondence Analysis)**
- Removes the **arch effect** from CA
- Axis units in **SD units** (standard deviations of turnover)
- If DCA Axis 1 length > 4 SD → unimodal methods preferred (CA/CCA)
- If DCA Axis 1 length < 3 SD → linear methods OK (PCA/RDA)

---

### Constrained Methods

These test how environmental variables explain community composition:

**CCA (Canonical Correspondence Analysis)**
- Species ~ environment, assuming unimodal response
- Biplot: species optima + environmental arrows
- Axis eigenvalues = constrained inertia

**RDA (Redundancy Analysis)**
- Species ~ environment, assuming linear response
- Biplot: species loadings + environmental arrows
- Use when DCA axis 1 < 3 SD

---

### Mantel Test

Tests correlation between two distance matrices (permutation-based).
- Mantel r: −1 to 1 (correlation strength)
- p-value: permutation significance
- Use for: spatial autocorrelation, distance-decay of similarity

---

### Reading an Ordination Plot

- **Points close together** → similar composition
- **Points far apart** → different composition
- **Arrows** (CCA/RDA) → environmental gradients (longer = stronger)
- **Species points** (CA/CCA) → species' optimal positions along gradients
- **Axis labels** show % variance explained

💡 **Tip**: Always report the stress (NMDS) or % variance explained (PCoA/CA)
alongside your ordination plots!
""",

    "multivariate_tests": """
## 📊 Multivariate Hypothesis Tests Guide

**Location:** Main Tab: 🔬 Scientific Tools › Subtab: Multivariate Tests

### Overview

Tests whether groups differ in multivariate response variables.
These are the multivariate extensions of ANOVA and t-tests.

---

### Non-Parametric Tests

**PERMANOVA (Permutational MANOVA)**
- Tests multivariate location differences between groups
- No distributional assumptions — uses permutations
- Works on any distance matrix
- Sensitive to both location AND dispersion differences
- Report: pseudo-F, p-value, R² (effect size)

**ANOSIM (Analysis of Similarities)**
- Rank-based test of group differences
- R statistic: −1 to 1
  - R ≈ 0: no difference between groups
  - R ≈ 1: groups completely separated
  - R < 0: within-group dissimilarity > between-group
- Less powerful than PERMANOVA but more robust to dispersion differences

**SIMPER (Similarity Percentages)**
- Not a hypothesis test — diagnostic breakdown
- Shows which species contribute most to between-group dissimilarity
- Reports: contribution %, cumulative %, mean abundance per group
- Use after PERMANOVA/ANOSIM to explain *what* differs

---

### Parametric Tests

**MANOVA (Multivariate ANOVA)**
- Requires multivariate normality and homogeneous covariance
- Test statistics: Wilks' Λ, Pillai's trace, Hotelling-Lawley, Roy's root
- More powerful than PERMANOVA when assumptions are met
- Use with 3+ groups

**Hotelling T² (Two-Sample)**
- Multivariate extension of the two-sample t-test
- Tests whether two group centroids differ
- Requires multivariate normality
- Reports: T² statistic, F-approximation, p-value

---

### Discriminant Analysis (LDA)

- Finds linear combinations of variables that best separate groups
- Canonical variates: directions of maximum between-group separation
- Cross-validated classification accuracy
- Useful for both testing and predicting group membership

---

### Workflow

```
1. Check groups have ≥ 5 observations each
2. Test normality? → Yes: MANOVA / Hotelling T²
                   → No:  PERMANOVA / ANOSIM
3. Significant? → Run SIMPER to find which variables differ
4. Visualise  → LDA biplot or ordination coloured by group
```

💡 **Tip**: PERMANOVA is the most widely used test in ecology because it makes
no distributional assumptions — start there if unsure!
""",

    "curve_fitting": """
## 📈 Curve Fitting & Non-Linear Models Guide

**Location:** Main Tab: 🔬 Scientific Tools › Subtab: Curve Fitting

### Overview

Fit non-linear and specialised regression models to your data,
with single-model fitting and multi-model comparison.

---

### Available Models

| Model | Equation | When to Use |
|-------|----------|-------------|
| **Power** | y = a·xᵇ | Allometric scaling, species-area |
| **Exponential (2p)** | y = a·e^(bx) | Growth/decay without bound |
| **Exponential (3p)** | y = a·e^(bx) + c | Growth/decay with baseline |
| **Logistic (4p)** | y = d + (a−d)/(1+(x/c)^b) | Dose-response, S-curves |
| **Sinusoidal** | y = A·sin(2πfx + φ) + offset | Seasonal, periodic data |
| **Gompertz** | y = a·e^(−b·e^(−cx)) | Asymmetric growth, tumour growth |
| **RMA Regression** | y = a + bx (Type II) | Both variables measured with error |
| **GLM** | g(E[y]) = Xβ | Generalised linear model |

---

### Single Model Fit

1. Select X (predictor) and Y (response) columns
2. Choose a model from the dropdown
3. Click **Fit Model**
4. Review the fitted parameters, R², RMSE, and AIC
5. Inspect the plot: data points + fitted curve + prediction band

---

### Multi-Model Comparison

1. Select X and Y columns
2. Choose which models to compare (checkboxes)
3. Click **Compare All**
4. A ranking table shows each model's AIC, BIC, R², and RMSE
5. ΔAIC is computed relative to the best model:
   - ΔAIC < 2: substantial support (competing models)
   - ΔAIC 2–10: less support
   - ΔAIC > 10: essentially no support

---

### RMA Regression (Type II)

Standard OLS minimises vertical residuals (assumes X is error-free).
RMA (Reduced Major Axis) minimises in both X and Y directions:
- Use when **both** variables are measured with error
- Common in allometry (e.g. body length vs. body mass)
- Reports slope, intercept, R², and confidence intervals

---

### GLM (Generalised Linear Model)

Extends regression to non-normal responses via link functions:

| Family | Link | Response Type |
|--------|------|---------------|
| Gaussian | Identity | Continuous |
| Poisson | Log | Count data |
| Binomial | Logit | Binary / proportion |
| Gamma | Log | Positive continuous |
| Inverse Gaussian | Inverse | Positive continuous |

---

### Interpreting Residual Plots

After fitting any model, check the residual plot:
- **Random scatter** → good fit
- **Pattern / curve** → model is misspecified, try a different form
- **Funnel shape** → heteroscedasticity (consider weighted regression or GLM)

💡 **Tip**: Always try Multi-Model Comparison first — let the data decide
which functional form is most appropriate!
""",

    "plugins": """
## 🔌 Plugin System Guide

**Location:** Main Tab: 🔌 Plugins

### Overview

The plugin system lets you add custom analyses, preprocessing steps,
or visualisations without modifying the core toolkit.

---

### Loading a Plugin

Three methods:
1. **From file**: Upload a `.py` file
2. **Paste code**: Write/paste Python code directly
3. **Example templates**: Load bundled example plugins

---

### Creating a Plugin

Every plugin needs three components:

**1. PLUGIN_INFO** — metadata dict:
```python
PLUGIN_INFO = {
    "name": "My Analysis",
    "description": "What it does",
    "category": "analysis",   # analysis | preprocessing | visualization
    "version": "1.0",
    "author": "Your Name"
}
```

**2. PLUGIN_PARAMETERS** — auto-generated UI controls:
```python
PLUGIN_PARAMETERS = {
    "alpha": {"type": "float", "default": 0.05,
              "min": 0.0, "max": 1.0,
              "description": "Significance level"},
    "method": {"type": "select",
               "options": ["anova", "kruskal"],
               "default": "anova",
               "description": "Test method"}
}
```

Supported types: `float`, `int`, `bool`, `str`, `select`

**3. process()** — main function:
```python
def process(data, columns, target=None, **params):
    # data: pandas DataFrame
    # columns: list of feature columns
    # target: optional target column
    # params: configured parameter values
    result_df = ...
    return {"summary": "Analysis complete", "data": result_df}
```

---

### Bundled Examples

| Plugin | Category | Description |
|--------|----------|-------------|
| **Enhanced Scatter** | Visualization | Scatter + regression + marginals |
| **Lag Features** | Preprocessing | Create lagged time series columns |
| **Outlier Removal** | Preprocessing | IQR / Z-score outlier filtering |

---

### Tips

- Return a `"data"` key (DataFrame) to enable CSV download of results
- Plugins run in an isolated namespace — no risk to the core toolkit
- Save frequently used plugins to files for quick reloading
- Check `example_plugins/README.md` for the full API reference

💡 **Tip**: Start from an example template and modify it to save time!
""",
}


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
            
            # Toggle between short and comprehensive tutorials
            tutorial_mode = st.radio(
                "Documentation Mode",
                ["📝 Quick Reference", "📖 Comprehensive Guide"],
                key="tutorial_mode",
                horizontal=True
            )
            use_comprehensive = tutorial_mode == "📖 Comprehensive Guide"

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
                "anova_extended": "📊 Statistics › ANOVA (Extended)",
                "probability_distributions": "📊 Statistics › Probability Distributions",
                "bayesian": "📊 Statistics › Bayesian Inference",
                "uncertainty": "📊 Statistics › Uncertainty Analysis",
                # Signal Processing group
                "signal_analysis": "🔊 Signal Processing › FFT/Wavelet",
                "spectral_extended": "🔊 Signal Processing › Extended Spectral",
                # Time Series group
                "timeseries": "⏱️ Time Series › Analysis",
                "arima_sarima": "⏱️ Time Series › ARIMA/SARIMA",
                "multivariate_ts": "⏱️ Time Series › Multivariate (VAR/VECM/DTW)",
                "causality": "⏱️ Time Series › Causality (Granger)",
                # Machine Learning group
                "machine_learning": "🤖 ML › Regression/Classification",
                "neural_networks": "🧠 ML › Neural Networks",
                "image_recognition": "🖼️ Image Recognition › Streamlit Tab",
                "biomass_segmentation": "🌿 ML › Biomass Segmentation",
                "pca": "🤖 ML › PCA (Principal Components)",
                "clustering": "🤖 ML › Clustering",
                "anomaly": "🤖 ML › Anomaly Detection",
                "dim_reduction": "🤖 ML › Dimensionality Reduction",
                "nonlinear": "🤖 ML › Non-Linear Analysis",
                # Scientific Tools group
                "ecology": "🔬 Scientific Tools › Community Ecology",
                "ordination": "🔬 Scientific Tools › Ordination",
                "multivariate_tests": "🔬 Scientific Tools › Multivariate Tests",
                "curve_fitting": "🔬 Scientific Tools › Curve Fitting",
                # Visualization group
                "visualization": "📈 Visualization › Plots",
                # Plugins
                "plugins": "🔌 Plugins › Plugin System",
            }

            # Use Streamlit's `key` to bind the selectbox directly to
            # `st.session_state.current_tutorial`. Add an `on_change` callback
            # that records the most recent selection to force an immediate
            # session-state update (helps with certain frontend update races).
            def _load_comprehensive_module():
                """Load `comprehensive_tutorial` without triggering heavy package imports.
                First try normal import; if that fails, load the module directly from file.
                """
                try:
                    from data_toolkit import comprehensive_tutorial as ct
                    return ct
                except Exception:
                    # Fallback: load module directly from file to avoid package-level imports
                    import importlib.util
                    from pathlib import Path
                    file_path = Path(__file__).resolve().parents[1] / "comprehensive_tutorial.py"
                    if not file_path.exists():
                        raise
                    spec = importlib.util.spec_from_file_location("comprehensive_tutorial_local", str(file_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod

            selected = st.selectbox(
                "Documentation Topic",
                options=list(tutorial_topics.keys()),
                format_func=lambda x: tutorial_topics[x],
                key="current_tutorial",
            )

            # Visible status line so users can see the currently selected topic
            # immediately after clicking (useful for diagnosing double-click issues).
            st.markdown(f"**Selected:** {tutorial_topics.get(st.session_state.current_tutorial, st.session_state.current_tutorial)}")

            # Show a compact source indicator immediately under the selectbox so
            # users do not have to scroll to the content area to see where the
            # text is coming from (comprehensive file vs short sidebar snippet).
            try:
                comp = _load_comprehensive_module()
                comp_topics = comp.get_all_topics()
                cur = st.session_state.current_tutorial
                
                # Keys are now identical between sidebar and comprehensive tutorials
                comp_topic = cur if cur in comp_topics else None

                if use_comprehensive and comp_topic:
                    source_text = "📖 Comprehensive Guide"
                elif comp_topic:
                    source_text = "📝 Quick Reference (📖 available)"
                else:
                    source_text = "📝 Quick Reference"

            except Exception:
                source_text = "📝 Quick Reference (comprehensive load failed)"

            st.markdown(f"**{source_text}**")

            # Render the tutorial content based on mode selection
            try:
                if use_comprehensive and 'comp' in locals() and comp is not None and 'comp_topic' in locals() and comp_topic:
                    st.markdown(comp.get_tutorial(comp_topic))
                else:
                    st.markdown(TUTORIALS.get(selected, "Topic not found in quick reference."))
            except Exception as e:
                # Show the short tutorial and surface a debug note
                st.markdown(TUTORIALS.get(selected, "Topic not found."))
                st.error(f"[Debug] Comprehensive tutorial load failed: {e}")

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


