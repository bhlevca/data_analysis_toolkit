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

Welcome! This toolkit provides comprehensive data analysis capabilities organized into **6 main tabs** with **subtabs** inside each group.

---

### 📂 Tab Structure (6 Main Tabs → Subtabs)

**Click a main tab, then click a subtab inside to access specific features:**

| Main Tab | Subtabs Inside |
|----------|----------------|
| **📁 Data** | Data Loading (upload files, select columns) |
| **📊 Statistics** | Descriptive Statistics, Hypothesis Tests, Bayesian Inference, Uncertainty Analysis |
| **🔊 Signal Processing** | FFT/Wavelet (frequency and time-frequency analysis) |
| **⏱️ Time Series** | Time Series Analysis, Causality (Granger) |
| **🤖 Machine Learning** | Regression/Classification, 🧠 Neural Networks, PCA, Clustering, Anomaly Detection, Dimensionality Reduction, Non-Linear Analysis |
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
"""
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
                # Signal Processing group
                "signal_analysis": "🔊 Signal Processing › FFT/Wavelet",
                # Time Series group
                "timeseries": "⏱️ Time Series › Analysis",
                "causality": "⏱️ Time Series › Causality (Granger)",
                # Machine Learning group
                "machine_learning": "🤖 ML › Regression/Classification",
                "neural_networks": "🧠 ML › Neural Networks",
                "pca": "🤖 ML › PCA (Principal Components)",
                "clustering": "🤖 ML › Clustering",
                "anomaly": "🤖 ML › Anomaly Detection",
                "dim_reduction": "🤖 ML › Dimensionality Reduction",
                "nonlinear": "🤖 ML › Non-Linear Analysis",
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


