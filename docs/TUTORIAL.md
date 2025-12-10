# 📚 Step-by-Step Tutorial: Advanced Data Analysis Toolkit v9

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Loading Your Data](#loading-your-data)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Statistical Analysis](#statistical-analysis)
6. [Machine Learning](#machine-learning)
7. [Uncertainty Quantification](#uncertainty-quantification)
8. [Non-Linear Analysis](#non-linear-analysis)
9. [Time Series Analysis](#time-series-analysis)
10. [Causality Analysis](#causality-analysis)
11. [Decision Flowcharts](#decision-flowcharts)
12. [Common Workflows](#common-workflows)

---

## Introduction

The Advanced Data Analysis Toolkit is a comprehensive application for exploring, analyzing, and modeling your data. This tutorial will guide you through each feature step-by-step.

### What Can This Toolkit Do?

| Task | Capability |
|------|------------|
| 📊 Explore Data | Summary statistics, distributions, correlations |
| 🎯 Detect Issues | Outliers, missing values, multicollinearity |
| 🤖 Build Models | Linear regression, Random Forest, Gradient Boosting |
| 📈 Quantify Uncertainty | Bootstrap CI, Bayesian inference, Monte Carlo |
| 🔀 Find Non-linear Patterns | Distance correlation, mutual information, GP |
| ⏱️ Analyze Time Series | ACF/PACF, stationarity, ARIMA |
| 🔗 Test Causality | Granger causality, lead-lag analysis |

---

## Getting Started

### Launching the App

```bash
# Navigate to the toolkit directory
cd advanced_data_toolkit

# Install dependencies
pip install -e .
pip install streamlit

# Launch the Streamlit app
python run_streamlit.py

# Or directly:
streamlit run src/data_toolkit/streamlit_app.py
```

### Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Advanced Data Analysis Toolkit              [Sidebar: Tutorial]
├─────────────────────────────────────────────────────────────────┤
│  📁 Data | 📊 Statistical | 🤖 ML | 📈 Bayesian | 🎲 Uncertainty │
│          | 🔀 Non-Linear | ⏱️ Time Series | 🔗 Causality | 📉 Viz │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        [Main Content Area]                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Sidebar Features:**
- 📚 Tutorial Guide (toggle on/off)
- Select tutorial topic
- ⚡ Rust acceleration toggle

---

## Loading Your Data

### Step 1: Upload Your File

1. Click the **📁 Data Loading** tab
2. Click **"Browse files"** 
3. Select your CSV or Excel file
4. Wait for the upload confirmation ✅

**Supported formats:** CSV, XLSX, XLS

### Step 2: Review Data Info

After loading, you'll see:
- **Rows**: Number of observations
- **Columns**: Number of variables
- **Memory**: Dataset size
- **Missing**: Count of missing values

### Step 3: Select Columns

| Selection | Purpose |
|-----------|---------|
| **Feature Columns** | Variables to analyze or use as predictors |
| **Target Column** | Variable to predict or explain |

**Tips:**
- Select multiple features using the multiselect dropdown
- Target is optional for exploratory analysis
- Start with a few columns, add more later

### Step 4: Preview Your Data

The data preview shows the first 10 rows. Use this to verify:
- ✅ Data loaded correctly
- ✅ Column names are sensible
- ✅ No obvious formatting issues

---

## Exploratory Data Analysis

**Goal:** Understand your data before modeling

### Recommended First Steps

```
1. Descriptive Statistics → See distributions
2. Correlation Matrix    → Find relationships
3. Box Plots            → Spot outliers
4. Outlier Detection    → Identify anomalies
```

### Understanding Descriptive Statistics

| Statistic | What It Tells You |
|-----------|-------------------|
| **Mean** | Average value |
| **Std** | Spread of values |
| **Min/Max** | Range of values |
| **25%/50%/75%** | Distribution shape |
| **Skewness** | Asymmetry (0 = symmetric) |
| **Kurtosis** | Tail heaviness (3 = normal) |

### Interpreting Correlations

```
Strong Positive:   r > 0.7    → Variables increase together
Moderate Positive: 0.3 < r < 0.7
Weak:             -0.3 < r < 0.3 → Little linear relationship
Moderate Negative: -0.7 < r < -0.3
Strong Negative:   r < -0.7   → One increases, other decreases
```

### When to Worry About Outliers

Consider removing outliers if:
- They're data entry errors
- They're from a different population
- They significantly affect your model

Keep outliers if:
- They're genuine extreme observations
- They contain important information
- Your analysis method is robust to outliers

---

## Statistical Analysis

### Analysis Selection Guide

**"I want to understand my data"**
→ Use: Descriptive Statistics

**"I want to see relationships"**
→ Use: Correlation Matrix
- Pearson: For linear relationships
- Spearman: For monotonic relationships (ranked data)
- Kendall: For ordinal data

**"I want to find unusual values"**
→ Use: Outlier Detection
- IQR method: Robust, no distribution assumptions
- Z-score: Assumes normal distribution

### Step-by-Step: Correlation Analysis

1. Go to **📊 Statistical** tab
2. Select correlation method (start with Pearson)
3. Click **🔗 Correlation Matrix**
4. Interpret the heatmap:
   - 🔵 Blue = positive correlation
   - ⚪ White = no correlation
   - 🔴 Red = negative correlation

### Step-by-Step: Outlier Detection

1. Go to **📊 Statistical** tab
2. Select method: IQR (recommended) or Z-score
3. Click **🎯 Outlier Detection**
4. Review results for each column:
   - Count of outliers
   - Percentage of data affected
   - Bounds used for detection

---

## Machine Learning

### Choosing the Right Model

```
Start Here
    │
    ▼
Is relationship linear? ──Yes──► Linear Regression
    │
    No
    ▼
Do you have many features? ──Yes──► Lasso (auto feature selection)
    │                               or Ridge (handles collinearity)
    No
    ▼
Do you need interpretability? ──Yes──► Decision Tree
    │
    No
    ▼
Do you need best accuracy? ──Yes──► Gradient Boosting
    │
    No
    ▼
Random Forest (good default for non-linear)
```

### Step-by-Step: Training a Model

1. Go to **🤖 ML** tab
2. Select model type from dropdown
3. Adjust parameters if needed:
   - **Alpha** for Ridge/Lasso (higher = more regularization)
   - **n_estimators** for tree models (more = better but slower)
4. Click **🎯 Train Model**
5. Review metrics:
   - **R²**: Higher is better (1.0 = perfect)
   - **RMSE**: Lower is better (in original units)
   - **MAE**: Lower is better (typical error)

### Step-by-Step: Cross-Validation

**Why do this?** To get reliable performance estimates

1. Set number of folds (5 is standard)
2. Click **🔄 Cross-Validation**
3. Look at:
   - Mean score: Expected performance
   - Std: Stability (lower = more stable)

### Understanding Feature Importance

1. Train a model first (Random Forest works best)
2. Click **📊 Feature Importance**
3. Interpret the bar chart:
   - Longer bars = more important features
   - Consider removing low-importance features

---

## Uncertainty Quantification

### When to Use Each Method

| Method | Use When |
|--------|----------|
| Bootstrap CI | You want robust confidence intervals |
| Monte Carlo | You need prediction uncertainty |
| Residual Analysis | You want to validate model assumptions |

### Step-by-Step: Bootstrap Confidence Intervals

1. Go to **🎲 Uncertainty** tab
2. Set parameters:
   - **Bootstrap samples**: 1000 is good (more = more precise)
   - **Confidence level**: 0.95 is standard
3. Click **🔄 Bootstrap CI**
4. Interpret results:
   - **Mean**: Best estimate of coefficient
   - **Std Error**: Uncertainty in estimate
   - **CI bounds**: True value likely within this range

### Step-by-Step: Residual Analysis

**Always do this after fitting a model!**

1. Click **🎯 Residual Analysis**
2. Check the diagnostics:

| Diagnostic | Good Value | Problem If |
|------------|------------|------------|
| Durbin-Watson | ≈ 2.0 | < 1.5 or > 2.5 (autocorrelation) |
| Normality p-value | > 0.05 | < 0.05 (non-normal residuals) |
| Q-Q Plot | Points on line | Points curve away |

3. If problems exist:
   - Transform variables
   - Add missing predictors
   - Try different model

---

## Non-Linear Analysis

### Key Concept: Pearson vs Distance Correlation

```
Pearson correlation only detects LINEAR relationships!

Example:
  y = x²  →  Pearson r ≈ 0  (misses the relationship!)
            Distance corr ≈ 0.7 (detects it!)
```

### Step-by-Step: Detecting Non-Linear Relationships

1. Go to **🔀 Non-Linear** tab
2. Click **📊 Distance Correlation**
3. Compare with Pearson:
   - Similar values → Linear relationship
   - Distance >> Pearson → Non-linear relationship!

### When Distance Correlation Helps

| Relationship | Pearson | Distance | Conclusion |
|--------------|---------|----------|------------|
| y = 2x + 1 | 0.99 | 0.99 | Linear ✅ |
| y = x² | 0.02 | 0.68 | Non-linear! Use polynomial |
| y = sin(x) | 0.05 | 0.71 | Non-linear! |
| Independent | 0.01 | 0.03 | No relationship |

### Step-by-Step: Polynomial Regression

1. First, plot your data to guess the degree needed
2. Select feature and polynomial degree
3. Click **📈 Polynomial Regression**
4. Start with degree 2, increase if R² is low
5. **Warning**: degree > 4 often overfits!

---

## Time Series Analysis

### The Time Series Workflow

```
Step 1: Plot the series
    │
    ▼
Step 2: Check stationarity (ADF test)
    │
    ├── Stationary (p < 0.05) ──► Continue to Step 3
    │
    └── Non-stationary ──► Difference the data, repeat Step 2
    │
    ▼
Step 3: Examine ACF/PACF
    │
    ▼
Step 4: Identify model (AR, MA, ARMA)
    │
    ▼
Step 5: Fit and validate
```

### Step-by-Step: Stationarity Test

1. Go to **⏱️ Time Series** tab
2. Select your time series column
3. Click **🔬 Stationarity Test**
4. Interpret:
   - **p < 0.05**: Series IS stationary ✅
   - **p ≥ 0.05**: Series is NOT stationary ⚠️

### Step-by-Step: Reading ACF/PACF

**ACF (Autocorrelation Function)**
- Shows correlation with lagged values
- Use to identify MA order

**PACF (Partial ACF)**
- Shows direct correlation (controlling for intermediate lags)
- Use to identify AR order

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| AR(p) | Tails off | Cuts off at lag p | AR(p) |
| MA(q) | Cuts off at lag q | Tails off | MA(q) |
| ARMA | Tails off | Tails off | ARMA |

### Step-by-Step: Decomposition

1. Click **🔄 Decomposition**
2. View the four components:
   - **Observed**: Original series
   - **Trend**: Long-term direction
   - **Seasonal**: Repeating pattern
   - **Residual**: What's left (should be noise)

---

## Causality Analysis

### ⚠️ Important Warning

**Granger causality ≠ True causation!**

Granger causality tests if X helps *predict* Y, not if X *causes* Y.

### Step-by-Step: Granger Causality Test

1. Go to **🔗 Causality** tab
2. Select the feature to test
3. Set max lag (start with 10)
4. Click **🔬 Granger Causality**
5. Interpret:
   - **p < 0.05**: X Granger-causes Y (has predictive power)
   - **p ≥ 0.05**: X does NOT Granger-cause Y

### Step-by-Step: Lead-Lag Analysis

1. Click **⏱️ Lead-Lag Analysis**
2. Find the best lag:
   - **Lag < 0**: X leads Y (X predicts future Y)
   - **Lag = 0**: Contemporaneous (move together)
   - **Lag > 0**: Y leads X (Y predicts future X)

### Example Interpretation

```
Best Lag: -3
Max Correlation: 0.75

Interpretation:
"Feature X leads Target Y by 3 periods with correlation 0.75"
"X might be a leading indicator for Y"
```

---

## Decision Flowcharts

### What Analysis Should I Use?

```
What's your goal?
    │
    ├── Understand my data
    │       └── 📊 Statistical → Descriptive Stats, Correlations
    │
    ├── Predict a value
    │       └── 🤖 ML → Train model, Cross-validate
    │
    ├── Understand uncertainty
    │       └── 🎲 Uncertainty → Bootstrap CI, Monte Carlo
    │
    ├── Check for non-linear patterns
    │       └── 🔀 Non-Linear → Distance Correlation, GP
    │
    ├── Analyze time-dependent data
    │       └── ⏱️ Time Series → ACF/PACF, ARIMA
    │
    └── Test if X predicts/causes Y
            └── 🔗 Causality → Granger Test, Lead-Lag
```

### Which Correlation Method?

```
What type of data?
    │
    ├── Continuous, normally distributed
    │       └── Pearson
    │
    ├── Continuous, non-normal OR monotonic relationship
    │       └── Spearman
    │
    ├── Ordinal (ranked categories)
    │       └── Kendall
    │
    └── Potentially non-linear
            └── Distance Correlation
```

### Which ML Model?

```
What do you need?
    │
    ├── Interpretable coefficients
    │       └── Linear Regression
    │
    ├── Handle multicollinearity
    │       └── Ridge Regression
    │
    ├── Automatic feature selection
    │       └── Lasso Regression
    │
    ├── Non-linear relationships
    │       └── Random Forest or Gradient Boosting
    │
    └── Best predictive accuracy
            └── Gradient Boosting (usually)
```

---

## Common Workflows

### Workflow 1: Quick Data Exploration

```
1. Load data                          [📁 Data Loading]
2. Check descriptive stats            [📊 Statistical]
3. Look at correlation heatmap        [📊 Statistical]
4. Create box plots for outliers      [📉 Visualization]
5. Check scatter matrix               [📉 Visualization]
```

### Workflow 2: Building a Predictive Model

```
1. Load data                          [📁 Data Loading]
2. Check for outliers                 [📊 Statistical]
3. Check correlations                 [📊 Statistical]
4. Check for non-linearity            [🔀 Non-Linear]
5. Train Linear Regression (baseline) [🤖 ML]
6. Try Random Forest                  [🤖 ML]
7. Cross-validate best model          [🤖 ML]
8. Check residuals                    [🎲 Uncertainty]
9. Get confidence intervals           [🎲 Uncertainty]
```

### Workflow 3: Time Series Forecasting

```
1. Load time series data              [📁 Data Loading]
2. Plot and visualize                 [📉 Visualization]
3. Test stationarity                  [⏱️ Time Series]
4. If non-stationary, difference      [External]
5. Examine ACF/PACF                   [⏱️ Time Series]
6. Decompose to see components        [⏱️ Time Series]
7. Fit ARIMA model                    [⏱️ Time Series]
```

### Workflow 4: Investigating Causal Relationships

```
1. Load data                          [📁 Data Loading]
2. Check correlations                 [📊 Statistical]
3. Check for non-linear relationships [🔀 Non-Linear]
4. Run lead-lag analysis              [🔗 Causality]
5. Test Granger causality             [🔗 Causality]
6. Interpret with caution!            [Remember: correlation ≠ causation]
```

---

## Tips for Best Results

### Data Preparation
- ✅ Remove or handle missing values
- ✅ Check for and address outliers
- ✅ Scale features for neural networks/SVM
- ✅ Log-transform heavily skewed variables

### Model Selection
- ✅ Start simple (Linear Regression)
- ✅ Use cross-validation for fair comparison
- ✅ Check residuals after fitting
- ✅ Consider interpretability vs accuracy trade-off

### Interpretation
- ✅ Report confidence intervals, not just point estimates
- ✅ Check statistical significance
- ✅ Be cautious about causal claims
- ✅ Validate on held-out data

### Performance
- ✅ Enable Rust acceleration for large datasets
- ✅ Start with fewer bootstrap samples, increase if needed
- ✅ Limit features in scatter matrix to 5

---

*Tutorial Version 9.0 - Last Updated December 2024*
