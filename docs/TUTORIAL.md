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
11. [Community Ecology](#community-ecology)
12. [Ordination](#ordination)
13. [Multivariate Hypothesis Tests](#multivariate-hypothesis-tests)
14. [Curve Fitting & Non-Linear Models](#curve-fitting--non-linear-models)
15. [Plugin System](#plugin-system)
16. [Decision Flowcharts](#decision-flowcharts)
17. [Common Workflows](#common-workflows)

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
| 🌿 Community Ecology | Alpha/beta diversity, rarefaction, SHE analysis |
| 🧭 Ordination | PCoA, NMDS, CA, DCA, CCA, RDA, Mantel test |
| 📊 Multivariate Tests | PERMANOVA, ANOSIM, SIMPER, MANOVA, Hotelling T², LDA |
| 📈 Curve Fitting | Power, exponential, logistic, Gompertz, RMA, GLM |
| 🔌 Plugins | Extend with custom analysis, preprocessing, or visualisation |

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
┌─────────────────────────────────────────────────────────────────────────┐
│  📊 Advanced Data Analysis Toolkit                   [Sidebar: Tutorial]│
├─────────────────────────────────────────────────────────────────────────┤
│  📁 Data | 📊 Statistics | 🔊 Signal | ⏱️ Time Series | 🤖 ML         │
│  🔬 Scientific Tools | 📈 Visualization | 📋 Reports | 🔌 Plugins     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                        [Main Content Area]                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
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

## Community Ecology

**Location:** 🔬 Scientific Tools → Community Ecology

### Data Format

Your data should have **sites as rows** and **species as columns**:

| site | sp_A | sp_B | sp_C |
|------|------|------|------|
| S1   | 12   | 0    | 5    |
| S2   | 0    | 8    | 7    |
| S3   | 4    | 3    | 0    |

### Step-by-Step: Alpha Diversity

1. Go to **🔬 Scientific Tools → Community Ecology**
2. Select the **Alpha Diversity** subtab
3. Choose species columns (numeric abundance columns)
4. Optionally select a sample label column
5. Click **Calculate Alpha Diversity**
6. Review: Shannon H', Simpson, Chao1, evenness, and more

**Interpreting indices:**

| Index | Meaning |
|-------|---------|
| **Shannon H'** | Higher = more diverse (0–~5 in practice) |
| **Simpson 1-D** | Higher = more even dominance (0–1) |
| **Chao1** | Estimated total species richness (≥ observed) |
| **Pielou J'** | Evenness: 1 = perfectly even, 0 = one dominant species |

### Step-by-Step: Beta Diversity

1. Select the **Beta Diversity** subtab
2. Choose species columns
3. Select a dissimilarity metric (Bray-Curtis for abundances, Jaccard for presence/absence)
4. View the distance matrix heatmap
5. Sites with low values (blue) are more similar

### Step-by-Step: Rarefaction

1. Select the **Rarefaction** subtab
2. Choose species columns
3. Set the maximum sample size
4. Review the rarefaction curves:
   - **Plateau** = sampling is sufficient
   - **Still rising** = more species likely remain undetected

### Step-by-Step: Species Accumulation

1. Select the **Accumulation** subtab
2. Choose species columns
3. The curve shows cumulative richness as more samples are added
4. Approaching an asymptote indicates adequate sampling effort

---

## Ordination

**Location:** 🔬 Scientific Tools → Ordination

### Choosing a Method

```
Do you have environmental variables?
    │
    ├── No  → Unconstrained
    │       Is species response unimodal?
    │           ├── Yes → CA / DCA
    │           └── No  → PCoA / NMDS
    │
    └── Yes → Constrained
            Is species response unimodal?
                ├── Yes → CCA
                └── No  → RDA
```

### Step-by-Step: PCoA / NMDS

1. Go to **🔬 Scientific Tools → Ordination**
2. Select **PCoA** or **NMDS**
3. Choose species columns (numeric)
4. Select a distance metric (Bray-Curtis is common)
5. Optionally choose a grouping column to colour points
6. Click **Run Ordination**
7. Interpret:
   - Points close together = similar community composition
   - For NMDS: check stress < 0.2

### Step-by-Step: CCA / RDA

1. Select **CCA** or **RDA**
2. Choose species columns (response)
3. Choose environmental columns (predictors)
4. Click **Run Ordination**
5. The biplot shows:
   - **Points** = sites in ordination space
   - **Arrows** = environmental gradients (longer = stronger effect)
   - **Species labels** (CCA) = species optima along gradients

### Step-by-Step: Mantel Test

1. Select **Mantel Test**
2. Choose columns for distance matrix 1
3. Choose columns for distance matrix 2
4. Select distance metrics and number of permutations
5. Interpret: significant p-value means the two distance matrices are correlated

---

## Multivariate Hypothesis Tests

**Location:** 🔬 Scientific Tools → Multivariate Tests

### Choosing a Test

| Situation | Test |
|-----------|------|
| Non-normal data, group differences | **PERMANOVA** |
| Rank-based group comparison | **ANOSIM** |
| Which species cause the difference? | **SIMPER** |
| Normal data, 3+ groups | **MANOVA** |
| Normal data, 2 groups | **Hotelling T²** |
| Classify and visualise groups | **LDA** |

### Step-by-Step: PERMANOVA

1. Go to **🔬 Scientific Tools → Multivariate Tests**
2. Select **PERMANOVA**
3. Choose response variables (numeric species/measurement columns)
4. Choose the grouping variable (categorical)
5. Select distance metric and number of permutations
6. Click **Run Test**
7. Interpret:
   - **Pseudo-F**: larger = stronger group effect
   - **p < 0.05**: groups differ significantly
   - **R²**: proportion of variation explained by groups

### Step-by-Step: SIMPER (after PERMANOVA)

1. Select **SIMPER** from the dropdown
2. Use the same response and grouping columns
3. Click **Run Test**
4. The results table shows:
   - Each species' contribution to between-group dissimilarity
   - Cumulative percentage (species ranked by contribution)
   - Mean abundance per group

### Step-by-Step: LDA (Discriminant Analysis)

1. Select **Discriminant Analysis (LDA)**
2. Choose measurement columns and grouping variable
3. Click **Run Test**
4. Review:
   - Classification accuracy (cross-validated)
   - Canonical variate plot (2D separation of groups)
   - Variable loadings (which variables best separate groups)

---

## Curve Fitting & Non-Linear Models

**Location:** 🔬 Scientific Tools → Curve Fitting

### Available Models

| Model | Equation | When to Use |
|-------|----------|-------------|
| Power | y = a·xᵇ | Allometric scaling, species-area |
| Exponential (2p) | y = a·e^(bx) | Unbounded growth/decay |
| Exponential (3p) | y = a·e^(bx) + c | Growth/decay with baseline |
| Logistic (4p) | y = d + (a-d)/(1+(x/c)^b) | Dose-response, S-curves |
| Sinusoidal | y = A·sin(2πfx + φ) + offset | Seasonal/periodic |
| Gompertz | y = a·e^(−b·e^(−cx)) | Asymmetric growth |
| RMA Regression | y = a + bx (Type II) | Both X and Y have error |
| GLM | g(E[y]) = Xβ | Generalised linear model |

### Step-by-Step: Single Model Fit

1. Go to **🔬 Scientific Tools → Curve Fitting**
2. Select **Single Model Fit** mode
3. Choose X (predictor) and Y (response) columns
4. Select a model from the dropdown
5. Click **Fit Model**
6. Review: fitted parameters, R², RMSE, AIC
7. Inspect the plot: data points + fitted curve

### Step-by-Step: Multi-Model Comparison

1. Select **Multi-Model Comparison** mode
2. Choose X and Y columns
3. Select which models to compare (checkboxes)
4. Click **Compare All**
5. The ranking table shows AIC, BIC, R², RMSE for each model
6. ΔAIC interpretation:
   - < 2: competing models (both supported)
   - 2–10: less support
   - > 10: essentially no support

### When to Use RMA vs OLS

| Scenario | Use |
|----------|-----|
| Only Y has measurement error | Standard OLS regression |
| Both X and Y have error | RMA regression |
| Common in: allometry, method comparison | RMA is preferred |

---

## Plugin System

**Location:** 🔌 Plugins

### Step-by-Step: Loading a Plugin

1. Go to the **🔌 Plugins** tab
2. Choose a loading method:
   - **From file**: Upload a `.py` file
   - **Paste code**: Write/paste Python code
   - **Example templates**: Load a bundled example
3. The plugin appears in the loaded plugins list

### Step-by-Step: Running a Plugin

1. Select a loaded plugin from the list
2. Configure any parameters (auto-generated from PLUGIN_PARAMETERS)
3. Click **Execute**
4. View results in the output area
5. If the plugin returns data, download it as CSV

### Creating Your Own Plugin

Every plugin needs:
1. `PLUGIN_INFO` dict — name, description, category
2. `PLUGIN_PARAMETERS` dict — configurable parameters with types/defaults
3. `process(data, columns, target, **params)` — main analysis function

See `example_plugins/` for ready-to-use templates.

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
    ├── Test if X predicts/causes Y
    │       └── 🔗 Causality → Granger Test, Lead-Lag
    │
    ├── Analyse species communities
    │       └── 🔬 Scientific Tools → Ecology, Ordination, Multivariate Tests
    │
    ├── Fit a non-linear model
    │       └── 🔬 Scientific Tools → Curve Fitting
    │
    └── Custom analysis
            └── 🔌 Plugins → Load and execute
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

### Workflow 5: Community Ecology Analysis

```
1. Load species × sites data          [📁 Data Loading]
2. Calculate alpha diversity           [🔬 Scientific Tools → Ecology]
3. Compare sites with beta diversity   [🔬 Scientific Tools → Ecology]
4. Check sampling with rarefaction     [🔬 Scientific Tools → Ecology]
5. Ordinate sites (PCoA or NMDS)       [🔬 Scientific Tools → Ordination]
6. Test group differences (PERMANOVA)  [🔬 Scientific Tools → Multivariate Tests]
7. Identify key species (SIMPER)       [🔬 Scientific Tools → Multivariate Tests]
```

### Workflow 6: Curve Fitting

```
1. Load data                           [📁 Data Loading]
2. Visualise scatter plot               [📈 Visualization]
3. Try Multi-Model Comparison           [🔬 Scientific Tools → Curve Fitting]
4. Select best model (lowest AIC)       [🔬 Scientific Tools → Curve Fitting]
5. Inspect residuals                    [🔬 Scientific Tools → Curve Fitting]
6. Report parameters + R² + AIC        [📋 Reports]
```

### Workflow 7: Extending with Plugins

```
1. Load data                           [📁 Data Loading]
2. Browse example templates             [🔌 Plugins]
3. Load or paste a custom plugin        [🔌 Plugins]
4. Configure parameters                 [🔌 Plugins]
5. Execute on your data                 [🔌 Plugins]
6. Download results                     [🔌 Plugins]
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

*Tutorial Version 9.1 - Last Updated April 2026*
