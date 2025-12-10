# Test Data Files

This directory contains synthetic datasets designed to test all features of the Advanced Data Analysis Toolkit.

## Files Overview

| File | Rows | Columns | Primary Use |
|------|------|---------|-------------|
| `general_analysis_data.csv` | 500 | 7 | Statistical analysis, outlier detection |
| `timeseries_data.csv` | 730 | 5 | Time series analysis, seasonality |
| `causality_data.csv` | 500 | 5 | Granger causality, lead-lag analysis |
| `clustering_data.csv` | 500 | 5 | K-Means, DBSCAN, PCA |
| `nonlinear_data.csv` | 400 | 13 | Distance correlation, mutual information |
| `bayesian_uncertainty_data.csv` | 300 | 5 | Bayesian regression, bootstrap CI |
| `regression_data.csv` | 600 | 10 | ML models, feature importance |
| `ml_classification_train.csv` | 300 | 8 | Classification model training (4-class customer segmentation) |
| `ml_classification_predict.csv` | 75 | 8 | Classification predictions with actual labels for evaluation |
| `ml_regression_train.csv` | 200 | 8 | Regression model training (house price prediction) |
| `ml_regression_predict.csv` | 50 | 8 | Regression predictions with actual values for evaluation |
| `signal_analysis_sample.csv` | 500 | 2 | Fourier analysis, wavelet analysis |

---

## Detailed Descriptions

### 1. general_analysis_data.csv

**Purpose**: General statistical analysis, correlation, outlier detection

**Columns**:
- `feature_1`: Normal distribution (μ=50, σ=10) with some outliers
- `feature_2`: Correlated with feature_1 (r ≈ 0.7)
- `feature_3`: Uniform distribution (0-100)
- `feature_4`: Exponential distribution with outliers
- `feature_5`: Normal distribution (μ=100, σ=25)
- `category`: Categorical column (A, B, C, D)
- `target`: Linear combination of features + noise

**Test These Features**:
- ✅ Descriptive statistics
- ✅ Correlation matrix (Pearson, Spearman)
- ✅ Distribution analysis
- ✅ Outlier detection (IQR and Z-score)
- ✅ Box plots
- ✅ Linear regression

---

### 2. timeseries_data.csv

**Purpose**: Time series analysis, trend detection, seasonality

**Columns**:
- `date`: Daily dates (2 years: 2022-01-01 to 2023-12-31)
- `sales`: Trend + annual + weekly seasonality + noise
- `revenue`: Different trend + seasonality pattern
- `temperature`: Stationary series (no trend)
- `stock_price`: Random walk (non-stationary)

**Test These Features**:
- ✅ ACF/PACF plots
- ✅ Stationarity tests (ADF test)
- ✅ Seasonal decomposition
- ✅ Rolling statistics
- ✅ ARIMA modeling (on stationary series)

**Expected Results**:
- `sales`, `revenue`: Non-stationary (has trend)
- `temperature`: Stationary
- `stock_price`: Non-stationary (random walk)

---

### 3. causality_data.csv

**Purpose**: Causality testing, lead-lag relationships

**Columns**:
- `leading_indicator`: The "cause" variable
- `lagged_3`: Follows leading_indicator with 3-step lag
- `lagged_5`: Follows leading_indicator with 5-step lag
- `independent`: Random, no causal relationship
- `outcome`: Influenced by leading_indicator and lagged_3

**Test These Features**:
- ✅ Granger causality test
- ✅ Lead-lag correlation analysis
- ✅ Correlation at different lags

**Expected Results**:
- `leading_indicator` should Granger-cause `lagged_3`, `lagged_5`, `outcome`
- Best lag for `lagged_3` should be around 3
- Best lag for `lagged_5` should be around 5
- `independent` should show no significant causality

---

### 4. clustering_data.csv

**Purpose**: Clustering algorithms, dimensionality reduction

**Columns**:
- `x`, `y`: 2D coordinates with 4 distinct clusters
- `z`: Derived feature (0.5x + 0.3y + noise)
- `w`: Random feature
- `true_cluster`: Ground truth labels (0, 1, 2, 3)

**Test These Features**:
- ✅ K-Means clustering (set k=4)
- ✅ DBSCAN clustering
- ✅ PCA (should show clear separation in first 2 components)
- ✅ Scatter plots

**Expected Results**:
- K-Means with k=4 should recover clusters well
- PCA first 2 components should show 4 distinct groups

---

### 5. nonlinear_data.csv

**Purpose**: Non-linear relationship detection

**Columns** (paired x-y relationships):
- `x_linear`, `y_linear`: Linear relationship (y = 2x + noise)
- `x_quadratic`, `y_quadratic`: Quadratic (y = x² + noise)
- `x_sinusoidal`, `y_sinusoidal`: Sine wave (y = 5sin(x) + noise)
- `x_circular`, `y_circular`: Circular pattern
- `x_exponential`, `y_exponential`: Exponential (y = e^x + noise)
- `x_independent`, `y_independent`: No relationship (random)
- `target`: Mixed combination

**Test These Features**:
- ✅ Distance correlation (detects non-linear relationships)
- ✅ Mutual information
- ✅ Polynomial regression
- ✅ Gaussian process regression
- ✅ Spline regression

**Expected Results**:
| Pair | Pearson r | Distance Correlation |
|------|-----------|---------------------|
| linear | High (~0.98) | High |
| quadratic | Low (~0) | High (~0.7) |
| sinusoidal | Low | High |
| circular | ~0 | High |
| exponential | Moderate | High |
| independent | ~0 | ~0 |

---

### 6. bayesian_uncertainty_data.csv

**Purpose**: Bayesian regression, uncertainty quantification

**Columns**:
- `predictor_1`: Normal (μ=10, σ=3)
- `predictor_2`: Normal (μ=5, σ=2)
- `predictor_3`: Normal (μ=0, σ=1)
- `response_homoscedastic`: Known linear model with constant noise (σ=2)
- `response_heteroscedastic`: Same model with varying noise

**True Model** (for validation):
```
y = 5.0 + 2.0*x1 - 1.5*x2 + 0.8*x3 + ε
where ε ~ N(0, 2.0)
```

**Test These Features**:
- ✅ Bayesian linear regression
- ✅ Bootstrap confidence intervals
- ✅ Prediction intervals
- ✅ Credible intervals
- ✅ Residual analysis

**Expected Results**:
- Recovered coefficients should be close to [2.0, -1.5, 0.8]
- Intercept should be close to 5.0
- 95% CI should contain true values

---

### 7. regression_data.csv

**Purpose**: ML model comparison, feature importance

**Columns**:
- `important_1`, `important_2`, `important_3`: High importance (coef: 3.0, 2.5, 2.0)
- `moderate_1`, `moderate_2`: Moderate importance (coef: 0.5, 0.3)
- `collinear_1`, `collinear_2`: Highly correlated with important_1, important_2
- `noise_1`, `noise_2`: Pure noise (no relationship)
- `target`: Linear combination with slight non-linearity

**Test These Features**:
- ✅ Linear regression
- ✅ Ridge/Lasso/ElasticNet (should handle collinearity)
- ✅ Random Forest (feature importance)
- ✅ Gradient Boosting
- ✅ Cross-validation
- ✅ Feature importance ranking

**Expected Results**:
- Feature importance: important_1 > important_2 > important_3 > moderate > noise
- Ridge/Lasso should shrink collinear coefficients
- R² should be high (~0.9+) for good models

---

## Quick Testing Workflow

1. **Load** `general_analysis_data.csv`
   - Go to Statistical tab → Run all analyses
   - Check outlier detection finds ~15 outliers

2. **Load** `timeseries_data.csv`
   - Go to Time Series tab
   - Run stationarity test on each column
   - Check decomposition on `sales`

3. **Load** `causality_data.csv`
   - Go to Causality tab
   - Test Granger causality: `leading_indicator` → `outcome`
   - Run lead-lag analysis

4. **Load** `clustering_data.csv`
   - Go to ML tab
   - Run K-Means with k=4
   - Run PCA, visualize first 2 components

5. **Load** `nonlinear_data.csv`
   - Go to Non-Linear tab
   - Compare Pearson vs Distance correlation
   - Test GP regression on sinusoidal pair

6. **Load** `bayesian_uncertainty_data.csv`
   - Go to Bayesian tab
   - Run Bayesian regression
   - Go to Uncertainty tab
   - Run Bootstrap CI, verify coefficients

7. **Load** `regression_data.csv`
   - Go to ML tab
   - Compare Linear, Ridge, Random Forest
   - Check feature importance ranking

---

## Plugin Testing

Use these datasets to test the example plugins:

- **lag_features.py**: Use with `timeseries_data.csv`
- **outlier_removal.py**: Use with `general_analysis_data.csv`
- **enhanced_scatter.py**: Use with `nonlinear_data.csv` (quadratic pair)

---

## Machine Learning Test Files

### 8. ml_classification_train.csv

**Purpose**: Training classification models (Logistic Regression, SVM, Decision Trees, Random Forest, KNN, Naive Bayes)

**Columns**:
- `sepal_length`: Sepal length in cm (iris-like data)
- `sepal_width`: Sepal width in cm
- `petal_length`: Petal length in cm
- `petal_width`: Petal width in cm
- `species`: Target class (setosa, versicolor, virginica)
- `is_setosa`: Binary target (1 = setosa, 0 = not setosa)

**Test These Features**:
- ✅ Logistic Regression with `is_setosa` as binary target
- ✅ SVM (Support Vector Machine) classification
- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier
- ✅ KNN (K-Nearest Neighbors) Classifier
- ✅ Naive Bayes Classifier
- ✅ Confusion matrix visualization
- ✅ Classification metrics (accuracy, precision, recall, F1-score)

**Expected Results**:
- High accuracy (>90%) for most classifiers on this well-separated data
- Setosa should be easily separable from other species
- Versicolor and virginica may have some overlap

---

### 9. ml_classification_predict.csv

**Purpose**: Making predictions with trained classification models

**Columns**:
- `sepal_length`, `sepal_width`, `petal_length`, `petal_width`: Input features
- (No target column - this is for prediction only)

**Usage**:
1. Train a model on `ml_classification_train.csv`
2. Load `ml_classification_predict.csv` 
3. Use the trained model to predict species/is_setosa

---

### 10. ml_regression_train.csv

**Purpose**: Training regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree, KNN, SVR, Random Forest, Gradient Boosting)

**Columns**:
- `temperature`: Temperature in °C (15-35 range)
- `humidity`: Relative humidity % (30-90 range)
- `wind_speed`: Wind speed in km/h (0-20 range)
- `pressure`: Atmospheric pressure in hPa (1000-1030 range)
- `solar_radiation`: Solar radiation in W/m² (100-500 range)
- `energy_output`: Target variable - power plant energy output (MW)

**Test These Features**:
- ✅ Linear Regression (baseline model)
- ✅ Ridge Regression (L2 regularization)
- ✅ Lasso Regression (L1 regularization)
- ✅ ElasticNet (L1+L2 regularization)
- ✅ Decision Tree Regressor
- ✅ KNN Regressor
- ✅ SVR (Support Vector Regression)
- ✅ Random Forest Regressor
- ✅ Gradient Boosting Regressor
- ✅ Feature importance analysis
- ✅ R², MAE, MSE, RMSE metrics

**Expected Results**:
- R² should be moderate (~0.5-0.8) due to realistic noise
- Solar radiation and temperature should be top predictors
- Tree-based models may capture non-linear relationships better

---

### 11. ml_regression_predict.csv

**Purpose**: Making predictions with trained regression models

**Columns**:
- `temperature`, `humidity`, `wind_speed`, `pressure`, `solar_radiation`: Input features
- (No energy_output target - this is for prediction only)

**Usage**:
1. Train a model on `ml_regression_train.csv`
2. Load `ml_regression_predict.csv`
3. Use the trained model to predict energy output

---

## ML Testing Workflow

### Classification Workflow:
1. **Load** `ml_classification_train.csv`
2. Go to **ML (Machine Learning)** tab
3. Select **Task Type: Classification**
4. Select features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
5. Select target: `is_setosa` (binary) or `species` (multi-class)
6. Choose model (e.g., Logistic Regression)
7. Click **Train Model**
8. Review confusion matrix and metrics
9. Load `ml_classification_predict.csv` for predictions

### Regression Workflow:
1. **Load** `ml_regression_train.csv`
2. Go to **ML (Machine Learning)** tab
3. Select **Task Type: Regression**
4. Select features: `temperature`, `humidity`, `wind_speed`, `pressure`, `solar_radiation`
5. Select target: `energy_output`
6. Choose model (e.g., Random Forest)
7. Click **Train Model**
8. Review R², MAE, MSE metrics
9. Load `ml_regression_predict.csv` for predictions
