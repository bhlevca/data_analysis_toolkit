# 📚 API Reference — Advanced Data Analysis Toolkit v4.0

This document provides comprehensive API documentation for all new modules added in v4.0, designed to enhance scientific research and data analysis capabilities.

---

## Table of Contents

1. [Effect Sizes Module](#effect-sizes-module)
2. [Model Validation Module](#model-validation-module)
3. [Data Quality Module](#data-quality-module)
4. [Feature Selection Module](#feature-selection-module)
5. [Report Generator Module](#report-generator-module)
6. [Interpretability Module](#interpretability-module)
7. [Survival Analysis Module](#survival-analysis-module)
8. [Advanced Time Series Module](#advanced-time-series-module)
9. [Domain-Specific Analysis Module](#domain-specific-analysis-module)
10. [Statistical Analysis Enhancements](#statistical-analysis-enhancements)

---

## Effect Sizes Module

**Location**: `data_toolkit.effect_sizes`

Calculates standardized effect sizes with confidence intervals for proper scientific reporting.

### Class: EffectSizeCalculator

```python
from data_toolkit.effect_sizes import EffectSizeCalculator

calc = EffectSizeCalculator(df)
```

#### Methods

##### `cohens_d(group1_col, group2_col, confidence_level=0.95)`
Calculates Cohen's d for two independent groups.

**Parameters:**
- `group1_col` (str): Column name for first group
- `group2_col` (str): Column name for second group
- `confidence_level` (float): Confidence level for CI (default: 0.95)

**Returns:** dict with keys:
- `cohens_d`: Effect size value
- `ci_lower`, `ci_upper`: Confidence interval bounds
- `interpretation`: Small/Medium/Large

**Interpretation:**
| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

```python
result = calc.cohens_d('treatment_scores', 'control_scores')
print(f"d = {result['cohens_d']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

##### `hedges_g(group1_col, group2_col, confidence_level=0.95)`
Hedges' g with small-sample bias correction.

##### `glass_delta(group1_col, group2_col)`
Glass's Δ using control group SD as denominator.

##### `eta_squared(between_ss, total_ss)`
Effect size for ANOVA (proportion of variance explained).

##### `omega_squared(f_statistic, df_between, df_within, n)`
Omega squared with population estimate adjustment.

##### `cramers_v(contingency_table)`
Effect size for chi-square tests on contingency tables.

##### `phi_coefficient(contingency_table)`
Phi coefficient for 2×2 contingency tables.

##### `odds_ratio(table_2x2, confidence_level=0.95)`
Calculates odds ratio with CI for 2×2 tables.

##### `risk_ratio(table_2x2, confidence_level=0.95)`
Calculates relative risk with CI.

##### `r_to_d(r)`
Converts correlation coefficient to Cohen's d.

##### `d_to_r(d)`
Converts Cohen's d to correlation coefficient.

---

## Model Validation Module

**Location**: `data_toolkit.model_validation`

Rigorous cross-validation, calibration analysis, and diagnostic tools.

### Class: ModelValidator

```python
from data_toolkit.model_validation import ModelValidator

validator = ModelValidator(df)
```

#### Methods

##### `cross_validate(model, feature_cols, target_col, cv=5, scoring=None)`
K-fold cross-validation with multiple metrics.

**Parameters:**
- `model`: Scikit-learn compatible model
- `feature_cols` (list): Feature column names
- `target_col` (str): Target column name
- `cv` (int): Number of folds (default: 5)
- `scoring` (list): Metrics to compute

**Returns:** dict with:
- `scores`: Dict of metric arrays per fold
- `mean_scores`: Mean of each metric
- `std_scores`: Std of each metric
- `fold_details`: Per-fold breakdown

```python
from sklearn.ensemble import RandomForestClassifier

result = validator.cross_validate(
    RandomForestClassifier(),
    ['feature_1', 'feature_2'],
    'target',
    cv=5,
    scoring=['accuracy', 'f1', 'roc_auc']
)
print(f"Mean accuracy: {result['mean_scores']['accuracy']:.3f} ± {result['std_scores']['accuracy']:.3f}")
```

##### `nested_cross_validation(model, param_grid, feature_cols, target_col, outer_cv=5, inner_cv=3)`
Nested CV for unbiased performance estimation during hyperparameter tuning.

**Parameters:**
- `model`: Base model
- `param_grid` (dict): Hyperparameter search space
- `outer_cv` (int): Outer fold count
- `inner_cv` (int): Inner fold count

**Returns:** dict with:
- `outer_scores`: Scores from outer loop
- `best_params`: Best params per outer fold
- `mean_score`, `std_score`: Summary statistics

##### `learning_curve_analysis(model, feature_cols, target_col, train_sizes=None, cv=5)`
Analyzes model performance vs training set size.

**Returns:** dict with:
- `train_sizes`: Actual training sizes
- `train_scores`: Training scores
- `test_scores`: Test scores
- `bias_variance_tradeoff`: Analysis summary

##### `calibration_analysis(model, feature_cols, target_col, n_bins=10)`
Checks probability calibration for classifiers.

**Returns:** dict with:
- `fraction_positives`: Actual positive rate per bin
- `mean_predicted_value`: Mean predicted probability per bin
- `brier_score`: Brier score loss
- `calibration_error`: Expected calibration error

##### `roc_analysis(model, feature_cols, target_col, cv=5)`
ROC curve analysis with cross-validated AUC.

**Returns:** dict with:
- `fpr`, `tpr`: ROC curve points
- `auc`: Area under curve
- `optimal_threshold`: Youden's J optimal threshold
- `sensitivity`, `specificity`: At optimal threshold

##### `residual_diagnostics(model, feature_cols, target_col)`
Comprehensive residual analysis for regression.

**Returns:** dict with:
- `residuals`: Residual values
- `standardized_residuals`: Standardized residuals
- `normality_test`: Shapiro-Wilk test result
- `heteroscedasticity_test`: Breusch-Pagan test
- `durbin_watson`: Autocorrelation statistic
- `influential_points`: Cook's distance outliers

---

## Data Quality Module

**Location**: `data_toolkit.data_quality`

Comprehensive missing data analysis, imputation, and quality assessment.

### Class: DataQualityAnalyzer

```python
from data_toolkit.data_quality import DataQualityAnalyzer

dqa = DataQualityAnalyzer(df)
```

#### Methods

##### `missing_data_summary(columns=None)`
Comprehensive missing data statistics.

**Returns:** DataFrame with columns:
- `missing_count`: Number of missing values
- `missing_pct`: Percentage missing
- `dtype`: Data type
- `unique_values`: Count of unique values

```python
summary = dqa.missing_data_summary()
print(summary[summary['missing_pct'] > 5])  # Columns with >5% missing
```

##### `missing_pattern_analysis(columns=None)`
Analyzes patterns of missingness.

**Returns:** dict with:
- `patterns`: DataFrame of missingness patterns
- `pattern_counts`: Frequency of each pattern
- `correlation_matrix`: Missingness correlations

##### `little_mcar_test(columns=None)`
Little's MCAR test for Missing Completely At Random.

**Returns:** dict with:
- `chi_square`: Test statistic
- `df`: Degrees of freedom
- `p_value`: Significance
- `is_mcar`: Boolean (p > 0.05 suggests MCAR)

##### `impute_missing(columns, method='mean', **kwargs)`
Imputes missing values with various methods.

**Parameters:**
- `columns` (list): Columns to impute
- `method` (str): One of:
  - `'mean'`, `'median'`, `'mode'`: Simple imputation
  - `'knn'`: K-Nearest Neighbors (requires `n_neighbors`)
  - `'mice'`: Multiple Imputation by Chained Equations
  - `'regression'`: Regression-based imputation

**Returns:** dict with:
- `imputed_data`: DataFrame with imputed values
- `imputation_details`: Method-specific details

```python
result = dqa.impute_missing(['income', 'age'], method='knn', n_neighbors=5)
df_imputed = result['imputed_data']
```

##### `multiple_imputation(columns, n_imputations=5)`
Creates multiple imputed datasets for proper uncertainty quantification.

**Returns:** dict with:
- `imputed_datasets`: List of imputed DataFrames
- `pooled_statistics`: Rubin's rules combined estimates

##### `detect_outliers(columns, method='iqr', threshold=1.5)`
Detects outliers using multiple methods.

**Parameters:**
- `method` (str): `'iqr'`, `'zscore'`, `'mad'`, `'isolation_forest'`
- `threshold` (float): Method-specific threshold

**Returns:** dict with:
- `outlier_mask`: Boolean mask of outliers
- `outlier_indices`: Row indices
- `outlier_counts`: Per-column counts
- `summary`: Overall summary

##### `transform_data(columns, method='log')`
Applies transformations for normality or variance stabilization.

**Parameters:**
- `method` (str): `'log'`, `'sqrt'`, `'boxcox'`, `'yeojohnson'`, `'zscore'`, `'minmax'`, `'robust'`

**Returns:** dict with:
- `transformed_data`: Transformed DataFrame
- `transformation_params`: Parameters for inverse transform

##### `generate_quality_report(output_path=None)`
Generates comprehensive data quality report.

**Returns:** dict with complete quality assessment including:
- Missing data analysis
- Outlier detection
- Distribution analysis
- Data type issues
- Duplicate detection

---

## Feature Selection Module

**Location**: `data_toolkit.feature_selection`

Multiple feature selection methods for optimal model building.

### Class: FeatureSelector

```python
from data_toolkit.feature_selection import FeatureSelector

selector = FeatureSelector(df)
```

#### Methods

##### `recursive_feature_elimination(feature_cols, target_col, n_features=None, step=1)`
Recursive Feature Elimination with cross-validation.

**Returns:** dict with:
- `selected_features`: List of selected feature names
- `feature_ranking`: Ranking of all features
- `cv_scores`: Cross-validation scores

```python
result = selector.recursive_feature_elimination(
    ['f1', 'f2', 'f3', 'f4', 'f5'],
    'target',
    n_features=3
)
print("Selected:", result['selected_features'])
```

##### `boruta_selection(feature_cols, target_col, max_iter=100)`
All-relevant feature selection using Boruta algorithm.

**Returns:** dict with:
- `confirmed_features`: Definitely relevant features
- `tentative_features`: Possibly relevant
- `rejected_features`: Not relevant
- `feature_importances`: Importance scores

##### `shap_selection(model, feature_cols, target_col, n_features=None)`
SHAP-based feature selection (requires `shap` package).

**Returns:** dict with:
- `selected_features`: Top features by SHAP importance
- `shap_values`: SHAP importance scores
- `feature_ranking`: Ordered feature list

##### `statistical_selection(feature_cols, target_col, method='mutual_info', k=10)`
Statistical tests for feature selection.

**Parameters:**
- `method` (str): `'mutual_info'`, `'f_classif'`, `'f_regression'`, `'chi2'`
- `k` (int): Number of features to select

**Returns:** dict with:
- `selected_features`: Top k features
- `scores`: Statistical scores
- `p_values`: P-values (where applicable)

##### `permutation_selection(model, feature_cols, target_col, n_repeats=10)`
Permutation importance-based selection.

**Returns:** dict with:
- `importances_mean`: Mean importance decrease
- `importances_std`: Standard deviation
- `selected_features`: Features above threshold

##### `lasso_selection(feature_cols, target_col, alpha=None)`
L1-regularization for sparse feature selection.

**Returns:** dict with:
- `selected_features`: Non-zero coefficient features
- `coefficients`: Lasso coefficients
- `optimal_alpha`: Selected alpha (if None provided)

##### `sequential_selection(feature_cols, target_col, direction='forward', n_features=None)`
Forward or backward stepwise selection.

**Parameters:**
- `direction` (str): `'forward'` or `'backward'`

**Returns:** dict with:
- `selected_features`: Ordered list of selected features
- `cv_scores`: Score at each step

##### `ensemble_selection(feature_cols, target_col, methods=None)`
Combines multiple methods for robust selection.

**Returns:** dict with:
- `consensus_features`: Features selected by multiple methods
- `method_results`: Individual method results
- `feature_votes`: Vote counts per feature

---

## Report Generator Module

**Location**: `data_toolkit.report_generator`

Automated generation of publication-ready analysis reports.

### Class: ReportGenerator

```python
from data_toolkit.report_generator import ReportGenerator, COMMON_CITATIONS

report = ReportGenerator(title="My Analysis Report")
```

#### Methods

##### `add_section(title, content, level=2)`
Adds a text section to the report.

```python
report.add_section("Introduction", "This study examines...")
report.add_section("Methods", "We used linear regression...", level=3)
```

##### `add_data_provenance(df, source_file=None, preprocessing_steps=None)`
Documents data origin and preprocessing.

```python
report.add_data_provenance(
    df,
    source_file="experiment_data.csv",
    preprocessing_steps=[
        "Removed duplicates",
        "Imputed missing values with median",
        "Log-transformed skewed variables"
    ]
)
```

##### `add_statistics_table(stats_dict, title="Statistical Results", caption=None)`
Adds a formatted statistics table.

```python
report.add_statistics_table(
    {'Mean': 45.2, 'SD': 12.3, 't-statistic': 3.45, 'p-value': 0.001},
    title="T-test Results"
)
```

##### `add_figure(fig, title, caption=None, format='png')`
Adds a matplotlib/plotly figure.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y)
report.add_figure(fig, "Trend Analysis", caption="Figure 1: Temporal trends")
```

##### `add_model_summary(model_results, model_name)`
Adds formatted model results.

##### `add_citation(key, full_citation=None)`
Adds a citation reference.

```python
report.add_citation('scipy')  # Uses COMMON_CITATIONS
report.add_citation('custom', "Author, A. (2024). Title. Journal, 1(1), 1-10.")
```

##### `generate_html(output_path=None)`
Generates HTML report.

**Returns:** HTML string (also saves to file if path provided)

##### `generate_markdown(output_path=None)`
Generates Markdown report.

```python
# Generate both formats
report.generate_html("report.html")
report.generate_markdown("report.md")
```

### Constants

```python
from data_toolkit.report_generator import COMMON_CITATIONS

# Available citations:
# 'scipy', 'numpy', 'pandas', 'scikit-learn', 'statsmodels', 
# 'matplotlib', 'seaborn', 'shap', 'lifelines'
```

---

## Interpretability Module

**Location**: `data_toolkit.interpretability`

Requires: `pip install data-toolkit[interpretability]`

### Class: ModelInterpreter

```python
from data_toolkit.interpretability import ModelInterpreter

interp = ModelInterpreter(model, df, feature_cols)
```

#### Methods

##### `shap_analysis(n_samples=100)`
SHAP (SHapley Additive exPlanations) analysis.

**Returns:** dict with:
- `shap_values`: SHAP values array
- `expected_value`: Base prediction
- `feature_importance`: Mean absolute SHAP values

```python
result = interp.shap_analysis()
# Access for plotting
import shap
shap.summary_plot(result['shap_values'], df[feature_cols])
```

##### `permutation_feature_importance(n_repeats=10)`
Model-agnostic permutation importance.

**Returns:** dict with:
- `importances_mean`: Mean importance
- `importances_std`: Standard deviation
- `feature_ranking`: Ordered features

##### `partial_dependence(feature, grid_resolution=50)`
Partial Dependence Plot data.

**Returns:** dict with:
- `values`: Feature values
- `average_prediction`: Mean prediction at each value
- `individual_predictions`: ICE curves (optional)

##### `lime_explain(instance_idx, n_features=10)`
LIME local explanation for a single prediction.

**Returns:** dict with:
- `explanation`: Feature contributions
- `local_prediction`: Predicted value
- `intercept`: LIME intercept

```python
# Explain prediction for row 42
explanation = interp.lime_explain(42)
for feature, weight in explanation['explanation']:
    print(f"{feature}: {weight:+.3f}")
```

##### `feature_interactions(top_n=10)`
Detects feature interactions using SHAP.

**Returns:** dict with:
- `interaction_values`: Pairwise interaction strengths
- `top_interactions`: Most important interactions

---

## Survival Analysis Module

**Location**: `data_toolkit.survival_analysis`

Requires: `pip install data-toolkit[survival]`

### Class: SurvivalAnalyzer

```python
from data_toolkit.survival_analysis import SurvivalAnalyzer

surv = SurvivalAnalyzer(df)
```

#### Methods

##### `kaplan_meier(time_col, event_col, group_col=None, confidence_level=0.95)`
Kaplan-Meier survival curve estimation.

**Parameters:**
- `time_col` (str): Time-to-event column
- `event_col` (str): Event indicator (1=event, 0=censored)
- `group_col` (str): Optional grouping variable

**Returns:** dict with:
- `survival_function`: DataFrame with time, survival probability, CI
- `median_survival`: Median survival time
- `survival_at_times`: Survival probabilities at key timepoints

```python
result = surv.kaplan_meier('time_to_event', 'event', group_col='treatment')
# Plot
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
for group, data in result['survival_function'].items():
    # ... plot each group
```

##### `cox_regression(time_col, event_col, covariate_cols, **kwargs)`
Cox Proportional Hazards regression.

**Returns:** dict with:
- `coefficients`: Hazard ratios and CIs
- `p_values`: Significance tests
- `concordance_index`: C-index
- `log_likelihood`: Model fit
- `proportional_hazards_test`: PH assumption test

```python
result = surv.cox_regression(
    'survival_time', 
    'death',
    ['age', 'treatment', 'stage']
)
print("Hazard Ratios:")
print(result['coefficients'])
```

##### `log_rank_test(time_col, event_col, group_col)`
Log-rank test for comparing survival curves.

**Returns:** dict with:
- `test_statistic`: Chi-square statistic
- `p_value`: Significance
- `degrees_of_freedom`: df

##### `cumulative_hazard(time_col, event_col, method='nelson-aalen')`
Nelson-Aalen cumulative hazard estimation.

##### `parametric_survival(time_col, event_col, distribution='weibull')`
Parametric survival models.

**Parameters:**
- `distribution` (str): `'weibull'`, `'exponential'`, `'lognormal'`, `'loglogistic'`

**Returns:** dict with:
- `parameters`: Distribution parameters
- `aic`, `bic`: Model selection criteria
- `survival_function`: Fitted survival curve

---

## Advanced Time Series Module

**Location**: `data_toolkit.advanced_timeseries`

Requires: `pip install data-toolkit[timeseries]`

### Class: AdvancedTimeSeriesAnalysis

```python
from data_toolkit.advanced_timeseries import AdvancedTimeSeriesAnalysis

ats = AdvancedTimeSeriesAnalysis(df)
```

#### Methods

##### `prophet_forecast(time_col, value_col, periods=30, **kwargs)`
Facebook Prophet forecasting.

**Returns:** dict with:
- `forecast`: DataFrame with predictions, lower/upper bounds
- `components`: Trend, seasonality components
- `model`: Fitted Prophet model

```python
result = ats.prophet_forecast('date', 'sales', periods=90)
forecast = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

##### `detect_changepoints(column, method='pelt', **kwargs)`
Changepoint detection (requires `ruptures`).

**Parameters:**
- `method` (str): `'pelt'`, `'binseg'`, `'window'`, `'dynp'`

**Returns:** dict with:
- `changepoints`: List of changepoint indices
- `segments`: Segment boundaries
- `n_changepoints`: Count

##### `dtw_distance(series1, series2, **kwargs)`
Dynamic Time Warping distance (requires `dtaidistance`).

**Returns:** dict with:
- `distance`: DTW distance
- `path`: Warping path
- `normalized_distance`: Length-normalized distance

##### `var_analysis(columns, maxlags=None)`
Vector Autoregression for multivariate series.

**Returns:** dict with:
- `model`: Fitted VAR model
- `coefficients`: VAR coefficients
- `aic`, `bic`: Model selection criteria
- `forecast`: Multi-step forecast

##### `granger_causality(cause_col, effect_col, maxlag=10)`
Granger causality test.

**Returns:** dict with:
- `results`: Test statistics at each lag
- `optimal_lag`: Lag with strongest evidence
- `is_causal`: Boolean (any significant lag)

##### `cointegration_test(columns, method='johansen')`
Cointegration testing for non-stationary series.

**Returns:** dict with:
- `trace_statistic`: Johansen trace test
- `eigenvalue_statistic`: Eigenvalue test
- `n_cointegrating`: Number of cointegrating relationships
- `cointegrating_vectors`: The vectors

##### `stationarity_test(column, method='adf')`
Comprehensive stationarity testing.

**Parameters:**
- `method` (str): `'adf'`, `'kpss'`, `'pp'` (Phillips-Perron)

**Returns:** dict with:
- `statistic`: Test statistic
- `p_value`: Significance
- `is_stationary`: Boolean
- `critical_values`: Reference values

---

## Domain-Specific Analysis Module

**Location**: `data_toolkit.domain_specific`

### Class: DomainAnalysis

```python
from data_toolkit.domain_specific import DomainAnalysis

domain = DomainAnalysis(df)
```

### Environmental Science Methods

##### `mann_kendall_test(column, alpha=0.05)`
Mann-Kendall trend test (non-parametric).

**Returns:** dict with:
- `trend`: 'increasing', 'decreasing', or 'no trend'
- `p_value`: Significance
- `z_statistic`: Z-score
- `tau`: Kendall's tau

```python
result = domain.mann_kendall_test('temperature')
print(f"Trend: {result['trend']} (p={result['p_value']:.4f})")
```

##### `sens_slope(column)`
Sen's slope estimator (robust trend magnitude).

**Returns:** dict with:
- `slope`: Median slope
- `intercept`: Intercept
- `ci_lower`, `ci_upper`: Confidence interval

##### `standardized_precipitation_index(column, scale=3)`
SPI for drought monitoring.

##### `extreme_value_analysis(column, method='gev')`
Extreme value distribution fitting.

### Ecology Methods

##### `shannon_diversity(abundance_cols)`
Shannon diversity index (H').

**Returns:** dict with:
- `diversity_index`: H' value
- `evenness`: Pielou's evenness
- `richness`: Species count

```python
result = domain.shannon_diversity(['species_a', 'species_b', 'species_c'])
print(f"Shannon H' = {result['diversity_index']:.3f}")
```

##### `simpson_diversity(abundance_cols)`
Simpson's diversity index.

##### `morans_i(value_col, x_col, y_col, **kwargs)`
Moran's I spatial autocorrelation.

**Returns:** dict with:
- `I`: Moran's I statistic
- `expected_I`: Expected value under null
- `z_score`: Standardized statistic
- `p_value`: Significance

### Clinical/Biomedical Methods

##### `bland_altman(method1_col, method2_col, confidence_level=0.95)`
Bland-Altman method comparison analysis.

**Returns:** dict with:
- `mean_difference`: Bias
- `std_difference`: SD of differences
- `lower_loa`, `upper_loa`: Limits of agreement
- `ci_bias`: CI for bias
- `proportional_bias_test`: Test result

```python
result = domain.bland_altman('lab_measurement', 'portable_measurement')
print(f"Bias: {result['mean_difference']:.2f}")
print(f"LoA: [{result['lower_loa']:.2f}, {result['upper_loa']:.2f}]")
```

##### `intraclass_correlation(raters_cols, model='two-way', type='agreement')`
ICC for reliability studies.

**Parameters:**
- `model` (str): `'one-way'`, `'two-way'`
- `type` (str): `'agreement'`, `'consistency'`

**Returns:** dict with:
- `icc`: ICC value
- `ci_lower`, `ci_upper`: Confidence interval
- `f_value`, `p_value`: ANOVA statistics
- `interpretation`: Poor/Moderate/Good/Excellent

##### `cohens_kappa(rater1_col, rater2_col, weights=None)`
Cohen's Kappa inter-rater agreement.

**Returns:** dict with:
- `kappa`: Kappa statistic
- `std_error`: Standard error
- `ci_lower`, `ci_upper`: CI bounds
- `interpretation`: Slight/Fair/Moderate/Substantial/Almost Perfect

---

## Statistical Analysis Enhancements

New methods added to `StatisticalAnalysis` class in v4.0:

### `multiple_testing_correction(p_values, method='bonferroni', alpha=0.05)`

Corrects p-values for multiple comparisons.

**Parameters:**
- `p_values` (array): Array of p-values
- `method` (str): Correction method
  - `'bonferroni'`: Bonferroni correction
  - `'holm'`: Holm-Bonferroni (step-down)
  - `'fdr_bh'`: Benjamini-Hochberg FDR
  - `'fdr_by'`: Benjamini-Yekutieli FDR
- `alpha` (float): Significance level

**Returns:** dict with:
- `corrected_pvalues`: Adjusted p-values
- `reject`: Boolean array of rejections
- `method`: Method used

```python
from data_toolkit.statistical_analysis import StatisticalAnalysis

sa = StatisticalAnalysis(df)
result = sa.multiple_testing_correction([0.01, 0.04, 0.03, 0.08], method='fdr_bh')
print("Significant:", result['reject'])
```

### `variance_inflation_factor(feature_cols)`

Calculates VIF for multicollinearity detection.

**Returns:** dict with:
- `vif_values`: DataFrame with VIF for each feature
- `high_vif_features`: Features with VIF > 10
- `recommendation`: Guidance on collinearity

**Interpretation:**
| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation |
| 1-5 | Moderate correlation |
| 5-10 | High correlation |
| >10 | Severe multicollinearity |

### `robust_statistics(column, confidence_level=0.95)`

Calculates robust measures resistant to outliers.

**Returns:** dict with:
- `median`: Median value
- `mad`: Median Absolute Deviation
- `trimmed_mean`: 10% trimmed mean
- `winsorized_mean`: 10% Winsorized mean
- `huber_location`: Huber's M-estimator
- `ci_lower`, `ci_upper`: Robust CI

### `robust_regression(feature_cols, target_col, method='huber')`

Regression resistant to outliers.

**Parameters:**
- `method` (str): `'huber'`, `'ransac'`, `'theilsen'`

**Returns:** dict with:
- `coefficients`: Robust coefficients
- `intercept`: Intercept
- `r_squared`: R² score
- `inliers_mask`: For RANSAC, which points are inliers

---

## Installation

### Base Installation
```bash
pip install data-toolkit
```

### Optional Dependencies

```bash
# Survival analysis (lifelines)
pip install data-toolkit[survival]

# Interpretability (SHAP, LIME)
pip install data-toolkit[interpretability]

# Advanced time series (Prophet, ruptures, dtaidistance)
pip install data-toolkit[timeseries]

# Ecology (pymannkendall, esda, libpysal)
pip install data-toolkit[ecology]

# Reporting (jinja2)
pip install data-toolkit[reporting]

# All optional dependencies
pip install data-toolkit[all]
```

---

## Quick Examples

### Complete Analysis Workflow

```python
import pandas as pd
from data_toolkit import (
    DataQualityAnalyzer,
    FeatureSelector,
    ModelValidator,
    EffectSizeCalculator,
    ReportGenerator
)

# Load data
df = pd.read_csv('experiment_data.csv')

# 1. Check data quality
dqa = DataQualityAnalyzer(df)
quality = dqa.generate_quality_report()
print(f"Missing: {quality['missing_summary']}")

# 2. Impute if needed
if quality['has_missing']:
    result = dqa.impute_missing(quality['columns_with_missing'], method='knn')
    df = result['imputed_data']

# 3. Select features
selector = FeatureSelector(df)
features = selector.ensemble_selection(['f1', 'f2', 'f3', 'f4', 'f5'], 'target')
selected = features['consensus_features']

# 4. Train and validate model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

validator = ModelValidator(df)
cv_results = validator.cross_validate(model, selected, 'target', cv=5)
print(f"CV Accuracy: {cv_results['mean_scores']['accuracy']:.3f}")

# 5. Calculate effect size
calc = EffectSizeCalculator(df)
effect = calc.cohens_d('treatment_outcome', 'control_outcome')
print(f"Cohen's d: {effect['cohens_d']:.3f} ({effect['interpretation']})")

# 6. Generate report
report = ReportGenerator("Experiment Analysis")
report.add_data_provenance(df, "experiment_data.csv")
report.add_statistics_table(cv_results['mean_scores'], "Cross-Validation Results")
report.generate_html("analysis_report.html")
```

---

## See Also

- [USER_MANUAL.md](USER_MANUAL.md) — GUI usage guide
- [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) — When to use each analysis
- [TUTORIAL.md](TUTORIAL.md) — Step-by-step tutorials
- [TODO.md](../TODO.md) — Planned features
