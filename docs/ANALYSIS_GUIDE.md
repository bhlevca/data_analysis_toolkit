# Advanced Data Analysis Toolkit — Feature Guide

This guide explains when to use the main analysis features, how to run them in the app, and how to interpret the results.

## Clustering

- When to use: use clustering when you want to discover natural groups in multivariate data (customer segments, behavior types, etc.). Prefer K-Means for spherical clusters, Hierarchical when you want a dendrogram, DBSCAN for arbitrary-shaped clusters and noise, and Gaussian Mixture Models for soft cluster assignments / probabilistic membership.
- How to use: select at least two numeric feature columns, choose the method from the Clustering tab, adjust parameters (n_clusters, eps, linkage) and click `Run Clustering`.
- What to expect: the app returns cluster labels, cluster sizes, and quality metrics such as Silhouette Score. K-Means and GMM produce cluster centers/means; DBSCAN returns a noise label (-1).
- Interpretation: 
  - Silhouette Score close to 1: well-separated clusters; near 0: overlapping; negative: likely incorrect clustering.
  - For DBSCAN, inspect the number of noise points to tune `eps` and `min_samples`.

# Advanced Data Analysis Toolkit — Feature Guide (Extended)

This extended guide explains when to use the main analysis features, how to run them in the app or via the Python API, and how to interpret the results. Each section includes short annotated examples you can run in a REPL.

## Clustering

- When to use: discover natural groups in multivariate data (customer segments, behavioral clusters, instrument modes).
- Recommended methods:
  - K-Means: fast, good for roughly spherical clusters of similar size.
  - Hierarchical (Agglomerative): produces a dendrogram helpful when you want nested clusters.
  - DBSCAN: density-based, finds arbitrarily shaped clusters and flags noise (useful when there is background noise).
  - Gaussian Mixture Model (GMM): probabilistic clustering, useful when clusters overlap and you want soft assignments.
- How to use (app): select numeric features, choose method and parameters (n_clusters / eps / linkage), click `Run Clustering`.
- How to use (API):

```python
from data_toolkit.ml_models import MLModels
ml = MLModels(df)
res = ml.kmeans_clustering(['feat1','feat2'], n_clusters=3)
print(res['cluster_sizes'])
print('Silhouette:', res.get('silhouette_score'))
```

- What you get:
  - `clusters`: array of labels for the rows used (rows with no NaN in selected features).
  - `cluster_sizes`: counts per cluster.
  - `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`: quality metrics.

- Interpretation (short):
  - Silhouette close to 1: clear separation; near 0: overlap; negative: bad clustering.
  - Davies-Bouldin: lower is better.
  - Calinski-Harabasz: higher is better.

## Anomaly Detection

- When to use: find rare or unexpected events in multivariate data (fraud, instrument failure, sensor glitches).
- Methods provided:
  - Isolation Forest: tree-based, good general-purpose detector.
  - Local Outlier Factor (LOF): local density comparisons, sensitive to local structure.
  - Minimum Covariance Determinant (MCD): robust multivariate distance-based detector.

- App usage: select features and contamination (expected fraction of anomalies), tune method parameters (n_estimators, n_neighbors), click `Detect Anomalies`.

```python
res = ml.isolation_forest_anomaly(['f1','f2','f3'], contamination=0.05)
print(res['n_anomalies'])
print('Indices:', res['anomaly_indices'][:10])
```

- What you get: labels (−1 anomaly, 1 normal), `anomaly_scores`, `anomaly_indices` (original DataFrame indices of flagged rows), and summary counts.
- Interpretation: inspect flagged rows. High magnitude scores or consistent flags across methods raise confidence.

## Fourier & Wavelets (FFT / PSD / CWT / DWT)

- When to use:
  - FFT/PSD: for stationary periodic signals (identify dominant frequencies).
  - CWT/DWT: for non-stationary signals or when you need time-frequency localization (transient bursts, changing oscillations).

- App usage: choose a single time series column, set sampling rate if known, choose analysis type and wavelet (for CWT), then run.

```python
from data_toolkit.timeseries_analysis import TimeSeriesAnalysis
ts = TimeSeriesAnalysis(df)
fft_res = ts.fourier_transform('column_name', sampling_rate=100.0)
print('Top frequencies:', fft_res['dominant_frequencies'])

cwt_res = ts.continuous_wavelet_transform('column_name', wavelet='morl')
print('Power shape:', cwt_res['power'].shape)
print('Periods (scales->periods):', cwt_res.get('periods')[:5])
print('COI length', len(cwt_res.get('coi', [])))
```

- Output fields:
  - FFT/PSD: `frequencies`, `magnitude`, `power`, `dominant_frequencies`.
  - CWT: `coefficients` (complex), `power` (abs(coeff)^2), `scales`, `periods` (scale→period mapping), `time`, `coi` (cone of influence array aligned to time).

- Interpreting CWT (Torrence & Compo style):
  - The 2D power map shows time × period (or frequency). High-power ridges indicate oscillatory components.
  - The COI line marks edge-effect region; regions outside COI (large periods near edges) should be interpreted cautiously.

## Example: Plotting CWT with COI (Torrence & Compo)

The app includes a Torrence & Compo-style plot: contourf of log(power), y-axis in periods, x-axis in time, with the cone of influence overlaid. This helps spot persistent vs transient oscillations.

## Tests

- The `tests/` folder contains small unit tests; run them with:

```bash
pytest tests/ -q
```

## Practical Tips

- Always inspect raw data for NaNs and scale. The app drops rows with NaNs in selected features — coordinates in results align to the cleaned rows unless explicitly returned as full-length indices (anomaly methods include aligned `anomaly_indices`).
- For clustering and distance-based methods, features should be on comparable scale; the toolkit scales internally but custom preprocessing is OK.

---

## New Analysis Features (v4.0)

### Effect Size Analysis

- **When to use**: After hypothesis testing to quantify the magnitude of differences or relationships. Effect sizes are essential for:
  - Meta-analysis compatibility
  - Power analysis for future studies
  - Practical significance interpretation
  - Journal publication requirements

- **Available effect sizes**:
  | Type | Use Case | Method |
  |------|----------|--------|
  | Cohen's d | Two-group means | `cohens_d()` |
  | Hedges' g | Small samples | `hedges_g()` |
  | Eta-squared | ANOVA | `eta_squared()` |
  | Cramér's V | Chi-square | `cramers_v()` |
  | Odds Ratio | 2×2 tables | `odds_ratio()` |

- **Interpretation (Cohen's d)**:
  - 0.2: Small effect
  - 0.5: Medium effect
  - 0.8: Large effect

```python
from data_toolkit.effect_sizes import EffectSizeCalculator
calc = EffectSizeCalculator(df)
result = calc.cohens_d('treatment', 'control')
```

### Model Validation

- **When to use**: Before trusting any machine learning model results. Critical for:
  - Avoiding overfitting
  - Unbiased hyperparameter selection
  - Publication-quality model evaluation

- **Key methods**:
  - `cross_validate()`: K-fold CV with multiple metrics
  - `nested_cross_validation()`: Unbiased performance during tuning
  - `learning_curve_analysis()`: Detect over/underfitting
  - `calibration_analysis()`: Probability calibration check
  - `residual_diagnostics()`: Comprehensive residual analysis

```python
from data_toolkit.model_validation import ModelValidator
validator = ModelValidator(df)
result = validator.nested_cross_validation(model, param_grid, features, target)
```

### Data Quality Analysis

- **When to use**: At the start of every analysis to understand data limitations:
  - Missing data assessment
  - Outlier detection
  - Distribution analysis
  - Imputation decisions

- **Missing data workflow**:
  1. `missing_data_summary()` — Quantify missingness
  2. `missing_pattern_analysis()` — Check patterns (MCAR/MAR/MNAR)
  3. `little_mcar_test()` — Test MCAR assumption
  4. `impute_missing()` or `multiple_imputation()` — Handle appropriately

```python
from data_toolkit.data_quality import DataQualityAnalyzer
dqa = DataQualityAnalyzer(df)
summary = dqa.missing_data_summary()
if summary['missing_pct'].max() > 5:
    result = dqa.multiple_imputation(cols_with_missing)
```

### Feature Selection

- **When to use**: Before model building to reduce dimensionality, improve interpretability, and avoid overfitting.

- **Method selection guide**:
  | Method | Best For |
  |--------|----------|
  | RFE | Linear models, interpretability |
  | Boruta | All-relevant feature discovery |
  | SHAP | Complex models, interaction detection |
  | Lasso | Sparse solutions, regularization |
  | Ensemble | Robust consensus selection |

```python
from data_toolkit.feature_selection import FeatureSelector
selector = FeatureSelector(df)
result = selector.ensemble_selection(features, target)
```

### Survival Analysis

- **When to use**: Time-to-event data with censoring:
  - Clinical trials (time to event/death)
  - Customer churn analysis
  - Equipment failure analysis
  - Subscription duration studies

- **Methods**:
  - `kaplan_meier()`: Non-parametric survival curves
  - `cox_regression()`: Hazard ratios for covariates
  - `log_rank_test()`: Compare survival curves
  - `parametric_survival()`: Weibull, exponential, etc.

```python
from data_toolkit.survival_analysis import SurvivalAnalyzer
surv = SurvivalAnalyzer(df)
result = surv.kaplan_meier('time', 'event', group_col='treatment')
```

### Model Interpretability

- **When to use**: To explain model predictions for:
  - Regulatory compliance
  - Scientific understanding
  - Debugging models
  - Stakeholder communication

- **Methods**:
  - `shap_analysis()`: Global feature importance
  - `lime_explain()`: Local instance explanations
  - `partial_dependence()`: Feature effect curves
  - `feature_interactions()`: Interaction detection

```python
from data_toolkit.interpretability import ModelInterpreter
interp = ModelInterpreter(model, df, features)
result = interp.shap_analysis()
```

### Advanced Time Series

- **When to use**: For sophisticated temporal analysis:
  - Forecasting with Prophet
  - Changepoint detection
  - Multivariate series (VAR)
  - Similarity measurement (DTW)

```python
from data_toolkit.advanced_timeseries import AdvancedTimeSeriesAnalysis
ats = AdvancedTimeSeriesAnalysis(df)
forecast = ats.prophet_forecast('date', 'value', periods=30)
changepoints = ats.detect_changepoints('value')
```

### Domain-Specific Analysis

- **Environmental Science**:
  - `mann_kendall_test()`: Non-parametric trend detection
  - `sens_slope()`: Robust trend magnitude
  - `standardized_precipitation_index()`: Drought analysis

- **Clinical Research**:
  - `bland_altman()`: Method comparison
  - `intraclass_correlation()`: Reliability studies
  - `cohens_kappa()`: Inter-rater agreement

- **Ecology**:
  - `shannon_diversity()`: Diversity indices
  - `simpson_diversity()`: Dominance measures
  - `morans_i()`: Spatial autocorrelation

```python
from data_toolkit.domain_specific import DomainAnalysis
domain = DomainAnalysis(df)
trend = domain.mann_kendall_test('temperature')
agreement = domain.bland_altman('method_a', 'method_b')
```

### Statistical Enhancements

- **Multiple testing correction**: Essential when performing many hypothesis tests
  - Bonferroni, Holm, FDR methods
  
- **Variance Inflation Factor (VIF)**: Detect multicollinearity
  - VIF > 10 indicates severe collinearity
  
- **Robust statistics**: Resistant to outliers
  - Median, MAD, trimmed mean, Huber M-estimator

```python
from data_toolkit.statistical_analysis import StatisticalAnalysis
sa = StatisticalAnalysis(df)
corrected = sa.multiple_testing_correction(p_values, method='fdr_bh')
vif = sa.variance_inflation_factor(features)
robust = sa.robust_statistics('column')
```

---

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).
