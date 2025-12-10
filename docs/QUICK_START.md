# 🚀 Quick Start Examples - Advanced Data Analysis Toolkit v2.0

## Installation & Setup

```python
# Install dependencies
pip install -r requirements.txt
pip install umap-learn  # Optional, for UMAP visualization

# Import modules
from src.data_toolkit.ml_models import MLModels
from src.data_toolkit.statistical_analysis import StatisticalAnalysis
from src.data_toolkit.timeseries_analysis import TimeSeriesAnalysis
from src.data_toolkit.comprehensive_tutorial import get_tutorial, get_short_tips
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
```

---

## 1. Machine Learning Examples

### K-Means Clustering
```python
ml = MLModels(df)

# Run K-Means clustering
results = ml.kmeans_clustering(
    features=['feature1', 'feature2', 'feature3'],
    n_clusters=3
)

print(f"Silhouette Score: {results['silhouette_score']:.3f}")
print(f"Cluster Sizes: {results['cluster_sizes']}")

# Visualize
fig = ml.plot_clustering_results(results)
fig.show()
```

### Anomaly Detection (Isolation Forest)
```python
# Detect anomalies
anomaly_results = ml.isolation_forest_anomaly(
    features=['feature1', 'feature2'],
    contamination=0.1  # Expect ~10% anomalies
)

print(f"Anomalies Found: {anomaly_results['n_anomalies']}")
print(f"Anomaly Indices: {anomaly_results['anomaly_indices'][:5]}")

# Visualize
fig = ml.plot_anomalies(anomaly_results)
fig.show()
```

### PCA Dimensionality Reduction
```python
# Reduce dimensions using PCA
pca_results = ml.pca_analysis(
    features=['feature1', 'feature2', 'feature3', 'feature4'],
    variance_threshold=0.95  # Keep 95% variance
)

print(f"Components Selected: {pca_results['n_components_selected']}")
print(f"Variance Explained: {pca_results['total_variance_explained']:.1%}")

# Plot
fig = ml.plot_pca_results(pca_results)
fig.show()
```

### t-SNE Visualization
```python
# Visualize high-dimensional data
tsne_results = ml.tsne_analysis(
    features=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
    perplexity=30
)

fig = ml.plot_dimensionality_reduction(tsne_results)
fig.show()
```

### Association Rules
```python
# Find association rules
rules = ml.apriori_rules(
    features=['feature1', 'feature2', 'feature3'],
    min_support=0.1,
    min_confidence=0.5,
    min_lift=1.5
)

print(f"Rules Found: {rules['n_rules']}")
for rule in rules['rules'][:5]:
    print(f"{rule['antecedent']} → {rule['consequent']}")
    print(f"  Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}\n")
```

---

## 2. Statistical Analysis Examples

### Comprehensive Descriptive Statistics
```python
stats = StatisticalAnalysis(df)

# Get detailed statistics including skewness, kurtosis
stats_df = stats.descriptive_stats(['feature1', 'feature2', 'feature3'])
print(stats_df)
```

### Distribution Fitting
```python
# Fit multiple probability distributions
fit_results = stats.fit_distributions('feature1')

print("Distribution Fit Results:")
for dist_name, params in fit_results.items():
    if 'error' not in params:
        print(f"{dist_name}: K-S Stat = {params.get('ks_stat', 'N/A'):.4f}")

# Visualize fitted distributions
fig = stats.plot_distribution_fit('feature1')
fig.show()
```

### Statistical Hypothesis Tests

#### Compare 2 Groups (t-test)
```python
# Independent t-test
results = stats.ttest_independent('group1_scores', 'group2_scores')

print(f"t-statistic: {results['statistic']:.4f}")
print(f"p-value: {results['p_value']:.4f}")
print(f"Significant: {results['significant']}")
print(f"Interpretation: {results['interpretation']}")
```

#### Compare 3+ Groups (ANOVA or Kruskal-Wallis)
```python
# One-Way ANOVA (parametric)
anova_results = stats.anova_oneway(['group1_data', 'group2_data', 'group3_data'])

print(f"F-statistic: {anova_results['statistic']:.4f}")
print(f"p-value: {anova_results['p_value']:.4f}")

# Kruskal-Wallis (non-parametric, if data not normal)
kw_results = stats.kruskal_wallis(['group1_data', 'group2_data', 'group3_data'])
print(f"H-statistic: {kw_results['statistic']:.4f}")
```

### Correlation Analysis
```python
# Comprehensive correlation analysis
corr_df = stats.pairwise_correlations(['feature1', 'feature2', 'feature3', 'feature4'])
print(corr_df)

# Visualize
fig = stats.plot_correlation_matrices(['feature1', 'feature2', 'feature3'])
fig.show()
```

### Outlier Detection
```python
# Detect outliers using IQR method
outliers = stats.outlier_detection(
    columns=['feature1', 'feature2'],
    method='iqr'
)

for col, result in outliers.items():
    print(f"{col}:")
    print(f"  Outliers: {result['n_outliers']} ({result['percentage']:.1f}%)")
    print(f"  Bounds: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")

# Visualize
fig = stats.plot_boxplots(['feature1', 'feature2', 'feature3'])
fig.show()
```

---

## 3. Time Series Analysis Examples

### Fourier Analysis (FFT)
```python
ts = TimeSeriesAnalysis(df)

# Compute FFT
fft_results = ts.fourier_transform('time_series_column', sampling_rate=1.0)

print("Dominant Frequencies:")
for i, freq in enumerate(fft_results['dominant_frequencies'][:5], 1):
    print(f"{i}. {freq:.4f} Hz")

# Visualize
fig = ts.plot_fft(fft_results, 'time_series_column', max_freq=10.0)
fig.show()
```

### Power Spectral Density (Welch's Method)
```python
# Smoother frequency estimate than FFT
psd_results = ts.power_spectral_density(
    'time_series_column',
    sampling_rate=1.0,
    window='hamming'
)

# Visualize
fig = ts.plot_power_spectral_density(psd_results, 'time_series_column')
fig.show()
```

### Wavelet Analysis

#### Continuous Wavelet Transform (CWT)
```python
# Time-frequency analysis
cwt_results = ts.continuous_wavelet_transform('time_series_column')

# Visualize
fig = ts.plot_wavelet_power(cwt_results, 'time_series_column')
fig.show()

# Interpretation: Darker colors = stronger signal at that time-frequency
```

#### Discrete Wavelet Transform (DWT)
```python
# Multi-resolution decomposition
dwt_results = ts.discrete_wavelet_transform(
    'time_series_column',
    wavelet='db4',
    level=3
)

# Visualize decomposition
fig = ts.plot_discrete_wavelet(dwt_results, 'time_series_column')
fig.show()
```

### Stationarity Testing
```python
# Check if time series is stationary
stationarity = ts.stationarity_test(['time_series_column'])

for col, result in stationarity.items():
    print(f"{col}:")
    print(f"  ADF p-value: {result['p_value']:.4f}")
    print(f"  Is Stationary: {result['is_stationary']}")
    if not result['is_stationary']:
        print(f"  → Apply differencing (d=1) for ARIMA")
```

### ARIMA Forecasting (requires stationary data)
```python
# Fit ARIMA model
arima_results = ts.arima_model('time_series_column', order=(1, 1, 1))

if 'error' not in arima_results:
    print(f"AIC: {arima_results['aic']:.2f}")
    print(f"BIC: {arima_results['bic']:.2f}")

    # Visualize fit
    fig = ts.plot_arima_fit(arima_results)
    fig.show()
```

---

## 4. Using the Comprehensive Tutorial System

```python
from src.data_toolkit.comprehensive_tutorial import get_tutorial, get_all_topics, get_short_tips

# Get all available topics
topics = get_all_topics()
print("Available Tutorials:", topics)

# Get detailed tutorial for a method
tutorial = get_tutorial("kmeans_clustering")
print(tutorial)

# Get quick tips
tips = get_short_tips()
print("Quick Tip for K-Means:", tips['kmeans'])
```

---

## 5. Complete Workflow Example

```python
import pandas as pd
from src.data_toolkit.ml_models import MLModels
from src.data_toolkit.statistical_analysis import StatisticalAnalysis

# Load data
df = pd.read_csv('customer_data.csv')

# Step 1: Exploratory Analysis
print("=== Step 1: Descriptive Statistics ===")
stats = StatisticalAnalysis(df)
stats_summary = stats.descriptive_stats(['age', 'income', 'score'])
print(stats_summary)

# Step 2: Check Distributions
print("\n=== Step 2: Distribution Analysis ===")
distributions = stats.fit_distributions('income')
print("Best fit distributions:", list(distributions.keys())[:3])

# Step 3: Statistical Tests
print("\n=== Step 3: Statistical Tests ===")
test_results = stats.ttest_independent('male_income', 'female_income')
print(f"Income difference significant: {test_results['significant']}")

# Step 4: Outlier Detection
print("\n=== Step 4: Outlier Detection ===")
outliers = stats.outlier_detection(['age', 'income', 'score'])
for col, result in outliers.items():
    print(f"{col}: {result['n_outliers']} outliers ({result['percentage']:.1f}%)")

# Step 5: Clustering
print("\n=== Step 5: Clustering (Customer Segmentation) ===")
ml = MLModels(df)
cluster_results = ml.kmeans_clustering(['age', 'income', 'score'], n_clusters=3)
print(f"Silhouette Score: {cluster_results['silhouette_score']:.3f}")
print(f"Cluster Sizes: {cluster_results['cluster_sizes']}")

# Step 6: Dimensionality Reduction for Visualization
print("\n=== Step 6: Visualization ===")
pca_results = ml.pca_analysis(['age', 'income', 'score', 'purchase_freq'])
print(f"Components needed for 95% variance: {pca_results['n_components_selected']}")

# Step 7: Anomaly Detection
print("\n=== Step 7: Anomaly Detection ===")
anomalies = ml.isolation_forest_anomaly(['age', 'income', 'score'])
print(f"Anomalies detected: {anomalies['n_anomalies']} ({anomalies['anomaly_percentage']:.1f}%)")

print("\n✅ Workflow Complete!")
```

---

## 6. Performance Tips

### For Large Datasets
```python
# Sample data for exploration
sample_df = df.sample(min(10000, len(df)))
stats = StatisticalAnalysis(sample_df)

# Full analysis on specific subset
full_stats = stats.descriptive_stats(['feature1', 'feature2'])
```

### For High-Dimensional Data
```python
# Use PCA first to reduce dimensionality
pca_results = ml.pca_analysis(all_features, variance_threshold=0.95)

# Then use reduced components for clustering/anomaly detection
reduced_features = pca_results['transformed_data']
```

### For Non-Normal Data
```python
# Check distribution
dist_results = stats.fit_distributions('feature1')

# Use non-parametric tests instead of t-test
results = stats.mann_whitney_u('group1', 'group2')
```

---

## 7. Common Workflows by Use Case

### Customer Segmentation
```python
# 1. Cluster customers
results = ml.kmeans_clustering(customer_features, n_clusters=4)

# 2. Analyze cluster characteristics
for cluster_id in range(4):
    cluster_data = df[results['clusters'] == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_data)} customers")
    print(cluster_data[customer_features].describe())
```

### Fraud Detection
```python
# 1. Detect anomalies
anomalies = ml.isolation_forest_anomaly(transaction_features, contamination=0.05)

# 2. Flag suspicious transactions
suspicious_idx = anomalies['anomaly_indices']
suspicious_transactions = df.loc[suspicious_idx]

# 3. Manual review
print(f"Review {len(suspicious_transactions)} suspicious transactions")
```

### Time Series Forecasting
```python
# 1. Check stationarity
stationarity = ts.stationarity_test(['price'])
if stationarity['price']['is_stationary']:
    d = 0
else:
    d = 1  # Need differencing

# 2. Fit ARIMA
arima = ts.arima_model('price', order=(1, d, 1))

# 3. Forecast (using fitted model parameters)
```

---

## 8. Troubleshooting

### "Feature scaling error"
→ All ML methods automatically scale features. No action needed.

### "Can't find wavelet module"
```python
# Install PyWavelets
pip install PyWavelets
```

### "UMAP not available"
```python
# Install optional dependency
pip install umap-learn
# Then use UMAP
results = ml.umap_analysis(features)
```

### "Memory error on large dataset"
```python
# Sample the data
df_sample = df.sample(frac=0.1)  # Use 10% for exploration
ml = MLModels(df_sample)
```

---

## 📖 Next Steps

1. **Explore Your Data**: Start with descriptive statistics and distributions
2. **Understand Relationships**: Use correlation and statistical tests
3. **Find Patterns**: Apply clustering and dimensionality reduction
4. **Detect Anomalies**: Use isolation forest or LOF
5. **Make Predictions**: Use ARIMA for time series or ML models for classification
6. **Validate Results**: Check interpretability and compare with domain knowledge

---

## 🆘 Getting Help

```python
# Get tutorial for any method
from src.data_toolkit.comprehensive_tutorial import get_tutorial

help_text = get_tutorial("pca_analysis")
print(help_text)

# Each tutorial includes:
# - What & Why
# - How to Use
# - Interpretation
# - Next Steps
```

---

*For more detailed documentation, see ENHANCEMENTS_v2.0.md and STREAMLIT_INTEGRATION_GUIDE.md*
