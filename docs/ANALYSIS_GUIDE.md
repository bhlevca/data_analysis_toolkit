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

If you'd like, I will now implement a full Torrence & Compo CWT plotting helper (with COI and optional significance contours). After that I'll add unit tests that validate COI and period outputs.
