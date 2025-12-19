"""
Comprehensive Tutorial & Help System for Advanced Data Analysis Toolkit

This module contains detailed explanations, usage guides, interpretations,
and next steps for all analysis methods.

Version: 2.0
"""

# Dictionary of comprehensive tutorials
COMPREHENSIVE_TUTORIALS = {
    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    "image_recognition": """
Image Recognition (Streamlit Image tab)

Step-by-step (Streamlit Image tab)
1) Select data folder
   - Click Browse and choose the folder containing either class subfolders
     (data_folder/<class_name>/*.jpg) or a `labels.csv` file mapping filenames to labels.
   - To upload labels.csv from the UI, use the upload control; the file is saved to the
     selected folder automatically.

2) Preview and validate
   - Use the Sample preview to inspect a few images per class. Confirm file paths resolve
     and images are not corrupt or zero-byte.
   - Recommended split: train 80%, val 10%, test 10% (the app accepts an optional split column).

3) Configure training
   - Image size: choose 64, 96, 128, 160 or 224. Larger sizes increase compute and memory use.
   - Batch size: 16, 32, 64. Larger batches need more memory and may speed up training.
   - Epochs: 10-50 typical. Use early stopping on validation loss to avoid overfitting.
   - Model type: CNN (from scratch) or Transfer Learning (recommended for small datasets).
   - Augmentation: enable to reduce overfitting on small datasets.
   - Learning rate / optimizer: use sensible defaults; reduce LR if training is unstable.

4) Expectations and heuristics
   - Small datasets (< 500 images per class): transfer learning + augmentation gives best results.
   - Higher image_size (e.g., 224) improves accuracy but increases memory and time; prefer GPU for large sizes.
   - If classes are imbalanced, consider class weights or oversampling to improve per-class recall.
   - Training time roughly scales with (dataset_size * epochs) / batch_size; monitor GPU/CPU usage.

5) Training and saving
   - Enable "Save model automatically" to persist the trained model as a `.keras` file with metadata
     (including `class_names` where possible).
   - If not saved automatically, use the Save trained model button after training to export `.keras`.
   - Prefer `.keras` format; `.h5` is supported as fallback but can cause deserialization issues for some models.

6) Prediction
   - Upload a single image or choose from dataset examples and click Predict.
   - The UI shows predicted label and probabilities; if class names are missing, numeric labels are shown.

Troubleshooting
- ReadFile / missing images: verify `labels.csv` filenames point to existing images under the selected folder.
- Unknown layer / Lambda deserialization errors: re-save model in `.keras` format in an environment where custom layers are defined,
  or provide `custom_objects` when loading the model.
- HDF5/legacy issues: prefer `.keras` to reduce compatibility problems.
- Low accuracy / overfitting: add augmentation, reduce model capacity, or collect more data.

Tips
- Keep `labels.csv` paths relative to the data folder for portability.
- Start with transfer learning at image_size=128 and batch_size=32, then increase size/filters if resources allow.
- Use validation monitoring and early stopping to pick the best model automatically.
""",

    "descriptive_stats": """
## Descriptive Statistics

### What & Why
Descriptive statistics summarize key characteristics of your data.
Use them as the first step to understand your dataset.

### How to Use
1. Select numerical columns you want to analyze
2. The toolkit calculates: mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis
3. Look for:
   - **Skewness**: Data symmetry (-0.5 to 0.5 = fairly symmetric)
   - **Kurtosis**: Peak sharpness (0 = normal distribution)

### Interpretation
- **Mean > Median**: Right-skewed distribution (positive skew)
- **Mean < Median**: Left-skewed distribution (negative skew)
- **High variance**: Data is spread out
- **Low variance**: Data is concentrated

### What to Do Next
- If data is skewed → Consider transformation (log, sqrt, Box-Cox)
- If variance is high → Investigate outliers or subgroups
- If distribution is non-normal → Use non-parametric tests
""",

    "correlation_analysis": """
## 🔗 Correlation Analysis

### What & Why
Correlation measures the linear (Pearson), monotonic (Spearman), or ordinal (Kendall)
relationship between two variables.

### How to Use
1. Select 2+ numerical columns
2. Three correlation methods available:
   - **Pearson**: Linear relationships (-1 to 1)
   - **Spearman**: Monotonic (ranks-based)
   - **Kendall**: Ordinal association

### Interpretation
| Correlation | Strength |
|-------------|----------|
| 0.00 - 0.19 | Very weak |
| 0.20 - 0.39 | Weak |
| 0.40 - 0.59 | Moderate |
| 0.60 - 0.79 | Strong |
| 0.80 - 1.00 | Very strong |

### Important Notes
- **Correlation ≠ Causation**
- Check p-value: if p < 0.05 → statistically significant
- Use Spearman if relationship is non-linear but monotonic

### What to Do Next
- Significant correlation → Investigate causality with lag analysis or causal models
- Weak/no correlation → Variables are independent
- Very high correlation → Check for multicollinearity in regression
""",

    "distribution_analysis": """
## 📈 Distribution Analysis

### What & Why
Understanding data distribution helps choose the right statistical tests
and transformations.

### How to Use
1. Select columns to analyze
2. The toolkit fits: Normal, Gamma, Exponential, Lognormal, Weibull, Beta
3. Shows Shapiro-Wilk test results (p < 0.05 = not normally distributed)

### Interpretation
- **Q-Q Plot**: Points near diagonal = normal distribution
- **Histogram shape**:
  - Bell-shaped → Normal distribution
  - Right-skewed → Exponential or Gamma
  - Bimodal → Two subgroups present

### Distribution Characteristics
- **Normal**: Mean = Median, symmetric, used as baseline
- **Gamma**: Right-skewed, positive values only (good for wait times)
- **Exponential**: Sharp decay, modeling time until event
- **Lognormal**: Log-transformed normal (good for biological/physical data)
- **Weibull**: Flexible shape, modeling failure/survival

### What to Do Next
- Non-normal → Use non-parametric tests or transform data
- Identify best-fit distribution → Use for simulation/modeling
- Multiple modes → Investigate subgroups in data
""",

    "statistical_tests": """
## 🧪 Statistical Hypothesis Tests

### Choosing the Right Test

#### For Comparing 2 Groups:
| Condition | Use Test |
|-----------|----------|
| Normal data, equal variances | Independent t-test |
| Non-normal data | Mann-Whitney U test |
| Paired observations | Paired t-test |
| Paired, non-normal | Wilcoxon Signed-Rank |

#### For 3+ Groups:
| Condition | Use Test |
|-----------|----------|
| Normal data | One-Way ANOVA |
| Non-normal data | Kruskal-Wallis |

#### For Categorical Data:
- **Chi-Square**: Association between categorical variables

### Understanding p-values
- **p < 0.05**: Statistically significant (reject null hypothesis)
- **p ≥ 0.05**: Not significant (fail to reject null hypothesis)
- **Smaller p**: Stronger evidence against null hypothesis

### Understanding Effect Size
- Even significant results need practical importance
- Check mean differences, not just p-values

### What to Do Next
- Significant result → Report with effect size and confidence interval
- Non-significant → More data needed? Or variables truly unrelated?
- Large p-value → Variables are independent
""",

    "outlier_detection": """
## 🎯 Outlier Detection

### Methods

#### IQR (Interquartile Range) Method
- **Lower bound**: Q1 - 1.5 × IQR
- **Upper bound**: Q3 + 1.5 × IQR
- Values outside bounds are outliers
- **Good for**: Univariate data with symmetric distributions

#### Z-Score Method
- Points with |z-score| > 3 are outliers
- **Good for**: Normal distributions

### Causes of Outliers
1. **Data Entry Error**: Fix or remove
2. **Measurement Error**: Document and possibly remove
3. **Real Extreme Events**: Keep, investigate separately
4. **Natural Variation**: Keep for accurate analysis

### What to Do Next
- Check outlier values manually
- Investigate cause (error vs. real)
- Perform analysis with and without outliers
- Report both versions if different conclusions
""",

    # =========================================================================
    # MACHINE LEARNING - CLUSTERING
    # =========================================================================

    "kmeans_clustering": """
## 🎯 K-Means Clustering

### What & Why
Partitions data into K clusters minimizing within-cluster variance.
Use when you want to discover natural groups in unsupervised data.

### How to Use
1. Select features to use for clustering
2. Choose number of clusters (K) - start with 3-5
3. Review silhouette score (higher = better, -1 to 1 scale)
4. Examine cluster sizes and centers

### Interpretation
- **Silhouette Score**:
  - > 0.5: Good clustering
  - 0.5-0.25: Weak clustering
  - < 0.25: Poor clustering

- **Davies-Bouldin Index**: Lower is better (measure of cluster separation)

### Choosing K (Number of Clusters)
- **Elbow Method**: Look for "elbow" in inertia plot
- **Silhouette Analysis**: Maximum silhouette score
- **Domain Knowledge**: Use context to determine natural groups

### Limitations
- Assumes spherical clusters
- Requires standardization (automatic in toolkit)
- K must be specified in advance

### What to Do Next
- Analyze cluster characteristics (means, sizes)
- Use for customer segmentation, pattern discovery
- Try different K values if clusters aren't interpretable
- Compare with Hierarchical Clustering for confirmation
""",

    "hierarchical_clustering": """
## 🌳 Hierarchical Clustering (Agglomerative)

### What & Why
Creates a dendrogram showing how observations merge into clusters.
Better than K-Means when cluster structure is unknown.

### How to Use
1. Select features for clustering
2. Choose linkage method:
   - **Ward**: Minimizes variance (most common)
   - **Complete**: Maximum distance between clusters (tight)
   - **Average**: Mean distance between clusters
   - **Single**: Minimum distance (prone to chaining)
3. Cut dendrogram at desired height to get clusters

### Interpretation
- **Dendrogram height**: Distance between clusters
- **Tall dendrograms**: Distinct clusters
- **Flat dendrograms**: Gradual merging, no clear structure

### Advantages over K-Means
- Don't need to specify K in advance
- Hierarchical structure reveals cluster relationships
- Better with elongated or irregular clusters

### What to Do Next
- Review dendrogram visually
- Choose cutting height based on dendrogram structure
- Compare cluster assignments with domain knowledge
- Use for taxonomy/hierarchy discovery
""",

    "dbscan_clustering": """
## 📍 DBSCAN (Density-Based Clustering)

### What & Why
Groups points that are closely packed, marks outliers as noise.
Excellent for finding arbitrarily-shaped clusters.

### How to Use
1. Select features for clustering
2. Adjust parameters:
   - **eps**: Maximum distance between points in a neighborhood
   - **min_samples**: Minimum points in eps-neighborhood to form cluster
3. Number of clusters found automatically

### Interpretation
- **-1 label**: Noise/outlier points
- **Few clusters**: eps too large or min_samples too high
- **Many clusters**: eps too small or min_samples too low

### Key Advantages
- Finds non-spherical clusters
- Detects outliers automatically
- No need to specify number of clusters

### Parameter Tuning
- Use k-distance graph to find eps value
- Plot distance to k-nearest neighbor, look for "knee"

### What to Do Next
- Investigate noise points (outliers)
- Adjust eps for different granularity
- Use silhouette score to validate quality
""",

    "gaussian_mixture_model": """
## 🔵 Gaussian Mixture Model (GMM)

### What & Why
Probabilistic model assuming data from mixture of Gaussians.
Gives probability of membership rather than hard assignment.

### How to Use
1. Select features for clustering
2. Choose number of components (Gaussians)
3. Review probabilities for each point

### Interpretation
- **Probabilities**: Soft assignment to clusters (0-1)
- **BIC/AIC**: Information criteria (lower = better fit)
- **Weights**: Proportion of each component

### Advantages
- Probabilistic framework (get uncertainty)
- Better when clusters overlap
- Good for classification or clustering
- Handles multivariate normal distributions

### When to Use
- When cluster overlap is expected
- Need probability estimates
- Formal statistical framework required

### What to Do Next
- Use probabilities for weighted analysis
- Compare models with different K using BIC
- Good starting point for classification
""",

    # =========================================================================
    # DIMENSIONALITY REDUCTION
    # =========================================================================

    "pca_analysis": """
## 📉 Principal Component Analysis (PCA)

### What & Why
Linear transformation finding orthogonal directions of maximum variance.
Use to reduce dimensions while preserving information.

### How to Use
1. Select numerical features
2. Toolkit standardizes automatically
3. Review explained variance plot:
   - Shows variance explained by each component
   - Cumulative curve shows total variance preserved
4. Choose components for 90-95% variance threshold

### Interpretation
- **Explained Variance Ratio**: % variance explained by each PC
- **Scree Plot**: Shows drop-off point (elbow)
- **Cumulative Variance**: Should reach plateau

### Example Reading
- PC1 explains 45% variance
- PC2 explains 20% variance
- PC3 explains 15% variance
- PC1+PC2+PC3 = 80% total variance

### Applications
- **Visualization**: Reduce to 2-3D for plotting
- **Denoising**: Keep main components, drop noise
- **Feature Extraction**: Use PCs as new features
- **Multicollinearity**: Solve using PC regression

### Limitations
- Only captures linear relationships
- Hard to interpret components
- Sensitive to scaling (automatic in toolkit)

### What to Do Next
- Use PC components as features for ML models
- Visualize 2D PCA plot for cluster identification
- Compare with t-SNE/UMAP for non-linear relationships
""",

    "tsne_analysis": """
## 🎨 t-Distributed Stochastic Neighbor Embedding (t-SNE)

### What & Why
Non-linear dimensionality reduction excellent for visualization.
Preserves local structure - similar points stay close.

### How to Use
1. Select features (works with many features)
2. Adjust perplexity (usually 5-50, default ~30)
3. Output is 2D or 3D for visualization
4. Color points by class/cluster to see separability

### Interpretation
- **Clusters in t-SNE**: Local groupings of similar points
- **Distances**: NOT meaningful (only local structure matters)
- **Separation**: Clear clusters = classes well-defined

### Perplexity Parameter
- **Too low (< 5)**: Only local structure, fragmented
- **Too high (> dataset size/5)**: Global structure, loses details
- **Good range**: 5-50 for most datasets

### Important Notes
- Non-deterministic (different runs give different plots)
- **Cannot transform new points** (rerun t-SNE)
- Use PCA first for very high-dimensional data

### Compared to PCA
| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Speed | Fast | Slow |
| Captures | Linear patterns | Non-linear patterns |
| Distances | Meaningful | Local only |
| New data | Can transform | Must rerun |

### What to Do Next
- Use for exploratory visualization
- Identify clusters visually before formal analysis
- Try different perplexity values
- Compare with UMAP for faster alternative
""",

    "umap_analysis": """
## 🌐 UMAP (Uniform Manifold Approximation and Projection)

### What & Why
Modern alternative to t-SNE: faster, preserves more global structure.
Excellent for large datasets and interactive visualization.

### How to Use
1. Select features for analysis
2. Adjust parameters:
   - **n_neighbors**: Balance local vs global (default 15)
   - **min_dist**: Minimum distance in embedding (default 0.1)
3. Output typically 2D for visualization

### Interpretation
- **Local structure**: Preserved like t-SNE
- **Global structure**: Better preserved than t-SNE
- **Cluster separation**: Clear separation = well-defined groups

### Parameter Tuning
- **More local**: Lower n_neighbors (5-10)
- **More global**: Higher n_neighbors (50+)
- **Tighter layout**: Lower min_dist
- **Looser layout**: Higher min_dist

### Advantages over t-SNE
- 10-100x faster
- Better global structure preservation
- Can transform new data
- More reproducible (minor variations)

### When to Use
- Very large datasets (>10,000 points)
- Need faster computation
- Want global structure and local detail
- Need to transform new data

### What to Do Next
- Experiment with n_neighbors for best visualization
- Use for interactive exploration
- Compare cluster separation with other methods
""",

    "anomaly_detection": """
## 🚨 Anomaly Detection

### What & Why
Identifies unusual points that deviate from normal patterns.
Critical for fraud detection, quality control, medical diagnosis.

### Methods

#### Isolation Forest
- **How**: Isolates anomalies using random partitions
- **Best for**: Multivariate outliers, irregular anomalies
- **Speed**: Very fast, scales well
- **Advantages**: Works without assuming distribution

#### Local Outlier Factor (LOF)
- **How**: Measures local density deviation
- **Best for**: Local anomalies in dense regions
- **Advantages**: Finds contextual outliers
- **Limitation**: Slower on large datasets

#### Minimum Covariance Determinant (MCD)
- **How**: Robust estimation of data distribution
- **Best for**: Scientific data, proteomics
- **Advantages**: Statistical foundation, handles multivariate data
- **Limitation**: Assumes roughly Gaussian data

### Interpretation
- **Anomaly Score**: Lower = more anomalous (Isolation Forest)
- **Contamination**: Expected proportion of anomalies (0.01-0.1)
- **Anomaly Indices**: Which rows are anomalous

### Causes & Actions
| Cause | Action |
|-------|--------|
| Data entry error | Fix or remove |
| Measurement error | Remove or flag |
| Real anomaly | Investigate, may be valuable |
| Rare event | Keep for analysis |

### What to Do Next
- Investigate top anomalies manually
- Compare methods (often give different results)
- Use contamination parameter based on domain knowledge
- Combine with visualization (PCA, t-SNE)
""",

    # =========================================================================
    # TIME SERIES ANALYSIS
    # =========================================================================

    "fourier_analysis": """
## 🔊 Signal Analysis — Fourier & Wavelet (FFT & Wavelet)

### What & Why
Decomposes time series into frequency components.
Reveals dominant frequencies and periodicities.

### How to Use
1. Select time series column
2. Fourier Transform computed automatically
3. Review magnitude and power spectra
4. Identify peaks (dominant frequencies)

### Interpretation
- **Dominant Frequency**: Most important periodic component
- **Power Spectrum**: How much variance at each frequency
- **Magnitude Spectrum**: Amplitude of each frequency component

### What Frequencies Tell You
- **Low frequency**: Long-term trends
- **High frequency**: Short-term noise or rapid oscillations
- **Peaks**: Dominant periodicities in data

### FFT vs Power Spectral Density
| Metric | FFT | Welch PSD |
|--------|-----|-----------|
| Method | Direct | Averaged |
| Noise | High | Reduced |
| Detail | Maximum | Smoothed |

### Applications
- **Periodicity Detection**: Find cycles in data
- **Filtering**: Remove unwanted frequencies
- **Compression**: Keep important frequencies
- **Signal Characterization**: Understand frequency content

### What to Do Next
- Look for unexpected periodicities
- Use PSD (Welch) for cleaner estimate
- Compare with wavelet analysis for time-varying frequencies
- Use dominant frequencies for forecasting
""",

    "wavelet_analysis": """
## 🌊 Wavelet Analysis (part of Signal Analysis)

### What & Why
Analyzes time-varying frequencies in non-stationary signals.
Better than Fourier for signals with changing characteristics.

### How to Use

#### Continuous Wavelet Transform (CWT)
1. Select time series
2. Specify scales or use defaults
3. Results show time-frequency representation
4. Darker colors = higher power at that time-frequency

#### Discrete Wavelet Transform (DWT)
1. Select time series
2. Choose wavelet family (db4, sym5, etc.)
3. Choose decomposition level
4. Shows approximation and detail coefficients

### Interpretation
- **CWT heatmap**: Vertical axis = scale (inverse frequency)
- **High values**: Strong signal component at that time-frequency
- **DWT coefficients**: Approximation (trends), Details (fluctuations)

### Wavelet vs Fourier
| Aspect | Fourier | Wavelet |
|--------|---------|---------|
| Time info | No | Yes |
| Non-stationary | Poor | Excellent |
| Localization | Global | Local |
| Detail | Constant | Variable |

### Use Cases
- **Fault Detection**: Find when frequencies change
- **Feature Extraction**: Use wavelet coefficients as features
- **Denoising**: Remove high-frequency noise via DWT
- **Compression**: Wavelet compression algorithms
- **Non-stationary Analysis**: Variable frequency content

### Wavelet Families
- **Daubechies (db)**: Compact support, general purpose
- **Symlets (sym)**: More symmetric than Daubechies
- **Coifs (coif)**: Even more symmetry
- **Morlet**: Optimal time-frequency resolution

### What to Do Next
- Use CWT for visual exploration of time-frequency patterns
- Use DWT for feature extraction and denoising
- Combine with classical time series methods
""",

    "stationarity_testing": """
## 🔄 Stationarity Testing

### What & Why
Stationary series have constant mean/variance over time.
Required for many time series models (ARIMA, etc.).

### Augmented Dickey-Fuller (ADF) Test
- **Null Hypothesis**: Series has unit root (non-stationary)
- **p-value < 0.05**: Reject null → stationary ✓
- **p-value ≥ 0.05**: Fail to reject → non-stationary ✗

### Visual Indicators of Non-stationarity
- Trend up/down over time
- Changing mean level
- Increasing or changing variance
- Seasonal patterns

### Making Non-stationary Data Stationary
| Method | When | How |
|--------|------|-----|
| Differencing | Linear trend | Take y_t - y_{t-1} |
| Seasonal Diff | Seasonal pattern | Take y_t - y_{t-period} |
| Detrending | Trend | Remove trend line |
| Log Transform | Variance increases | Take log(y_t) |

### Why Stationarity Matters
- Many statistical tests assume stationarity
- ARIMA models work on stationary data
- Helps find true relationships, not spurious ones

### What to Do Next
- Test all time series for stationarity first
- If non-stationary, apply transformations
- Re-test until stationary
- Then apply ARIMA or other models
""",

    "arima_modeling": """
## 📊 ARIMA Modeling

### What & Why
Autoregressive Integrated Moving Average - powerful forecasting model.
Works for stationary (or made-stationary) time series.

### Understanding ARIMA(p,d,q)
- **p (AR)**: Autoregressive order (past values)
- **d (I)**: Integration order (differencing needed)
- **q (MA)**: Moving average order (past errors)

### Determining Parameters
1. **Check stationarity**: Non-stationary? d ≥ 1
2. **ACF/PACF plots**:
   - ACF tails off, PACF cuts off → AR(p)
   - ACF cuts off, PACF tails off → MA(q)
   - Both tail off → ARMA(p,q)
3. **Information Criteria**: Use AIC or BIC to compare

### Example Interpretations
- **ARIMA(1,0,0)**: AR(1) on stationary data
- **ARIMA(0,1,1)**: Differenced data with MA(1) component
- **ARIMA(1,1,1)**: Common choice for non-stationary data

### Reading Results
- **AIC/BIC**: Lower is better (trade model fit vs complexity)
- **Residuals**: Should be white noise (no patterns)
- **Forecasts**: With confidence intervals

### Limitations
- Assumes linear relationships
- Needs stationarity or differencing
- Univariate only (use ARIMAX for exogenous variables)
- Not ideal for multiple seasonality

### What to Do Next
- Test residuals for white noise
- If residuals show patterns, try different p,d,q
- Use for short-term forecasting (1-5 steps ahead)
- Consider SARIMA for seasonal data
""",

    # =========================================================================
    # ASSOCIATION RULES & CAUSALITY
    # =========================================================================

    "association_rules": """
## 🔗 Association Rule Learning (Apriori)

### What & Why
Discovers "if-then" rules in data patterns.
Used in market basket analysis, medical diagnosis, recommendation systems.

### Understanding Metrics

#### Support
- Proportion of transactions with both items
- **Higher = more common association**
- Typical threshold: 0.1-0.3

#### Confidence
- P(Consequent | Antecedent)
- "If customer buys A, what's probability they buy B?"
- **Higher = stronger rule**
- Typical threshold: 0.5-0.8

#### Lift
- Confidence / Baseline probability
- **Lift > 1**: Positive association
- **Lift = 1**: Items independent
- **Lift < 1**: Negative association
- **Higher = stronger effect**

### Interpretation Example
**Rule: {bread} → {milk}**
- Support: 0.15 (15% of transactions have both)
- Confidence: 0.75 (75% who buy bread also buy milk)
- Lift: 2.5 (2.5x more likely to buy milk when buying bread)

### Applications
- **Marketing**: Recommend products together
- **Medical**: Find symptom-disease associations
- **Web**: Predict what users click next
- **Inventory**: Plan cross-selling strategies

### Challenges
- Many rules generated (need high thresholds)
- Can find spurious associations (correlation ≠ causation)
- Needs categorical or binned data

### What to Do Next
- Filter by lift (practical importance)
- Investigate top rules manually
- Test rules on new data
- Consider temporal aspects
""",

    "causality_analysis": """
## 🔬 Causality Analysis

### What & Why
Determines if X causes Y (not just correlated).
Critical for decision-making and policy evaluation.

### Granger Causality
- **Tests**: Does past X help predict Y?
- **p < 0.05**: X "Granger-causes" Y
- **Important**: This is predictive causality, not true causation!

### Lead-Lag Analysis
- Shows at what time delay correlation is strongest
- **Negative lag**: X leads Y (X comes first)
- **Positive lag**: Y leads X (Y comes first)

### Three Criteria for Causation
1. **Temporal precedence**: Cause before effect
2. **Covariation**: Cause and effect related
3. **No confounding**: No third variable explains both

### Why Correlation ≠ Causation
**Examples of confounding:**
- Ice cream sales & drowning (temperature confounds both)
- Shoe size & reading ability (age confounds both)
- Yellow fingers & lung cancer (smoking confounds both)

### Methods to Test Causality
| Method | Use When |
|--------|----------|
| Granger Causality | Time series data |
| Instrumental Variables | Have good instrument |
| Difference-in-Differences | Policy/intervention |
| Causal Forests | Machine learning approach |
| RCT | Can do experiment |

### What to Do Next
- Show temporal precedence
- Check for confounding variables
- Validate with domain expertise
- Consider alternative explanations
- Design study to test causality
""",

    # =========================================================================
    # EXTENDED ANOVA & POST-HOC TESTS (NEW)
    # =========================================================================

    "anova_extended": """
## 🧪 Extended ANOVA & Post-Hoc Tests

### What & Why
ANOVA extensions for complex experimental designs with multiple factors,
repeated measurements, and follow-up pairwise comparisons.

---

### Two-Way Factorial ANOVA

**Purpose:** Test main effects of two factors AND their interaction.

**Model:** Y = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + ε

| Effect | Question |
|--------|----------|
| **Main Effect A** | Does factor A affect outcome (ignoring B)? |
| **Main Effect B** | Does factor B affect outcome (ignoring A)? |
| **Interaction A×B** | Does effect of A depend on level of B? |

**Interpretation:**
- Significant main effect: Factor influences outcome overall
- Significant interaction: Effects are not additive (interpret main effects cautiously)
- Always examine interaction first before interpreting main effects

**Example:** Studying plant growth with Fertilizer (A: Low/High) and Water (B: Low/High)
- Main effect of Fertilizer: High fertilizer increases growth
- Main effect of Water: More water increases growth  
- Interaction: Extra fertilizer helps more when water is abundant

---

### Repeated-Measures ANOVA

**Purpose:** Compare conditions when same subjects are measured multiple times.

**Advantages over Independent ANOVA:**
- Controls for individual differences
- More statistical power
- Requires fewer subjects

**Assumptions:**
1. **Sphericity**: Equal variances of differences between conditions
   - Test with Mauchly's test
   - If violated (p < 0.05): Use Greenhouse-Geisser or Huynh-Feldt correction

**Effect Size (Partial η²):**
| Value | Interpretation |
|-------|----------------|
| 0.01 | Small |
| 0.06 | Medium |
| 0.14+ | Large |

---

### Post-Hoc Tests

**When to use:** After significant omnibus ANOVA to determine which specific groups differ.

#### Tukey's Honest Significant Difference (HSD)
- Controls family-wise error rate
- All pairwise comparisons
- Best for equal sample sizes
- Reports: Mean difference, SE, Q-statistic, adjusted p-value

#### Bonferroni Correction
- Most conservative approach
- Divides α by number of comparisons
- Good for few planned comparisons
- Reports: t-statistic, raw p-value, adjusted p-value, Cohen's d

**Effect Size (Cohen's d):**
| Value | Interpretation |
|-------|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8+ | Large |

---

### How to Use in the Toolkit

1. **Two-Way ANOVA:**
   - Select continuous dependent variable
   - Select two categorical factors
   - Review main effects and interaction in results table

2. **Repeated-Measures ANOVA:**
   - Select dependent variable (measurement)
   - Select subject/ID column
   - Select within-subjects factor (conditions)
   - Check sphericity warning

3. **Post-Hoc Tests:**
   - Select dependent variable
   - Select grouping variable
   - Choose Tukey HSD or Bonferroni
   - Review pairwise comparison table

### What to Do Next
- If interaction is significant: Plot interaction and interpret conditionally
- If main effects only: Report effect sizes and confidence intervals
- Document which comparisons were planned vs. exploratory
""",

    # =========================================================================
    # PROBABILITY DISTRIBUTIONS (NEW)
    # =========================================================================

    "probability_distributions": """
## 📊 Probability Distribution Analysis

### What & Why
Fit theoretical probability distributions to empirical data for:
- Understanding data-generating processes
- Simulation and Monte Carlo methods
- Risk assessment and reliability analysis
- Parametric statistical inference

---

### Extended Distribution Fitting

The toolkit fits 12+ distributions and ranks them by goodness-of-fit:

| Distribution | Use Case |
|-------------|----------|
| **Normal** | Symmetric, bell-shaped data |
| **t-distribution** | Symmetric with heavier tails than normal |
| **Gamma** | Positive, right-skewed data (waiting times) |
| **Exponential** | Memoryless processes (time between events) |
| **Lognormal** | Positive, right-skewed (multiplicative processes) |
| **Weibull** | Reliability analysis, survival times |
| **Laplace** | Double-exponential, heavier tails |
| **Logistic** | Similar to normal but heavier tails |
| **Pareto** | Power-law distributions (80/20 rule) |
| **Cauchy** | Very heavy tails, no mean exists |
| **Chi-squared** | Sum of squared normals |
| **Beta** | Bounded [0,1] data (proportions) |

---

### Model Selection Criteria

#### AIC (Akaike Information Criterion)
- AIC = 2k - 2ln(L)
- k = number of parameters, L = likelihood
- **Lower is better**
- Penalizes complexity to avoid overfitting

#### BIC (Bayesian Information Criterion)  
- BIC = k·ln(n) - 2ln(L)
- n = sample size
- **Lower is better**
- Stricter penalty than AIC for large samples

#### Kolmogorov-Smirnov Test
- Tests if data comes from specified distribution
- **p > 0.05**: Cannot reject that data follows distribution
- **p < 0.05**: Data significantly differs from distribution

---

### Random Variable Analysis

For each fitted distribution, compute:

**Moments:**
- Mean (expected value)
- Variance (spread)
- Skewness (asymmetry)
- Kurtosis (tail heaviness)

**Quantiles:**
- Median (50th percentile)
- Quartiles (25th, 75th)
- Extreme percentiles (1st, 99th)

**Confidence Intervals:**
- Parameter uncertainty from MLE
- Bootstrap confidence intervals

---

### QQ (Quantile-Quantile) Plots

**Interpretation:**
- Points on diagonal line = good fit
- Curved pattern = wrong distribution family
- S-shape = different skewness/kurtosis
- Outliers = extreme values deviating from line

**Common Patterns:**
| Pattern | Meaning |
|---------|---------|
| Light right tail | Points below line at right |
| Heavy right tail | Points above line at right |
| Short tails | S-curve bowing inward |
| Long tails | S-curve bowing outward |

---

### How to Use

1. Load data and select numeric column
2. Go to Statistics → Probability Distributions
3. Select distributions to fit (or use "All")
4. Review AIC/BIC ranking table
5. Check QQ plots for top candidates
6. Use best-fit distribution for inference

### What to Do Next
- Use fitted distribution for simulation
- Calculate probability of extreme events
- Perform parametric hypothesis tests
- Report uncertainty in distribution parameters
""",

    # =========================================================================
    # EXTENDED SPECTRAL ANALYSIS (NEW)
    # =========================================================================

    "spectral_extended": """
## 🔊 Extended Spectral Analysis

### What & Why
Advanced frequency-domain analysis for paired signals and non-stationary data.
Goes beyond standard FFT to analyze relationships between signals.

---

### Coherence Analysis

**Purpose:** Measure linear correlation between two signals as function of frequency.

**Formula:** Cxy(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))

| Value | Interpretation |
|-------|----------------|
| 0 | No linear relationship at frequency f |
| 0.5 | Moderate coherence |
| 1 | Perfect linear relationship |

**Use Cases:**
- EEG channel synchronization
- Mechanical vibration analysis
- Climate teleconnections
- Audio signal processing

**Parameters:**
- **nperseg**: Segment length (longer = better frequency resolution)
- **noverlap**: Segment overlap (typically 50%)

---

### Cross-Wavelet Transform (XWT)

**Purpose:** Shows common power in time-frequency space between two signals.

**Advantages over Coherence:**
- Reveals time-varying relationships
- Shows phase relationships
- Handles non-stationary signals

**Interpretation:**
- High power = both signals have strong oscillations at that time-frequency
- Phase arrows show lead/lag relationship
- Cone of influence marks edge effects

**Parameters:**
- **scales**: Range of wavelet scales (auto-selected based on sampling)
- **wavelet**: Mother wavelet (Morlet is default)

---

### Wavelet Coherence (WTC)

**Purpose:** Localized coherence varying in both time and frequency.

**Advantages:**
- Detects intermittent coupling
- Shows when/where signals are correlated
- Handles non-stationary relationships

**Statistical Significance:**
- Compare to surrogate distribution
- Monte Carlo significance testing
- 95% confidence contours

**Interpretation Guide:**
| Color | Meaning |
|-------|---------|
| Red/Yellow | High coherence (signals coupled) |
| Blue/Green | Low coherence (signals independent) |
| Arrows → | In-phase relationship |
| Arrows ← | Anti-phase relationship |
| Arrows ↓ | Signal 1 leads by 90° |
| Arrows ↑ | Signal 2 leads by 90° |

---

### Harmonic Analysis

**Purpose:** Extract sinusoidal components using least-squares fitting.

**Method:** Fits model: y(t) = Σ [Aᵢ·cos(2πfᵢt) + Bᵢ·sin(2πfᵢt)]

**Output:**
- Dominant frequencies
- Amplitudes (Rᵢ = √(Aᵢ² + Bᵢ²))
- Phases (φᵢ = arctan(Bᵢ/Aᵢ))
- Explained variance per component

**Use Cases:**
- Tidal analysis
- Seasonal decomposition
- Removing known periodic signals
- Detecting hidden periodicities

---

### Choosing the Right Method

| Question | Method |
|----------|--------|
| Frequency relationship (stationary)? | Coherence |
| Time-frequency relationship? | XWT or WTC |
| Localized coherence? | WTC |
| Extract dominant periods? | Harmonic Analysis |
| Quick power spectrum? | Standard FFT |

### How to Use

1. Go to Signal Processing → Extended Spectral
2. Select two signal columns
3. Choose analysis type
4. Adjust parameters if needed
5. Interpret plots and metrics

### What to Do Next
- Compare coherence at different frequency bands
- Identify time periods of high coherence
- Extract phase relationships for lead-lag analysis
- Use significant frequencies in models
""",

    # =========================================================================
    # ARIMA/SARIMA FORECASTING (NEW)
    # =========================================================================

    "arima_sarima": """
## ⏱️ ARIMA & SARIMA Forecasting

### What & Why
Box-Jenkins methodology for time series forecasting.
ARIMA for non-seasonal, SARIMA for seasonal patterns.

---

### ARIMA(p, d, q) Components

| Parameter | Meaning | Identification |
|-----------|---------|----------------|
| **p** | Autoregressive order | PACF cutoff |
| **d** | Differencing order | ADF test (0 if stationary) |
| **q** | Moving average order | ACF cutoff |

**AR(p):** Current value depends on p past values
- Yₜ = φ₁Yₜ₋₁ + φ₂Yₜ₋₂ + ... + φₚYₜ₋ₚ + εₜ

**I(d):** Differencing to achieve stationarity
- d=1: ΔYₜ = Yₜ - Yₜ₋₁
- d=2: Δ²Yₜ = ΔYₜ - ΔYₜ₋₁

**MA(q):** Current value depends on q past errors
- Yₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q

---

### SARIMA(p,d,q)(P,D,Q,m)

Extends ARIMA with seasonal components:

| Parameter | Meaning | Example |
|-----------|---------|---------|
| **P** | Seasonal AR order | Relationship with same season last year |
| **D** | Seasonal differencing | Remove yearly pattern |
| **Q** | Seasonal MA order | Seasonal error structure |
| **m** | Seasonal period | 12 for monthly, 4 for quarterly |

**Common Seasonal Periods:**
- Monthly data: m=12
- Quarterly data: m=4
- Weekly data: m=52 (or 7 for daily)
- Hourly data: m=24

---

### Auto-ARIMA

**Purpose:** Automatic parameter selection via grid search.

**Algorithm:**
1. Test stationarity (ADF test)
2. Determine d (differencing order)
3. Search grid of (p, q) values
4. Select model with lowest AIC/BIC
5. Validate residuals

**Parameters to Set:**
- max_p, max_q: Maximum orders to try
- seasonal: Enable SARIMA search
- m: Seasonal period (if seasonal)

---

### Model Diagnostics

**Residual Checks (should pass all):**

| Test | Good Result | Problem If Failed |
|------|-------------|-------------------|
| **Ljung-Box** | p > 0.05 | Autocorrelation remains |
| **Normality** | p > 0.05 | Non-normal errors |
| **Heteroscedasticity** | p > 0.05 | Variance changes over time |

**Residual Plots:**
- ACF of residuals: No significant spikes
- Histogram: Approximately normal
- Q-Q plot: Points on diagonal

---

### Forecasting

**Point Forecasts:**
- Expected value at each future time point
- Based on fitted model parameters

**Confidence Intervals:**
- 95% CI: True value expected here 95% of time
- Widens as forecast horizon increases
- Assumes model is correctly specified

**Forecast Horizons:**
- Short-term (1-3 periods): Most accurate
- Medium-term (3-12 periods): Moderate accuracy
- Long-term (12+ periods): Use with caution

---

### Common ARIMA Models

| Model | Use Case |
|-------|----------|
| ARIMA(1,1,1) | Simple baseline for most series |
| ARIMA(1,1,0) | AR(1) with differencing |
| ARIMA(0,1,1) | MA(1) with differencing (exponential smoothing) |
| ARIMA(2,1,2) | More complex dynamics |
| SARIMA(1,1,1)(1,1,1,12) | Monthly seasonal data |

### How to Use

1. Go to Time Series → ARIMA/SARIMA
2. Select time series column
3. Enable Auto-ARIMA or set parameters manually
4. Set forecast horizon
5. Review fit statistics and residual diagnostics
6. Generate forecasts with confidence intervals

### What to Do Next
- Compare AIC/BIC across model specifications
- Validate on holdout test set
- Check residuals for remaining patterns
- Consider external regressors (ARIMAX) if predictors available
""",

    # =========================================================================
    # MULTIVARIATE TIME SERIES (NEW)
    # =========================================================================

    "multivariate_ts": """
## 📈 Multivariate Time Series Analysis

### What & Why
Analyze multiple time series jointly to capture interdependencies,
test causality, and forecast related variables together.

---

### VAR (Vector Autoregression)

**Purpose:** Model each variable as function of lagged values of ALL variables.

**Model:** Yₜ = A₁Yₜ₋₁ + A₂Yₜ₋₂ + ... + AₚYₜ₋ₚ + εₜ

Where Y is a vector of all variables and A are coefficient matrices.

**Key Features:**
- Captures feedback between variables
- Granger causality tests built-in
- Impulse response functions
- Variance decomposition

**Lag Selection:**
| Criterion | Interpretation |
|-----------|----------------|
| AIC | Balance fit and complexity |
| BIC | More parsimonious (fewer lags) |
| HQIC | Between AIC and BIC |

**Granger Causality:**
- Tests if past X improves prediction of Y
- F-test for joint significance of lags
- p < 0.05: X "Granger-causes" Y

---

### VECM (Vector Error Correction Model)

**Purpose:** VAR for cointegrated (non-stationary but related) series.

**When to Use:**
- Series are non-stationary (fail ADF)
- Series share long-run equilibrium (cointegration)
- Want to model both short-run and long-run dynamics

**Johansen Cointegration Test:**
| Result | Interpretation |
|--------|----------------|
| r = 0 | No cointegration → Use VAR on differences |
| r = 1 | One cointegrating relationship |
| r = k | Full rank (all series are stationary) |

**Error Correction Term:**
- Speed of adjustment to equilibrium
- Should be negative and significant
- |α| closer to 1 = faster adjustment

**Example:** Stock prices and dividends
- Both non-stationary
- Share long-run relationship (cointegrated)
- VECM captures short-run deviations and adjustment

---

### DTW (Dynamic Time Warping)

**Purpose:** Measure similarity between time series with different speeds/phases.

**Algorithm:**
1. Build cost matrix between all point pairs
2. Find optimal warping path (minimum cost alignment)
3. DTW distance = accumulated cost along path

**Advantages over Euclidean:**
- Handles time shifts and stretching
- Robust to different lengths
- Captures shape similarity

**Use Cases:**
- Pattern recognition
- Speech recognition
- Gesture classification
- Finding similar time series

**Parameters:**
- **Window**: Constraint on warping (Sakoe-Chiba band)
- **Step pattern**: Allowed moves in alignment

**DTW Distance Matrix:**
- Compute pairwise DTW between all series
- Use for clustering (hierarchical, k-medoids)
- Basis for k-NN classification

---

### Choosing Between Methods

| Situation | Recommended Method |
|-----------|-------------------|
| Stationary series | VAR |
| Non-stationary, cointegrated | VECM |
| Pattern similarity | DTW |
| Causality testing | VAR with Granger tests |
| Forecasting multiple series | VAR or VECM |

### How to Use

1. **VAR:**
   - Go to Time Series → Multivariate
   - Select multiple numeric columns
   - Choose VAR analysis
   - Review Granger causality results
   - Examine impulse responses

2. **VECM:**
   - Same interface as VAR
   - Enable cointegration test
   - If cointegrated, VECM is used automatically

3. **DTW:**
   - Select series to compare
   - View alignment path visualization
   - Compute distance matrix for clustering

### What to Do Next
- Use VAR forecasts with impulse response scenarios
- Interpret cointegrating vectors economically
- Cluster similar series using DTW distance matrix
- Validate with out-of-sample forecasting
""",

    # =========================================================================
    # BIOMASS SEGMENTATION (NEW)
    # =========================================================================

    "biomass_segmentation": """
## 🌿 Biomass Segmentation (U-Net)

### What & Why
Deep learning for semantic segmentation of aquatic imagery.
Identifies and quantifies Dreissena mussels, Cladophora algae, and other vegetation.

---

### U-Net Architecture

**Encoder (Contracting Path):**
- Conv layers extract features
- Max pooling reduces spatial size
- Captures "what" is in image

**Bottleneck:**
- Deepest layer with smallest spatial size
- Maximum abstraction

**Decoder (Expanding Path):**
- Transposed convolutions upsample
- Skip connections from encoder
- Recovers "where" information

**Skip Connections:**
- Copy features from encoder to decoder
- Preserve fine-grained spatial information
- Critical for precise boundaries

---

### Class Labels

| Class ID | Label | Color |
|----------|-------|-------|
| 0 | Background | Blue |
| 1 | Dreissena mussels | Red |
| 2 | Cladophora algae | Green |
| 3 | Other vegetation | Yellow |

---

### Loss Functions

**Dice Loss:**
- Better for imbalanced classes
- Directly optimizes IoU-like metric
- Dice = 2|A∩B| / (|A| + |B|)

**Combined Loss:**
- Weighted combination of Dice and cross-entropy
- Balances pixel-wise and region-wise accuracy

---

### Data Preparation

**Image Requirements:**
- Consistent resolution (e.g., 256x256)
- RGB format
- Clear visibility

**Mask Requirements:**
- Same size as images
- Integer class labels (0, 1, 2, 3)
- One mask per image

**Augmentation (Recommended):**
- Rotation (0-360°)
- Flipping (horizontal, vertical)
- Brightness/contrast adjustment
- Elastic deformation

---

### Training Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Image size | 256x256 | Balance accuracy and memory |
| Batch size | 8-16 | Reduce if memory limited |
| Epochs | 50-100 | Use early stopping |
| Learning rate | 1e-4 | Reduce if unstable |
| Validation split | 0.2 | 20% for validation |

**Early Stopping:**
- Monitor validation Dice coefficient
- Patience: 10-15 epochs
- Restore best weights

---

### Transfer Learning (VGG Backbone)

**Advantages:**
- Pretrained on ImageNet
- Better feature extraction
- Faster convergence
- Works with less data

**Freezing Strategy:**
1. Start with frozen encoder
2. Train decoder only (5-10 epochs)
3. Unfreeze and fine-tune all (lower LR)

---

### Output Analysis

**Per-Class Metrics:**
- Coverage percentage (% of image)
- Pixel count per class
- Class distribution histogram

**Spatial Analysis:**
- Density maps
- Connected component analysis
- Boundary detection

**Biomass Estimation:**
- Coverage × calibration factor = biomass
- Requires ground truth calibration

---

### How to Use

1. **Prepare Data:**
   - Organize images and masks in folders
   - Ensure consistent naming
   - Split into train/val/test

2. **Configure Model:**
   - Go to ML → Biomass Segmentation
   - Set image size and architecture
   - Enable augmentation for small datasets

3. **Train:**
   - Monitor loss curves
   - Check validation Dice score
   - Save best model

4. **Predict:**
   - Upload new images
   - View segmentation masks
   - Analyze class distributions

### What to Do Next
- Validate on independent test images
- Compare predicted vs. manual segmentation
- Calibrate coverage to actual biomass
- Apply to time series for temporal analysis
""",
}

def get_tutorial(topic: str) -> str:
    """Get comprehensive tutorial for a topic"""
    return COMPREHENSIVE_TUTORIALS.get(topic,
        "Tutorial not found. Available topics: " + ", ".join(COMPREHENSIVE_TUTORIALS.keys()))

def get_all_topics() -> list:
    """Get list of all available tutorial topics"""
    return list(COMPREHENSIVE_TUTORIALS.keys())

def get_short_tips() -> dict:
    """Quick tips for all methods"""
    return {
        "descriptive_stats": "Start here! Shows mean, median, range, skewness, kurtosis",
        "correlation": "Look for Pearson (linear), Spearman (monotonic), or Kendall correlations",
        "distributions": "Fit data to Normal, Gamma, Exponential, Lognormal, Weibull, or Beta",
        "statistical_tests": "Compare groups: t-test (2 groups), ANOVA (3+), chi-square (categorical)",
        "outliers": "IQR or Z-score methods. Investigate cause: error, measurement issue, or real event?",
        "kmeans": "Partition into K clusters. Use silhouette score to assess quality.",
        "hierarchical": "Dendrogram shows cluster structure. Better than K-Means for unknown K.",
        "dbscan": "Find non-circular clusters & outliers. Adjust eps and min_samples for results.",
        "gaussian_mixture": "Probabilistic clustering with soft assignments and uncertainty estimates.",
        "pca": "Reduce dimensions while preserving ~95% variance. Great for visualization!",
        "tsne": "Non-linear visualization. Shows local structure. Use for cluster discovery.",
        "umap": "Faster t-SNE alternative. Preserves more global structure. Can transform new data.",
        "isolation_forest": "Fast anomaly detection. Find outliers automatically.",
        "lof": "Local outlier factor. Good for detecting contextual anomalies.",
        "mcd": "Robust anomaly detection. Great for scientific/proteomics data.",
        "fourier": "Find dominant frequencies and periodicities in time series.",
        "wavelet": "Analyze time-varying frequencies. Best for non-stationary signals.",
        "stationarity": "Check if time series is suitable for ARIMA. Use ADF test.",
        "arima": "Forecast stationary time series. Use ACF/PACF to choose p,d,q.",
        "granger": "Test if past X predicts Y. Remember: predictive causality ≠ causation!",
        "association": "Find if-then rules. Use lift to find strong associations.",
    }
