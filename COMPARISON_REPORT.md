# Comprehensive Comparison Report
## data_analysis_toolkit v4.0 vs PAST 5 & Internal Audit

**Date**: 2025-07-16  
**Scope**: Feature-by-feature comparison against PAST 5 manual + internal USER_MANUAL.md audit

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Internal Audit: USER_MANUAL.md vs Implementation](#2-internal-audit)
3. [PAST 5 Comparison: Feature Matrix](#3-past-5-comparison)
4. [Detailed Gap Analysis by Category](#4-detailed-gap-analysis)
5. [Features Where We Excel Beyond PAST 5](#5-features-where-we-excel)
6. [Priority Implementation Roadmap](#6-priority-implementation-roadmap)
7. [Recommendations](#7-recommendations)

---

## 1. Executive Summary

### Overall Assessment

**data_analysis_toolkit v4.0** is a powerful, modern statistical analysis platform with **220+ methods** across 21+ modules. It significantly exceeds PAST 5 in machine learning, Bayesian analysis, neural networks, signal processing, and interactive visualization. However, it has **critical gaps** in ecological/paleontological statistics — precisely the domain where PAST 5 excels.

### Key Metrics

| Metric | data_analysis_toolkit | PAST 5 |
|--------|----------------------|--------|
| Total methods/features | ~220+ | ~200+ |
| Univariate statistics | ✅ Comprehensive (20+ tests) | ✅ Comprehensive |
| Multivariate ordination | ⚠️ PCA, t-SNE, UMAP only | ✅ PCA, CA, DCA, NMDS, PCO, CCA, RDA |
| Community ecology | ❌ Shannon/Simpson only | ✅ Full suite (15+ indices) |
| Machine learning | ✅✅ 25+ models | ❌ None |
| Neural networks | ✅ MLP, LSTM, GRU, Autoencoder | ❌ None |
| Bayesian analysis | ✅ Full suite | ❌ None |
| Signal processing | ✅✅ FFT, wavelets, coherence | ⚠️ Basic spectral |
| Morphometrics | ❌ None | ✅ Full suite |
| Stratigraphic tools | ❌ None | ✅ CONISS, range charts |
| Interactive UI | ✅ Streamlit + Plotly | ⚠️ Desktop GUI |
| Report generation | ✅ HTML/MD/LaTeX | ⚠️ Basic export |

### Critical Findings

1. **Neural Networks tab implemented but NOT wired to UI** — fully coded but `render()` never called
2. **Plugin system documented in manual but NOT implemented** — zero code exists
3. **USER_MANUAL.md documents v1.0** — app is v4.0 with 10+ undocumented modules
4. **Entire multivariate ordination suite missing** — CA, DCA, NMDS, PCO, CCA, RDA, PERMANOVA
5. **Ecological diversity toolkit nearly empty** — only Shannon/Simpson, missing rarefaction, Fisher's α, Chao1, species accumulation, beta diversity
6. **Zero morphometric capability** — no Procrustes, landmarks, thin-plate splines
7. **Zero stratigraphic tools** — no CONISS, range charts, biostratigraphic correlation

---

## 2. Internal Audit

### 2.1 Features Implemented but NOT in USER_MANUAL.md

The USER_MANUAL.md documents a v1.0 architecture with 13 tabs. The actual app (v4.0) has 8 main tabs with 20+ subtabs. The following **entire modules** are undocumented:

| Module | Location | Key Features |
|--------|----------|--------------|
| Effect Sizes | `effect_sizes.py` | Cohen's d, Hedges' g, Glass's Δ, η², ω², Cramér's V, phi, OR, Cohen's κ |
| Model Validation | `model_validation.py` | k-fold CV, nested CV, learning curves, calibration, ROC, residual diagnostics |
| Data Quality | `data_quality.py` | MCAR/MAR/MNAR detection, Little's test, KNN/MICE imputation, quality reports |
| Feature Selection | `feature_selection.py` | RFE, Boruta, SHAP-based, permutation, LASSO, sequential, ensemble |
| Interpretability | `interpretability.py` | SHAP values, LIME, PDP, ICE plots, permutation importance |
| Advanced Time Series | `advanced_timeseries.py` | Prophet, changepoint (PELT), DTW, VAR, impulse response |
| Survival Analysis | `survival_analysis.py` | Kaplan-Meier, Cox PH, log-rank, parametric survival, hazard plots |
| Domain-Specific | `domain_specific.py` | Environmental (Mann-Kendall, SPI, EVA), Clinical (Bland-Altman, ICC), Ecology |
| Signal Processing | `signal_analysis.py` | FFT, PSD, CWT, DWT, coherence, cross-wavelet, harmonic analysis |
| Report Generation | `report_generator.py` | HTML embedded reports, Markdown, LaTeX, Python script export |
| Bayesian Analysis | `bayesian_analysis.py` | Bayesian regression, MCMC, credible intervals, posterior predictive, prior sensitivity |
| Causality Analysis | `causality_analysis.py` | Granger causality, transfer entropy, lead-lag correlation |
| Sensitivity Analysis | `sensitivity_analysis.py` | Morris screening, Sobol indices, OAT analysis |
| CART Analysis | `cart_analysis.py` | Decision trees, rule extraction, Monte Carlo uncertainty |

### 2.2 Features in USER_MANUAL.md but NOT Implemented

| Feature | Manual Section | Status |
|---------|---------------|--------|
| Plugin System | Section 14 | ❌ **COMPLETELY MISSING** — manual describes plugin loading, code editor, auto-forms |
| Neural Networks Tab | Section 12.3 | ⚠️ **Code exists** (`neural_networks_tab.py`) but `render()` never called in `main()` |

### 2.3 Tab Architecture Discrepancy

**Manual documents (v1.0)**:
1. Data Explorer, 2. Basic Statistics, 3. Advanced Statistics, 4. Regression Analysis, 5. Time Series, 6. Machine Learning, 7. Signal Processing, 8. Distribution Analysis, 9. Correlation Analysis, 10. Visualization, 11. Report Generation, 12. Neural Networks, 13. Plugins

**Actual app (v4.0)**:
1. 📊 Data, 2. 📈 Statistics (7 subtabs), 3. 📡 Signal Processing, 4. ⏳ Time Series (5 subtabs), 5. 🤖 Machine Learning (3 subtabs), 6. 🔬 Scientific Tools (2 subtabs), 7. 📊 Visualization, 8. 📋 Reports

---

## 3. PAST 5 Comparison: Feature Matrix

### Legend
- ✅ = Fully implemented
- ⚠️ = Partially implemented
- ❌ = Not implemented
- N/A = Not applicable to that software

### 3.1 Univariate Statistics

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Descriptive statistics (mean, median, mode, SD, etc.) | ✅ | ✅ |
| Skewness & kurtosis | ✅ | ✅ |
| Percentiles / quantiles | ✅ | ✅ |
| Distribution fitting (normal, exp, lognormal, Weibull, beta, gamma) | ✅ | ✅ |
| Shapiro-Wilk test | ✅ | ✅ |
| Anderson-Darling test | ✅ | ✅ |
| Kolmogorov-Smirnov test | ✅ | ✅ |
| Jarque-Bera test | ❌ | ✅ |
| Lilliefors test | ❌ | ✅ |
| One-sample t-test | ✅ | ✅ |
| Sign test | ✅ | ✅ |
| QQ plots | ✅ | ✅ |
| Box plots / violin plots | ✅ | ✅ |
| Histogram with density | ✅ | ✅ |

### 3.2 Two-Sample Tests

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Independent t-test | ✅ | ✅ |
| Paired t-test | ✅ | ✅ |
| Welch's t-test | ✅ | ✅ |
| F-test for variances | ✅ (Levene, Bartlett) | ✅ |
| Mann-Whitney U | ✅ | ✅ |
| Wilcoxon signed-rank | ✅ | ✅ |
| Kolmogorov-Smirnov 2-sample | ✅ | ✅ |
| Permutation test (2-sample) | ❌ | ✅ |

### 3.3 Multi-Sample Tests

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| One-way ANOVA | ✅ | ✅ |
| Two-way ANOVA | ✅ | ✅ |
| Repeated measures ANOVA | ✅ | ✅ |
| Kruskal-Wallis | ✅ | ✅ |
| Friedman test | ✅ | ✅ |
| Mood's median test | ✅ | ✅ |
| Chi-squared test | ✅ | ✅ |
| Tukey HSD post-hoc | ✅ | ✅ |
| Bonferroni correction | ✅ | ✅ |
| Brown-Forsythe test | ✅ | ⚠️ |
| Dunn's test | ❌ | ✅ |

### 3.4 Correlation & Association

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Pearson correlation | ✅ | ✅ |
| Spearman rank correlation | ✅ | ✅ |
| Kendall tau | ✅ | ✅ |
| Distance correlation | ✅ | ❌ |
| Mutual Information / MIC | ✅ | ❌ |
| Correlation matrix / heatmap | ✅ | ✅ |
| Partial correlation | ❌ | ✅ |
| Mantel test | ❌ | ✅ |

### 3.5 Regression & Curve Fitting

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Linear regression (OLS) | ✅ | ✅ |
| Polynomial regression | ✅ | ✅ |
| Ridge / Lasso / ElasticNet | ✅ | ❌ |
| SVM regression | ✅ | ❌ |
| Gaussian Process regression | ✅ | ❌ |
| Spline regression | ✅ | ❌ |
| Neural network regression | ✅ | ❌ |
| Random Forest regression | ✅ | ❌ |
| Gradient Boosting regression | ✅ | ❌ |
| Reduced Major Axis (RMA/Type II) | ❌ | ✅ |
| Allometric (power) fitting | ❌ | ✅ |
| Exponential fitting | ❌ | ✅ |
| Logistic curve fitting | ❌ | ✅ |
| Sinusoidal fitting | ❌ | ✅ |
| Generalized Linear Models (GLM) | ❌ | ✅ |

### 3.6 Multivariate Ordination

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| PCA (Principal Component Analysis) | ✅ | ✅ |
| t-SNE | ✅ | ❌ |
| UMAP | ✅ | ❌ |
| ICA (Independent Component Analysis) | ✅ | ❌ |
| Autoencoder dimensionality reduction | ✅ | ❌ |
| CA (Correspondence Analysis) | ❌ | ✅ |
| DCA (Detrended Correspondence Analysis) | ❌ | ✅ |
| NMDS (Non-metric MDS) | ❌ | ✅ |
| PCO/PCoA (Principal Coordinates Analysis) | ❌ | ✅ |
| CCA (Canonical Correspondence Analysis) | ❌ | ✅ |
| RDA (Redundancy Analysis) | ❌ | ✅ |

### 3.7 Multivariate Hypothesis Testing

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| MANOVA | ❌ | ✅ |
| Discriminant Analysis (CVA/LDA) | ❌ | ✅ |
| ANOSIM | ❌ | ✅ |
| PERMANOVA (NPMANOVA) | ❌ | ✅ |
| SIMPER | ❌ | ✅ |
| Hotelling's T² | ❌ | ✅ |
| Box's M test | ❌ | ✅ |

### 3.8 Cluster Analysis

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Hierarchical clustering | ✅ | ✅ |
| K-means | ✅ | ✅ |
| DBSCAN | ✅ | ❌ |
| Gaussian Mixture Models | ✅ | ❌ |
| Silhouette analysis | ✅ | ❌ |
| Davies-Bouldin index | ✅ | ❌ |
| Calinski-Harabasz index | ✅ | ❌ |
| Neighbor joining | ❌ | ✅ |
| CONISS (stratigraphically constrained) | ❌ | ✅ |
| UPGMA | ✅ (via linkage) | ✅ |
| Ward's method | ✅ | ✅ |

### 3.9 Diversity & Community Ecology

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Shannon index (H') | ✅ | ✅ |
| Simpson index (1-D) | ✅ | ✅ |
| Fisher's alpha | ❌ | ✅ |
| Margalef index | ❌ | ✅ |
| Menhinick index | ❌ | ✅ |
| Berger-Parker dominance | ❌ | ✅ |
| Rarefaction (individual-based) | ❌ | ✅ |
| Rarefaction (sample-based) | ❌ | ✅ |
| Species accumulation curves | ❌ | ✅ |
| Chao1 / Chao2 estimators | ❌ | ✅ |
| ACE / ICE estimators | ❌ | ✅ |
| Beta diversity (Jaccard) | ❌ | ✅ |
| Beta diversity (Sørensen) | ❌ | ✅ |
| Beta diversity (Bray-Curtis) | ❌ | ✅ |
| Whittaker beta diversity | ❌ | ✅ |
| SHE analysis | ❌ | ✅ |

### 3.10 Time Series Analysis

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| ACF / PACF | ✅ | ✅ |
| Stationarity tests (ADF) | ✅ | ✅ |
| ARIMA / SARIMA | ✅ | ✅ |
| Seasonal decomposition | ✅ | ❌ |
| Prophet forecasting | ✅ | ❌ |
| Changepoint detection (PELT) | ✅ | ❌ |
| Dynamic Time Warping | ✅ | ❌ |
| VAR / VECM models | ✅ | ❌ |
| Cross-correlation | ✅ | ✅ |
| Rolling statistics | ✅ | ✅ |
| Mann-Kendall trend test | ✅ | ✅ |
| Theil-Sen estimator | ✅ | ❌ |
| Runs test | ✅ | ✅ |
| Lomb periodogram | ❌ | ✅ |
| LOWESS smoothing | ❌ | ✅ |
| Moving average smoothing | ✅ | ✅ |

### 3.11 Signal Processing / Spectral Analysis

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| FFT / Power spectrum | ✅ | ✅ |
| Welch PSD | ✅ | ❌ |
| Continuous Wavelet Transform (CWT) | ✅ | ✅ |
| Discrete Wavelet Transform (DWT) | ✅ | ❌ |
| Wavelet coherence | ✅ | ❌ |
| Cross-wavelet transform | ✅ | ❌ |
| Coherence analysis | ✅ | ❌ |
| Harmonic analysis | ✅ | ❌ |
| Savitzky-Golay filter | ✅ | ❌ |
| Lomb-Scargle periodogram | ❌ | ✅ |

### 3.12 Morphometrics

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Landmark-based analysis | ❌ | ✅ |
| Procrustes superimposition (GPA) | ❌ | ✅ |
| Thin-plate spline deformation | ❌ | ✅ |
| Relative warps (PCA of shapes) | ❌ | ✅ |
| Fourier shape analysis | ❌ | ✅ |
| Centroid size computation | ❌ | ✅ |

### 3.13 Stratigraphic Tools

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Biostratigraphic range charts | ❌ | ✅ |
| CONISS (constrained clustering) | ❌ | ✅ |
| Unitary associations | ❌ | ✅ |
| Seriation | ❌ | ✅ |
| Spindle diagrams | ❌ | ✅ |

### 3.14 Machine Learning & Advanced Modeling

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Random Forest classifier/regressor | ✅ | ❌ |
| Gradient Boosting | ✅ | ❌ |
| SVM classifier | ✅ | ❌ |
| KNN classifier | ✅ | ❌ |
| Naive Bayes | ✅ | ❌ |
| Neural Networks (MLP/LSTM/GRU) | ✅ (not in UI) | ❌ |
| Autoencoders | ✅ | ❌ |
| Transfer learning (images) | ✅ | ❌ |
| Anomaly detection (IF, LOF, One-Class SVM) | ✅ | ❌ |
| SHAP / LIME interpretability | ✅ | ❌ |
| Feature selection (RFE, Boruta) | ✅ | ❌ |
| Hyperparameter tuning | ✅ | ❌ |
| Cross-validation (k-fold, nested) | ✅ | ❌ |
| Learning curves | ✅ | ❌ |

### 3.15 Bayesian & Causal Analysis

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Bayesian regression | ✅ | ❌ |
| MCMC sampling | ✅ | ❌ |
| Prior sensitivity analysis | ✅ | ❌ |
| Posterior predictive checks | ✅ | ❌ |
| Model comparison (Bayes factors) | ✅ | ❌ |
| Granger causality | ✅ | ❌ |
| Transfer entropy | ✅ | ❌ |

### 3.16 Effect Sizes

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Cohen's d | ✅ | ⚠️ |
| Hedges' g | ✅ | ❌ |
| Glass's delta | ✅ | ❌ |
| Eta-squared / partial η² | ✅ | ❌ |
| Omega-squared | ✅ | ❌ |
| Cramér's V | ✅ | ❌ |
| Odds ratio | ✅ | ❌ |
| Cohen's kappa | ✅ | ❌ |

### 3.17 Survival Analysis

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Kaplan-Meier estimator | ✅ | ✅ |
| Cox proportional hazards | ✅ | ❌ |
| Log-rank test | ✅ | ✅ |
| Parametric survival models | ✅ | ❌ |
| Hazard ratio forest plots | ✅ | ❌ |

### 3.18 Resampling & Randomization

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Bootstrap CI | ✅ | ✅ |
| Permutation tests (general) | ⚠️ (feature importance only) | ✅ |
| Jackknife | ❌ | ✅ |
| Monte Carlo simulation | ✅ (CART uncertainty) | ✅ |
| Exact tests | ❌ | ✅ |

### 3.19 Specialized Plots

| Feature | data_analysis_toolkit | PAST 5 |
|---------|----------------------|--------|
| Scatter + regression overlay | ✅ | ✅ |
| 3D scatter | ✅ | ⚠️ |
| Heatmaps | ✅ | ✅ |
| Dendrograms | ✅ | ✅ |
| Interactive Plotly plots | ✅ | ❌ |
| Ternary diagrams | ❌ | ✅ |
| Rose diagrams (circular) | ❌ | ✅ |
| Spindle diagrams | ❌ | ✅ |
| Bubble plots | ❌ | ✅ |
| Survivorship curves | ✅ | ✅ |

---

## 4. Detailed Gap Analysis

### 4.1 Critical Gaps (High Impact, Core to Scientific Research)

#### GAP-1: Multivariate Ordination Suite
**Impact**: HIGH — Essential for ecological, environmental, and paleontological research  
**Missing methods**: CA, DCA, NMDS, PCO/PCoA, CCA, RDA  
**Why it matters**: These are standard methods in community ecology, environmental science, and paleontology. Any researcher comparing species assemblages across sites or environmental gradients needs these. PAST 5 has all of them.  
**Implementation complexity**: MEDIUM — `scikit-learn` has some (PCoA via `sklearn.manifold.MDS`), `scikit-bio` has NMDS/PCoA/PERMANOVA, `prince` has CA/MCA. DCA and CCA may need custom implementation.  
**Suggested libraries**: `scikit-bio`, `prince`, `skbio`, custom implementations

#### GAP-2: Multivariate Hypothesis Testing
**Impact**: HIGH — Required for testing group differences in multivariate data  
**Missing methods**: PERMANOVA, ANOSIM, MANOVA, discriminant analysis (CVA/LDA), SIMPER, Hotelling's T², Box's M  
**Why it matters**: Cannot test if species compositions differ between habitats, treatments, or time periods without PERMANOVA/ANOSIM.  
**Implementation complexity**: MEDIUM — `scikit-bio` has PERMANOVA/ANOSIM, `sklearn` has LDA, `scipy` or `statsmodels` for MANOVA  
**Suggested libraries**: `scikit-bio`, `sklearn.discriminant_analysis`, `statsmodels`

#### GAP-3: Diversity & Community Ecology
**Impact**: HIGH — Fundamental to ecological research  
**Missing methods**: Fisher's α, Margalef, Berger-Parker, rarefaction, species accumulation, Chao1/2, ACE/ICE, beta diversity (Jaccard, Sørensen, Bray-Curtis), SHE analysis  
**Why it matters**: Currently only Shannon & Simpson. Any biodiversity study needs rarefaction and richness estimators.  
**Implementation complexity**: LOW-MEDIUM — `scikit-bio` has most diversity metrics, rarefaction is straightforward  
**Suggested libraries**: `scikit-bio`, custom implementations

#### GAP-4: Curve Fitting Suite
**Impact**: MEDIUM-HIGH — Common in all sciences  
**Missing methods**: RMA/Type II regression, allometric (power) fit, exponential fit, logistic curve fit, sinusoidal fit, GLM  
**Why it matters**: Power/allometric fitting is essential in biology (body mass scaling). RMA regression is standard when both X and Y have error. GLM generalizes many analysis types.  
**Implementation complexity**: LOW — `scipy.optimize.curve_fit` for nonlinear, `statsmodels` for GLM, custom for RMA  
**Suggested libraries**: `scipy.optimize`, `statsmodels`

### 4.2 Important Gaps (Moderate Impact)

#### GAP-5: Morphometrics
**Impact**: MEDIUM — Critical for paleontology/biology, less relevant to general data science  
**Missing methods**: ALL — Procrustes (GPA), thin-plate spline, landmarks, relative warps, Fourier shape analysis  
**Why it matters**: Entire field of shape analysis missing. Essential for comparing morphological variation.  
**Implementation complexity**: HIGH — Needs specialized math, coordinate transformations, interpolation  
**Suggested libraries**: `morphops`, custom implementations, `scikit-image` for some

#### GAP-6: Stratigraphic Tools
**Impact**: MEDIUM — Specific to paleontology/geology  
**Missing methods**: ALL — CONISS, range charts, unitary associations, seriation, spindle diagrams  
**Implementation complexity**: HIGH — domain-specific, few Python libraries available  

#### GAP-7: Resampling Methods
**Impact**: MEDIUM — General statistical computing  
**Missing methods**: General permutation tests (not just feature importance), jackknife, exact tests  
**Implementation complexity**: LOW — straightforward to implement  

#### GAP-8: Missing Statistical Tests
**Impact**: LOW-MEDIUM  
**Missing**: Jarque-Bera, Lilliefors, Dunn's post-hoc, partial correlation  
**Implementation complexity**: LOW — all available in `scipy`/`statsmodels`/`scikit-posthocs`

### 4.3 Nice-to-Have Gaps (Low Impact)

#### GAP-9: Specialized Visualizations
**Missing**: Ternary diagrams, rose diagrams, spindle diagrams, bubble plots  
**Implementation**: `plotly` has ternary support natively; rose diagrams via polar bar charts

#### GAP-10: Additional Time Series
**Missing**: Lomb-Scargle periodogram, LOWESS smoothing  
**Implementation**: `scipy.signal.lombscargle`, `statsmodels.nonparametric.lowess`

---

## 5. Features Where We Excel Beyond PAST 5

Our toolkit has **substantial advantages** over PAST 5 in modern data science:

| Category | Our Advantage |
|----------|--------------|
| **Machine Learning** | 25+ models (RF, GBM, SVM, KNN, NB) vs zero in PAST 5 |
| **Deep Learning** | MLP, LSTM, GRU, autoencoders, transfer learning — none in PAST 5 |
| **Bayesian Analysis** | Full Bayesian regression, MCMC, prior sensitivity — none in PAST 5 |
| **Causal Inference** | Granger causality, transfer entropy — none in PAST 5 |
| **Signal Processing** | Welch PSD, DWT, CWT, coherence, cross-wavelet — PAST 5 has only basic FFT/wavelet |
| **Model Interpretability** | SHAP, LIME, PDP, ICE plots — none in PAST 5 |
| **Feature Engineering** | RFE, Boruta, sequential selection — none in PAST 5 |
| **Advanced Time Series** | Prophet, changepoint detection, DTW, VAR — none in PAST 5 |
| **Effect Sizes** | Full suite (10+ measures) — PAST 5 has minimal |
| **Survival Analysis** | Cox PH, parametric models, hazard plots — PAST 5 has only KM + log-rank |
| **Interactive Visualization** | Plotly 3D, interactive dashboards — PAST 5 has static plots |
| **Report Generation** | HTML/MD/LaTeX automated reports — PAST 5 has basic export |
| **Data Quality** | MCAR/MAR/MNAR analysis, multiple imputation — none in PAST 5 |
| **Sensitivity Analysis** | Morris, Sobol, OAT — none in PAST 5 |
| **Rust Acceleration** | Optional Rust-compiled modules for speed — PAST 5 is pure interpreted |
| **Modern Dimensionality Reduction** | t-SNE, UMAP, ICA, autoencoders — PAST 5 only has classical ordination |
| **Anomaly Detection** | Isolation Forest, LOF, One-Class SVM — none in PAST 5 |

---

## 6. Priority Implementation Roadmap

### Phase 1: Critical Fixes (Immediate — Days)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P0 | **Wire Neural Networks tab into main UI** | 1 hour | HIGH — already coded, just needs render call |
| P0 | **Update USER_MANUAL.md for v4.0** | 2-3 days | HIGH — manual is 3 versions behind |
| P1 | **Add missing simple tests** (Jarque-Bera, Lilliefors, Dunn's, partial correlation) | 1 day | MEDIUM |
| P1 | **Add curve fitting** (power, exponential, logistic, sinusoidal, RMA) | 1-2 days | MEDIUM-HIGH |

### Phase 2: Ecological Statistics Module (Weeks)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P1 | **Diversity indices** (Fisher's α, Margalef, Berger-Parker, Chao1/2, ACE) | 2-3 days | HIGH |
| P1 | **Rarefaction & species accumulation** curves | 1-2 days | HIGH |
| P1 | **Beta diversity** (Jaccard, Sørensen, Bray-Curtis dissimilarity) | 1-2 days | HIGH |
| P2 | **Create Ecology/Community Analysis subtab** in Scientific Tools | 1 day | HIGH |

### Phase 3: Multivariate Ordination (Weeks)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P1 | **NMDS** ordination | 2-3 days | HIGH |
| P1 | **PCoA** (Principal Coordinates Analysis) | 1-2 days | HIGH |
| P1 | **CA** (Correspondence Analysis) | 2 days | HIGH |
| P2 | **DCA** (Detrended CA) | 2-3 days | MEDIUM |
| P2 | **CCA** (Canonical CA) | 2-3 days | MEDIUM |
| P2 | **RDA** (Redundancy Analysis) | 2 days | MEDIUM |

### Phase 4: Multivariate Testing (Weeks)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P1 | **PERMANOVA** (permutational MANOVA) | 2-3 days | HIGH |
| P1 | **ANOSIM** | 1-2 days | HIGH |
| P2 | **LDA/CVA** (discriminant analysis) | 2 days | MEDIUM |
| P2 | **MANOVA** | 2 days | MEDIUM |
| P2 | **SIMPER** | 1-2 days | MEDIUM |
| P3 | **Mantel test** | 1 day | LOW-MEDIUM |

### Phase 5: Morphometrics & Stratigraphy (Months — if desired)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P3 | **Procrustes analysis (GPA)** | 1 week | MEDIUM (domain-specific) |
| P3 | **Landmark I/O & visualization** | 3-5 days | MEDIUM |
| P3 | **Thin-plate spline** | 1 week | MEDIUM |
| P4 | **CONISS** | 3-5 days | LOW (niche) |
| P4 | **Range charts / spindle diagrams** | 2-3 days | LOW |

### Phase 6: Plugin System (Weeks)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P2 | **Implement plugin system** as documented in manual | 1-2 weeks | MEDIUM-HIGH |

---

## 7. Recommendations

### 7.1 Immediate Actions (Do Now)

1. **Wire the Neural Networks tab** — add `render_neural_networks_tab()` call in `streamlit_app.py` `main()`. This is literally a one-line fix for a fully implemented feature.

2. **Rewrite USER_MANUAL.md** — the manual is dangerously outdated. Users will be confused by references to 13 tabs that no longer exist and will miss 14 powerful modules.

3. **Add `scipy` curve fitting wrapper** — power, exponential, logistic, and sinusoidal fits are trivial to add via `scipy.optimize.curve_fit` and would close the most obvious regression gap vs PAST 5.

### 7.2 Strategic Recommendations

4. **Create an "Ecology & Diversity" module** (`ecology.py`) — consolidate diversity indices, beta diversity, rarefaction, and species accumulation. This is where PAST 5 dominates and our toolkit is weakest.

5. **Create a "Multivariate Ordination" module** (`ordination.py`) — add NMDS, PCoA, CA, CCA, RDA. Use `scikit-bio` as the foundation. This is the second-biggest gap.

6. **Add PERMANOVA + ANOSIM** — these are the multivariate equivalents of ANOVA/Kruskal-Wallis. Critical for any ecological research.

7. **Consider `scikit-bio` as a dependency** — it provides PERMANOVA, ANOSIM, PCoA, NMDS, diversity indices, distance matrices, and beta diversity in one well-maintained package.

### 7.3 What NOT to Prioritize

8. **Morphometrics** — unless your user base specifically needs shape analysis, this is a large investment for a niche audience. Consider as a plugin.

9. **Stratigraphic tools** — CONISS, range charts, and unitary associations are highly specialized paleontological tools. Low ROI unless targeting paleontologists specifically.

10. **Full parity with PAST 5** — PAST 5 is a 20-year-old desktop application specifically designed for paleontologists. Our toolkit's strength is modern data science + web interface. Focus on closing the ecological statistics gaps (diversity, ordination, PERMANOVA) rather than replicating every niche feature.

---

## Appendix A: Implementation Quick Wins

### A.1 One-Line Fix: Neural Networks Tab

In `streamlit_app.py`, add the neural networks tab render call in the main function where tabs are rendered.

### A.2 Curve Fitting (Power, Exponential, Logistic, Sinusoidal)

```python
# These can all be added via scipy.optimize.curve_fit:
from scipy.optimize import curve_fit

def power_fit(x, a, b): return a * x**b
def exponential_fit(x, a, b): return a * np.exp(b * x)
def logistic_fit(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
def sinusoidal_fit(x, A, f, phi, offset): return A * np.sin(2*np.pi*f*x + phi) + offset
```

### A.3 Diversity Indices

```python
# Fisher's alpha: iterative solution to S = a * ln(1 + N/a)
# Margalef: (S - 1) / ln(N)
# Berger-Parker: Nmax / N
# Chao1: S_obs + (f1*(f1-1)) / (2*(f2+1))
# where f1 = singletons, f2 = doubletons
```

### A.4 Suggested New Dependencies

```
scikit-bio>=0.5.8    # PERMANOVA, ANOSIM, PCoA, NMDS, diversity
prince>=0.8.0        # CA, MCA, FAMD
scikit-posthocs>=0.7 # Dunn's test, other post-hoc tests
plotly>=5.0          # (already used) — ternary support built-in
```

---

## Appendix B: Complete Feature Count

| Category | Our Count | PAST 5 Count | Gap |
|----------|-----------|---------------|-----|
| Univariate tests | 20+ | 15+ | We lead |
| Multivariate ordination | 4 | 7 | -3 |
| Multivariate testing | 0 | 7 | -7 |
| Cluster methods | 6 | 5 | +1 |
| Diversity indices | 2 | 15+ | -13 |
| Regression/curve fitting | 10 | 10 | Tied (different methods) |
| Time series | 25+ | 10 | +15 |
| Signal processing | 15+ | 5 | +10 |
| Machine learning | 25+ | 0 | +25 |
| Bayesian | 7 | 0 | +7 |
| Effect sizes | 10+ | 2 | +8 |
| Morphometrics | 0 | 6 | -6 |
| Stratigraphic | 0 | 5 | -5 |
| Visualization types | 60+ | 30+ | +30 |
| **TOTAL estimated** | **~220+** | **~200+** | **We lead overall** |

---

*Report generated from systematic codebase analysis of data_analysis_toolkit v4.0 compared against PAST 5 (Hammer et al.) feature set.*
