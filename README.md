# Advanced Data Analysis Toolkit v9.1

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready data analysis toolkit with **15 analysis tabs**, interactive Plotly visualizations, and multiple interfaces (Streamlit web app, Jupyter notebooks, desktop GUI).

---

## 📂 Tab Organization (6 Categories)

The toolkit is organized into **6 main categories** with **15 specialized tabs**:

### 📁 DATA
| Tab | Description |
|-----|-------------|
| **Data Loading** | Upload CSV/Excel, preview data, select feature and target columns |

### 📊 STATISTICS
| Tab | Description | Key Methods |
|-----|-------------|-------------|
| **Descriptive Statistics** | Summary statistics, correlations, outliers | Mean, Std, Correlation (Pearson/Spearman/Kendall), IQR outliers |
| **Hypothesis Tests** | Statistical significance tests | t-test, ANOVA, Chi-square, Shapiro-Wilk normality |
| **Bayesian Inference** | Probabilistic analysis | Posterior distributions, Credible Intervals (CI), BIC model comparison |
| **Uncertainty Analysis** | Confidence and prediction intervals | Bootstrap CI, Residual analysis, Monte Carlo, Prediction intervals |

### 🔊 SIGNAL PROCESSING
| Tab | Description | Key Methods |
|-----|-------------|-------------|
| **Signal Processing (FFT/Wavelet)** | Frequency and time-frequency analysis | FFT, PSD (Welch), CWT (Continuous Wavelet), DWT (Discrete Wavelet) |

### ⏱️ TIME SERIES
| Tab | Description | Key Methods |
|-----|-------------|-------------|
| **Time Series Analysis** | Temporal pattern analysis | ACF/PACF, ADF stationarity test, Seasonal decomposition, ARIMA |
| **Causality (Granger)** | Predictive causality testing | Granger causality test, Lead-lag correlation analysis |

### 🤖 MACHINE LEARNING
| Tab | Description | Key Methods |
|-----|-------------|-------------|
| **ML Regression/Classification** | Supervised learning models | Linear/Ridge/Lasso/ElasticNet, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting |
| **PCA (Principal Components)** | Linear dimensionality reduction | PCA scree plot, Biplot with feature loading vectors |
| **Clustering** | Unsupervised grouping | K-Means, Hierarchical (Agglomerative), DBSCAN, GMM (Gaussian Mixture) |
| **Anomaly Detection** | Outlier identification | Isolation Forest, LOF (Local Outlier Factor), MCD |
| **Dimensionality Reduction** | Visualization & preprocessing | PCA, SVD, t-SNE, UMAP, ICA |
| **Non-Linear Analysis** | Non-linear relationships | Distance correlation, Mutual Information (MI), Polynomial regression, GPR |

### 📈 VISUALIZATION
| Tab | Description | Key Plots |
|-----|-------------|-----------|
| **Visualization & Plots** | Interactive charts | Scatter Matrix, Correlation Heatmap, Box plots, 3D Scatter, **Linear Regression with Statistics (slope, intercept, R², p-value)** |

---

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)

```bash
# Navigate to the toolkit directory
cd advanced_data_toolkit

# Install dependencies
pip install -e .

# Run the Streamlit app
streamlit run src/data_toolkit/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Option 2: VS Code Task

If using VS Code, press `Ctrl+Shift+B` to run the "🌐 Run Streamlit" task.

### Option 3: Python Scripts

```python
import pandas as pd
from data_toolkit import StatisticalAnalysis, MLModels

# Load data
df = pd.read_csv('your_data.csv')

# Statistical analysis
stats = StatisticalAnalysis(df)
print(stats.descriptive_stats(['col1', 'col2']))
print(stats.correlation_matrix(['col1', 'col2', 'col3']))

# Machine learning
ml = MLModels(df)
results = ml.train_model(['feature1', 'feature2'], 'target', 'Random Forest Regressor')
print(f"R² Score: {results['r2']:.4f}")
```

---

## 📊 Feature Details

### Regression Models (Predicting Continuous Values)

| Model | Full Name | Description |
|-------|-----------|-------------|
| **Linear Regression** | Ordinary Least Squares | Fits y = mx + b; interpretable baseline |
| **Ridge Regression** | L2-regularized Linear Regression | Prevents overfitting with squared penalty |
| **Lasso Regression** | L1-regularized Linear Regression | Automatic feature selection (sets coefficients to 0) |
| **ElasticNet** | L1 + L2 Regularization | Combines Ridge and Lasso benefits |
| **Decision Tree Regressor** | CART Decision Tree | Non-linear splits, interpretable rules |
| **KNN Regressor** | K-Nearest Neighbors Regressor | Averages k nearest training points |
| **SVR** | Support Vector Regression | Uses kernel tricks for non-linear regression |
| **Random Forest Regressor** | Ensemble of Decision Trees | Bagging for robust predictions |
| **Gradient Boosting Regressor** | Sequential Boosting | Iteratively corrects errors; highest accuracy |

### Classification Models (Predicting Categories)

| Model | Full Name | Description |
|-------|-----------|-------------|
| **Logistic Regression** | Logistic/Sigmoid Classifier | Probabilistic binary/multiclass classifier |
| **SVM** | Support Vector Machine | Finds optimal separating hyperplane |
| **Decision Tree Classifier** | CART Classifier | Rule-based classification |
| **KNN Classifier** | K-Nearest Neighbors | Majority vote of k nearest neighbors |
| **Naive Bayes (Gaussian)** | Gaussian Naive Bayes | Assumes feature independence; fast |
| **Random Forest Classifier** | Ensemble Classifier | Voting from many decision trees |
| **Gradient Boosting Classifier** | Sequential Boosting | State-of-the-art accuracy |

### Evaluation Metrics

**Regression Metrics:**
| Metric | Full Name | Interpretation |
|--------|-----------|----------------|
| **R²** | R-squared (Coefficient of Determination) | 0-1; higher = better; 0.7+ is good |
| **RMSE** | Root Mean Squared Error | Same units as target; lower = better |
| **MAE** | Mean Absolute Error | Average absolute prediction error |
| **MSE** | Mean Squared Error | Squared error; penalizes large errors |

**Classification Metrics:**
| Metric | Interpretation |
|--------|----------------|
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of predicted positives, how many are correct? |
| **Recall** | Of actual positives, how many found? |
| **F1-Score** | Harmonic mean of precision and recall |

---

## 🔊 Signal Processing Acronyms

| Acronym | Full Name | Description |
|---------|-----------|-------------|
| **FFT** | Fast Fourier Transform | Converts time domain → frequency domain |
| **PSD** | Power Spectral Density | Power distribution across frequencies (Welch method) |
| **CWT** | Continuous Wavelet Transform | Time-frequency representation with scalable wavelets |
| **DWT** | Discrete Wavelet Transform | Multi-scale decomposition for denoising |
| **COI** | Cone of Influence | Edge effect region in wavelet transforms |

**Wavelet Types:**
- **morl** (Morlet): Good frequency resolution
- **mexh** (Mexican Hat): Good time resolution
- **db4/db8** (Daubechies): General-purpose discrete wavelets
- **haar**: Simplest wavelet for step detection

---

## ⏱️ Time Series Acronyms

| Acronym | Full Name | Description |
|---------|-----------|-------------|
| **ACF** | Autocorrelation Function | Correlation with lagged values |
| **PACF** | Partial Autocorrelation Function | Direct correlation after removing intermediate effects |
| **ADF** | Augmented Dickey-Fuller Test | Tests for stationarity (p < 0.05 = stationary) |
| **ARIMA** | AutoRegressive Integrated Moving Average | ARIMA(p,d,q) time series model |

---

## 📉 Dimensionality Reduction Acronyms

| Acronym | Full Name | Description |
|---------|-----------|-------------|
| **PCA** | Principal Component Analysis | Linear; maximizes variance |
| **SVD** | Singular Value Decomposition | Matrix factorization; works on sparse data |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding | Non-linear; preserves local structure |
| **UMAP** | Uniform Manifold Approximation and Projection | Non-linear; preserves local + global structure |
| **ICA** | Independent Component Analysis | Finds statistically independent components |

---

## 🎯 Clustering Acronyms

| Acronym | Full Name | Description |
|---------|-----------|-------------|
| **K-Means** | K-Means Clustering | Minimizes within-cluster variance |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise | Finds arbitrary-shaped clusters |
| **GMM** | Gaussian Mixture Model | Probabilistic soft clustering |
| **LOF** | Local Outlier Factor | Density-based anomaly detection |
| **MCD** | Minimum Covariance Determinant | Robust covariance estimation |

---

## 📊 Statistical Test Acronyms

| Acronym | Full Name | When to Use |
|---------|-----------|-------------|
| **ANOVA** | Analysis of Variance | Compare means of 3+ groups |
| **CI** | Confidence Interval | Range where true value likely lies |
| **BIC** | Bayesian Information Criterion | Model comparison (lower = better) |
| **AIC** | Akaike Information Criterion | Model comparison (lower = better) |

---

## 📈 Linear Regression Plot (with Statistics)

The **Visualization & Plots** tab includes a **Linear Regression Plot** that shows:

1. **Scatter plot** of data points
2. **Best-fit regression line**: y = slope × x + intercept
3. **95% Confidence Interval** band (optional)
4. **Statistical parameters:**
   - **Slope (m)**: Change in Y per unit change in X
   - **Intercept (b)**: Y value when X = 0
   - **R² (R-squared)**: Variance explained (0-1)
   - **p-value**: Statistical significance of the slope
   - **Correlation (r)**: Pearson correlation coefficient
   - **Standard Error**: Uncertainty in slope estimate

---

## 📁 Test Data Files

Sample datasets in `test_data/`:

| File | Rows | Purpose |
|------|------|---------|
| `general_analysis_data.csv` | 500 | Statistical analysis |
| `regression_data.csv` | 600 | ML regression |
| `timeseries_data.csv` | 730 | Time series analysis |
| `causality_data.csv` | 500 | Granger causality testing |
| `nonlinear_data.csv` | 400 | Non-linear relationships |
| `bayesian_uncertainty_data.csv` | 300 | Bayesian analysis |
| `clustering_data.csv` | 500 | Clustering |
| `ml_classification_train.csv` | 60 | Classification training |
| `ml_classification_predict.csv` | 10 | Classification prediction |
| `ml_regression_train.csv` | 30 | Regression training |
| `ml_regression_predict.csv` | 10 | Regression prediction |

---

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Steps

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install the package**:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "from data_toolkit import StatisticalAnalysis; print('✅ Installation successful!')"
   ```

### Optional: Rust Acceleration (10-50x speedup)

```bash
cd rust_extensions
pip install maturin
maturin develop --release
```

---

## 🎨 Interactive Charts

All charts use **Plotly** for full interactivity:

| Action | How |
|--------|-----|
| **Zoom** | Click and drag |
| **Pan** | Shift + drag |
| **Hover** | Mouse over for values |
| **Download** | Camera icon → PNG |
| **Reset** | Double-click |

---

## 📚 Additional Documentation

- `USER_MANUAL.md` - Detailed user manual
- `TUTORIAL.md` - Step-by-step tutorial
- `QUICK_START.md` - Quick start guide
- `CONTRIBUTING.md` - Contribution guidelines
- `test_data/README.md` - Test data documentation

---

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -e .
```

### Streamlit Won't Start
```bash
pip install streamlit
streamlit run src/data_toolkit/streamlit_app.py --server.port 8502
```

### Missing Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels plotly streamlit
```

---

## 📝 Version History

| Version | Changes |
|---------|---------|
| **9.1** | Reorganized GUI into 6 categories, added Linear Regression Plot with statistics, expanded documentation |
| 9.0 | Streamlit interface, Plotly charts |
| 8.0 | Rust acceleration, plugin system |

---

## 📄 License

MIT License - see LICENSE file.

---

## 🤝 Contributing

See CONTRIBUTING.md for guidelines.
