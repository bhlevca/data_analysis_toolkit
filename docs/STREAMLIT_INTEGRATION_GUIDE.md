# Streamlit Integration Guide — Image Tab

This document explains how to access and use the new Image Recognition tab inside the Streamlit app.

Open the app:

```bash
streamlit run src/data_toolkit/streamlit_app.py --server.port 8501
```

Navigate: Machine Learning → Image Recognition

Controls in the tab:
- Generate Synthetic Dataset: creates `images/`, `predict_examples/`, and `labels.csv` under the chosen output folder.
- Train CNN Model:
  - **Data folder**: click `Browse` to open a compact folder browser (or `Native Browse` to open your OS folder dialog when running Streamlit locally), then `Select` the dataset folder. The selected folder is shown in the **Selected data folder** field. The folder must contain `images/` and `labels.csv` (or upload `labels.csv` via the UI).
  - **Save model**: use the `Save model automatically` checkbox to write the trained model as a `.keras` file during training; if you choose not to auto-save, a `Save trained model` widget appears after training so you can persist the in-memory model.
  - Tunable hyperparameters: epochs, batch size, image size, CNN depth, base filters, dense units.
- Predict Single Image: Upload or choose from images found in the selected data folder (the UI lists candidates from `predict_examples/` and `images/`).

Notes: the UI prefers `.keras` format for saved models and will only show legacy `.h5` files if no `.keras` files are present in `models/`.
- Interactive Labeling: review `predict_examples/` images, assign labels; labeled images are moved into `images/` and appended to `labels.csv` as `train` split entries.

Notes and troubleshooting:
- Ensure TensorFlow is installed for training (`pip install -e .[neural]`)
- If CPU-only, prefer smaller image sizes (64–128) and fewer epochs for responsiveness.
- Uploaded `labels.csv` will be stored in the selected data folder before training.
# 🎨 Streamlit UI Integration Guide for v2.0 Features

## Overview
This guide provides code snippets and recommendations for integrating the new v2.0 features into the Streamlit application.

---

## 1. ML Module Integration

### Clustering Tab Example
```python
st.subheader("🎯 Clustering Analysis")

clustering_method = st.selectbox(
    "Choose Clustering Method:",
    ["K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Model"]
)

if clustering_method == "K-Means":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    n_init = st.slider("Number of Initializations", 5, 20, 10)

    results = ml.kmeans_clustering(feature_cols, n_clusters=n_clusters, n_init=n_init)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
        st.metric("Davies-Bouldin Index", f"{results['davies_bouldin_index']:.3f}")
    with col2:
        st.write("Cluster Sizes:", results['cluster_sizes'])

    fig = ml.plot_clustering_results(results)
    st.plotly_chart(fig, use_container_width=True)
```

### Dimensionality Reduction Tab
```python
st.subheader("📉 Dimensionality Reduction")

dim_method = st.selectbox(
    "Choose Method:",
    ["PCA", "SVD", "t-SNE", "UMAP", "ICA"]
)

if dim_method == "PCA":
    variance_threshold = st.slider("Variance Threshold", 0.80, 0.99, 0.95)
    results = ml.pca_analysis(feature_cols, variance_threshold=variance_threshold)

    st.write(f"Selected Components: {results['n_components_selected']}")
    st.write(f"Total Variance Explained: {results['total_variance_explained']:.2%}")

    # Plot scree plot
    fig = ml.plot_pca_results(results)
    st.plotly_chart(fig, use_container_width=True)

elif dim_method == "t-SNE":
    perplexity = st.slider("Perplexity", 5, 50, 30)
    results = ml.tsne_analysis(feature_cols, perplexity=perplexity)

    fig = ml.plot_dimensionality_reduction(results)
    st.plotly_chart(fig, use_container_width=True)
```

### Anomaly Detection Tab
```python
st.subheader("🚨 Anomaly Detection")

anomaly_method = st.selectbox(
    "Detection Method:",
    ["Isolation Forest", "Local Outlier Factor", "Minimum Covariance Determinant"]
)

contamination = st.slider("Expected Contamination Rate", 0.01, 0.2, 0.1)

if anomaly_method == "Isolation Forest":
    n_estimators = st.slider("Number of Estimators", 50, 200, 100)
    results = ml.isolation_forest_anomaly(feature_cols, contamination=contamination)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anomalies Found", results['n_anomalies'])
        st.metric("Anomaly %", f"{results['anomaly_percentage']:.2f}%")
    with col2:
        st.write("Anomaly Indices (first 10):", results['anomaly_indices'][:10])

    fig = ml.plot_anomalies(results)
    st.plotly_chart(fig, use_container_width=True)
```

---

## 2. Statistical Analysis Integration

### Enhanced Statistical Tests Tab
```python
st.subheader("🧪 Statistical Hypothesis Tests")

test_type = st.selectbox(
    "Choose Test Type:",
    ["Compare 2 Groups", "Compare 3+ Groups", "Categorical Association"]
)

if test_type == "Compare 2 Groups":
    col1, col2 = st.columns(2)
    with col1:
        col1_select = st.selectbox("Column 1:", numerical_cols)
    with col2:
        col2_select = st.selectbox("Column 2:", numerical_cols)

    test_method = st.radio("Test Method:", ["Parametric (t-test)", "Non-parametric (Mann-Whitney)"])

    if test_method == "Parametric (t-test)":
        results = stats.ttest_independent(col1_select, col2_select)
        st.write(f"t-statistic: {results['statistic']:.4f}")
        st.write(f"p-value: {results['p_value']:.4f}")
        st.write(f"**Interpretation**: {results['interpretation']}")
    else:
        results = stats.mann_whitney_u(col1_select, col2_select)
        st.write(f"U-statistic: {results['statistic']:.4f}")
        st.write(f"p-value: {results['p_value']:.4f}")
        st.write(f"**Interpretation**: {results['interpretation']}")
```

### Distribution Fitting Tab
```python
st.subheader("📊 Distribution Fitting & Analysis")

selected_col = st.selectbox("Select Column:", numerical_cols)

# Fit multiple distributions
results = stats.fit_distributions(selected_col)

# Display results
col1, col2, col3 = st.columns(3)
for i, (dist_name, fit_data) in enumerate(results.items()):
    if 'error' not in fit_data:
        with st.columns(3)[i % 3]:
            st.write(f"**{dist_name}**")
            st.write(f"K-S Stat: {fit_data.get('ks_stat', 'N/A'):.4f}")

# Plot fitted distributions
fig = stats.plot_distribution_fit(selected_col)
st.plotly_chart(fig, use_container_width=True)

# Show detailed analysis
st.subheader("Distribution Analysis")
analysis = stats.distribution_analysis([selected_col])
st.write(analysis[selected_col])
```

---

## 3. Time Series Analysis Integration

### Fourier Analysis Tab
```python
st.subheader("📡 Fourier Analysis")

ts_col = st.selectbox("Select Time Series:", numerical_cols)
sampling_rate = st.number_input("Sampling Rate", value=1.0, min_value=0.1)

# FFT Analysis
fft_results = ts.fourier_transform(ts_col, sampling_rate=sampling_rate)

# PSD Analysis
psd_results = ts.power_spectral_density(ts_col, sampling_rate=sampling_rate)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dominant Frequencies (FFT)")
    for i, freq in enumerate(fft_results['dominant_frequencies'][:5]):
        st.write(f"{i+1}. {freq:.4f} Hz")

with col2:
    st.subheader("Dominant Frequencies (PSD)")
    for i, freq in enumerate(psd_results['dominant_frequencies'][:5]):
        st.write(f"{i+1}. {freq:.4f} Hz")

# Plots
fig_fft = ts.plot_fft(fft_results, ts_col)
st.plotly_chart(fig_fft, use_container_width=True)

fig_psd = ts.plot_power_spectral_density(psd_results, ts_col)
st.plotly_chart(fig_psd, use_container_width=True)
```

### Wavelet Analysis Tab
```python
st.subheader("🌊 Wavelet Analysis")

ts_col = st.selectbox("Select Time Series:", numerical_cols)

wavelet_type = st.radio("Wavelet Analysis Type:",
                        ["Continuous (CWT)", "Discrete (DWT)"])

if wavelet_type == "Continuous (CWT)":
    results = ts.continuous_wavelet_transform(ts_col)

    fig = ts.plot_wavelet_power(results, ts_col)
    st.plotly_chart(fig, use_container_width=True)

    st.info("🔍 **How to Read**: Darker colors = stronger signal at that time-frequency")

else:  # Discrete
    wavelet_family = st.selectbox("Wavelet Family:", ["db4", "db8", "sym5", "coif5"])
    level = st.slider("Decomposition Level", 1, 5, 3)

    results = ts.discrete_wavelet_transform(ts_col, wavelet=wavelet_family, level=level)

    fig = ts.plot_discrete_wavelet(results, ts_col)
    st.plotly_chart(fig, use_container_width=True)
```

---

## 4. Tutorial System Integration

### Sidebar Tutorial Widget
```python
def render_help_sidebar():
    with st.sidebar:
        st.markdown("### 📚 Method Help")

        help_topic = st.selectbox(
            "Get Help On:",
            comprehensive_tutorial.get_all_topics(),
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if help_topic:
            tutorial = comprehensive_tutorial.get_tutorial(help_topic)
            st.markdown(tutorial)

        st.markdown("---")
        st.markdown("### ⚡ Quick Tips")
        tips = comprehensive_tutorial.get_short_tips()
        selected_method = st.selectbox("Method:", list(tips.keys()))
        st.info(tips[selected_method])
```

### Help Modal in Each Tab
```python
col1, col2 = st.columns([0.95, 0.05])
with col2:
    if st.button("❓"):
        with st.expander("About this analysis"):
            st.markdown(comprehensive_tutorial.get_tutorial("kmeans_clustering"))
```

---

## 5. Interactive Parameter Controls

### Slider Controls for Adjustable Parameters
```python
st.subheader("⚙️ Method Parameters")

params_col1, params_col2, params_col3 = st.columns(3)

with params_col1:
    # For clustering
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=min(10, len(df)//5),
        value=3,
        help="Adjust for different granularity of clustering"
    )

with params_col2:
    # For statistical tests
    alpha = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.1,
        value=0.05,
        step=0.01,
        help="Threshold for p-value. Common: 0.05"
    )

with params_col3:
    # For dimensionality reduction
    variance_threshold = st.slider(
        "Variance Threshold",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Keep components explaining this much variance"
    )

# Update results based on parameters
if st.button("Run Analysis", key="run_with_params"):
    # Run analysis with selected parameters
    pass
```

---

## 6. Recommended Tab Organization

```python
# Main navigation
tabs = st.tabs([
    "📊 Data Overview",
    "📈 Statistical Analysis",
    "🧪 Statistical Tests",
    "🤖 Clustering",
    "📉 Dimensionality Reduction",
    "🚨 Anomaly Detection",
    "⏱️ Time Series",
    "📡 Fourier Analysis",
    "🌊 Wavelet Analysis",
    "🔗 Association Rules",
    "📚 Help & Tutorials"
])

with tabs[0]:  # Data Overview
    # Descriptive stats, distributions
    pass

with tabs[1]:  # Statistical Analysis
    # Correlations, distributions, outliers
    pass

# ... and so on for each tab
```

---

## 7. Export & Reporting

### Add Export Options
```python
st.subheader("📥 Export Results")

export_format = st.selectbox("Format:", ["CSV", "JSON", "PDF"])

if st.button("Export"):
    if export_format == "CSV":
        st.download_button(
            label="Download CSV",
            data=results_df.to_csv(),
            file_name="analysis_results.csv",
            mime="text/csv"
        )
    elif export_format == "JSON":
        st.download_button(
            label="Download JSON",
            data=json.dumps(results_dict, indent=2),
            file_name="analysis_results.json",
            mime="application/json"
        )
```

---

## 8. Performance Optimization Tips

1. **Cache Results**: Use `@st.cache_data` for expensive computations
2. **Progressive Loading**: Show results as they compute
3. **Sample Large Data**: For exploration, use sample before full analysis
4. **Async Operations**: Use threads for multiple long-running analyses
5. **Lazy Loading**: Load visualizations only when requested

---

## 9. User Experience Improvements

### Add Informational Cards
```python
import streamlit as st

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Best Method for Clustering",
        value="K-Means",
        delta="Fast & interpretable"
    )

with col2:
    st.metric(
        label="Best Method for Visualization",
        value="t-SNE/UMAP",
        delta="Non-linear"
    )

with col3:
    st.metric(
        label="Best Method for High-Dim",
        value="PCA",
        delta="Linear & interpretable"
    )
```

### Add Warnings for Data Issues
```python
if len(df) < 30:
    st.warning("⚠️ Sample size < 30: Results may be unreliable")

if df.isnull().sum().sum() > 0:
    st.warning(f"⚠️ Missing values detected: {df.isnull().sum().sum()} cells")

if df.select_dtypes('object').shape[1] > 0:
    st.warning("⚠️ Categorical columns detected: May need encoding for ML methods")
```

---

## 10. Testing Checklist

- [ ] All new methods load without errors
- [ ] Interactive sliders update results in real-time
- [ ] Tutorials display correctly
- [ ] Visualizations render properly
- [ ] Help system accessible from all tabs
- [ ] Export functionality works
- [ ] Performance acceptable for large datasets
- [ ] Error messages clear and actionable

---

## Next Steps

1. **Phase 1**: Integrate ML module (clustering, anomaly detection)
2. **Phase 2**: Integrate statistical tests and distribution fitting
3. **Phase 3**: Integrate time series (FFT, wavelets)
4. **Phase 4**: Add help system and tutorials
5. **Phase 5**: Optimize UI/UX and add interactive controls

---

*This guide provides starting points. Adapt based on your specific UI design and user preferences.*
