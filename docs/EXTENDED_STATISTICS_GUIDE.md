# Extended Statistics Guide

This guide covers the extended statistical tests and distribution operations available in the toolkit.

## Overview

The `extended_statistics` module provides additional statistical tests and distribution operations that complement the main `statistical_analysis` module:

### Statistical Tests
- **Kolmogorov-Smirnov** - Goodness-of-fit and two-sample distribution comparison
- **Anderson-Darling** - More sensitive normality test (focuses on tails)
- **Runs Test** - Test for randomness in sequences
- **Sign Test** - Non-parametric paired comparison
- **Mood's Median Test** - Non-parametric test for equal medians
- **Friedman Test** - Non-parametric repeated measures ANOVA
- **Bartlett's Test** - Test for equal variances (assumes normality)
- **Brown-Forsythe Test** - Robust test for equal variances

### Distribution Operations
- **Kernel Density Estimation (KDE)** - Non-parametric density estimation
- **Percentile/Quantile Calculations** - Flexible quantile computation
- **Moment Calculations** - Raw, central, and standardized moments
- **Entropy Measures** - Shannon entropy and related measures
- **Distribution Sampling** - Sample from fitted distributions
- **Probability Calculations** - CDF, PDF, and survival function values

---

## Installation

The module is part of the data_toolkit package:

```python
from data_toolkit import ExtendedStatisticalTests, DistributionOperations
```

---

## Statistical Tests

### Kolmogorov-Smirnov Tests

#### One-Sample KS Test
Tests if data follows a specified distribution.

```python
import pandas as pd
import numpy as np
from data_toolkit import ExtendedStatisticalTests

# Create test data
np.random.seed(42)
df = pd.DataFrame({
    'measurements': np.random.normal(100, 15, 200)
})

# Initialize and run test
est = ExtendedStatisticalTests(df)
result = est.kolmogorov_smirnov_1sample('measurements', 'norm')

print(f"KS Statistic: {result['statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Interpretation: {result['interpretation']}")
```

#### Two-Sample KS Test
Tests if two samples come from the same distribution.

```python
df = pd.DataFrame({
    'group_a': np.random.normal(100, 15, 100),
    'group_b': np.random.exponential(50, 100)
})

est = ExtendedStatisticalTests(df)
result = est.kolmogorov_smirnov_2sample('group_a', 'group_b')

print(f"Different distributions: {result['reject_null']}")
```

---

### Anderson-Darling Test

More sensitive to distribution tails than KS test. Best for normality testing.

```python
result = est.anderson_darling('measurements', 'norm')

print(f"Statistic: {result['statistic']:.4f}")
print(f"Critical values: {result['critical_values']}")
print(f"Reject at significance levels: {result['reject_at_levels']}")
```

---

### Runs Test for Randomness

Tests if a sequence shows random behavior or patterns.

```python
df = pd.DataFrame({
    'sequence': [1, 2, 3, 2, 1, 3, 2, 4, 1, 5, 2, 3, 1, 4, 2]
})

est = ExtendedStatisticalTests(df)
result = est.runs_test('sequence', cutoff='median')

print(f"Number of runs: {result['n_runs']}")
print(f"Expected runs: {result['expected_runs']:.2f}")
print(f"Random sequence: {not result['reject_null']}")
```

---

### Sign Test

Non-parametric test for paired data. Tests if median difference is zero.

```python
df = pd.DataFrame({
    'before_treatment': np.random.normal(100, 10, 50),
    'after_treatment': np.random.normal(100, 10, 50) + 5  # Treatment effect
})

est = ExtendedStatisticalTests(df)
result = est.sign_test('after_treatment', 'before_treatment')

print(f"Positive differences: {result['n_positive']}")
print(f"Negative differences: {result['n_negative']}")
print(f"Significant effect: {result['reject_null']}")
```

---

### Mood's Median Test

Non-parametric test for comparing medians of multiple groups.

```python
df = pd.DataFrame({
    'group_1': np.random.normal(100, 15, 50),
    'group_2': np.random.normal(105, 15, 50),
    'group_3': np.random.normal(110, 15, 50)
})

est = ExtendedStatisticalTests(df)
result = est.mood_median_test('group_1', 'group_2', 'group_3')

print(f"Grand median: {result['grand_median']:.2f}")
print(f"Medians differ: {result['reject_null']}")
```

---

### Friedman Test

Non-parametric alternative to repeated measures ANOVA.

```python
# Repeated measures design
df = pd.DataFrame({
    'baseline': np.random.normal(100, 10, 30),
    'week_1': np.random.normal(105, 10, 30),
    'week_2': np.random.normal(110, 10, 30),
    'week_3': np.random.normal(108, 10, 30)
})

est = ExtendedStatisticalTests(df)
result = est.friedman_test(['baseline', 'week_1', 'week_2', 'week_3'])

print(f"Chi-square statistic: {result['statistic']:.4f}")
print(f"Time effect: {result['reject_null']}")
```

---

### Variance Tests

#### Bartlett's Test
Tests for equal variances. **Assumes normality**.

```python
df = pd.DataFrame({
    'low_var': np.random.normal(100, 5, 100),
    'high_var': np.random.normal(100, 20, 100)
})

est = ExtendedStatisticalTests(df)
result = est.bartlett_test('low_var', 'high_var')

print(f"Variances differ: {result['reject_null']}")
```

#### Brown-Forsythe Test
Robust to non-normality (uses median instead of mean).

```python
result = est.brown_forsythe_test('low_var', 'high_var')
print(f"Variances differ (robust): {result['reject_null']}")
```

---

## Distribution Operations

### Kernel Density Estimation

Non-parametric estimation of the probability density function.

```python
from data_toolkit import DistributionOperations

df = pd.DataFrame({
    'values': np.concatenate([
        np.random.normal(50, 10, 100),
        np.random.normal(80, 5, 50)  # Bimodal
    ])
})

do = DistributionOperations(df)
result = do.kernel_density_estimation('values', n_points=200)

# Plot the KDE
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(result['x'], result['density'], 'b-', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title(f"KDE (bandwidth={result['bandwidth']:.3f}, mode≈{result['mode_estimate']:.1f})")
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Percentile Calculations

Flexible percentile/quantile computation.

```python
# Default percentiles
result = do.percentiles('values')
print(result)
# Output: {'p1': 32.5, 'p5': 38.2, 'p10': 42.1, 'p25': 47.8, 
#          'p50': 55.3, 'p75': 70.1, 'p90': 81.2, 'p95': 85.4, 'p99': 91.8}

# Custom percentiles
result = do.percentiles('values', percentiles=[10, 25, 50, 75, 90])
print(f"IQR: {result['p75'] - result['p25']:.2f}")
```

---

### Moment Calculations

Calculate raw, central, and standardized moments.

```python
result = do.moments('values')

print("Raw moments:")
print(f"  Mean (m1): {result['raw_moments']['m1']:.2f}")

print("\nCentral moments:")
print(f"  Variance (μ2): {result['central_moments']['mu2']:.2f}")

print("\nStandardized moments:")
print(f"  Skewness: {result['standardized_moments']['skewness']:.4f}")
print(f"  Kurtosis: {result['standardized_moments']['kurtosis']:.4f}")
print(f"  Excess kurtosis: {result['standardized_moments']['excess_kurtosis']:.4f}")
```

**Interpretation:**
- Skewness = 0: Symmetric
- Skewness > 0: Right-skewed (tail to the right)
- Skewness < 0: Left-skewed (tail to the left)
- Excess kurtosis = 0: Normal-like tails
- Excess kurtosis > 0: Heavy tails (leptokurtic)
- Excess kurtosis < 0: Light tails (platykurtic)

---

### Entropy Calculations

Measure uncertainty/information content in the data.

```python
result = do.entropy('values', bins=50)

print(f"Shannon entropy: {result['shannon_entropy']:.4f} nats")
print(f"Normalized entropy: {result['normalized_entropy']:.4f}")
print(f"Negentropy: {result['negentropy']:.4f}")
```

**Interpretation:**
- Higher entropy = more uncertainty/randomness
- Normalized entropy: 0 = deterministic, 1 = maximum randomness
- Negentropy: departure from Gaussianity (higher = less Gaussian)

---

### Distribution Sampling

Generate random samples from a distribution fitted to your data.

```python
# Fit distribution to data and generate new samples
result = do.distribution_sampling('values', 'norm', n_samples=1000)

print(f"Generated {result['generated_n']} samples from {result['distribution']}")
print(f"Fitted parameters (loc, scale): {result['parameters']}")

# Access samples
samples = result['samples']
```

Available distributions: `'norm'`, `'expon'`, `'gamma'`, `'beta'`, `'lognorm'`, `'uniform'`, etc.

---

### Probability Calculations

Calculate CDF, PDF, and survival function values.

```python
result = do.probability_calculations('values', 'norm', values=[40, 60, 80])

for calc in result['calculations']:
    print(f"\nValue: {calc['value']:.1f}")
    print(f"  P(X ≤ {calc['value']:.1f}) = {calc['cdf']:.4f}")
    print(f"  P(X > {calc['value']:.1f}) = {calc['sf']:.4f}")
    print(f"  Density f({calc['value']:.1f}) = {calc['pdf']:.4f}")
```

---

### KDE Comparison Plot

Compare distributions across multiple columns.

```python
df = pd.DataFrame({
    'control': np.random.normal(100, 15, 100),
    'treatment_a': np.random.normal(110, 12, 100),
    'treatment_b': np.random.normal(105, 18, 100)
})

do = DistributionOperations(df)
fig = do.plot_kde_comparison(['control', 'treatment_a', 'treatment_b'])
plt.show()
```

---

## Best Practices

### Choosing the Right Test

| Situation | Recommended Test |
|-----------|------------------|
| Test normality | Anderson-Darling (more sensitive) or KS test |
| Compare two distributions | Two-sample KS test |
| Paired data (non-parametric) | Sign test or Wilcoxon |
| Test for randomness | Runs test |
| Compare medians (multiple groups) | Mood's median test |
| Repeated measures (non-parametric) | Friedman test |
| Test equal variances (normal data) | Bartlett's test |
| Test equal variances (any distribution) | Brown-Forsythe test |

### Sample Size Recommendations

- **KS test**: n ≥ 20 for reliable results
- **Anderson-Darling**: n ≥ 8 minimum, n ≥ 20 recommended
- **Sign test**: n ≥ 10 paired observations
- **Friedman test**: n ≥ 10 subjects per treatment

### Common Pitfalls

1. **Multiple testing**: When running many tests, adjust α for multiple comparisons
2. **KS sensitivity**: KS test is less sensitive in the tails; use Anderson-Darling for normality
3. **Bartlett's assumption**: Don't use Bartlett's test if data is non-normal
4. **KDE bandwidth**: Default bandwidth may over- or under-smooth; try different values

---

## Complete Example

```python
import pandas as pd
import numpy as np
from data_toolkit import ExtendedStatisticalTests, DistributionOperations

# Load or create data
np.random.seed(42)
df = pd.DataFrame({
    'control': np.random.normal(100, 15, 100),
    'treatment': np.random.normal(110, 12, 100)
})

# Statistical Tests
est = ExtendedStatisticalTests(df)

# 1. Check normality
print("=== Normality Tests ===")
for col in ['control', 'treatment']:
    result = est.anderson_darling(col, 'norm')
    print(f"{col}: {result['interpretation']}")

# 2. Compare variances
print("\n=== Variance Test ===")
result = est.brown_forsythe_test('control', 'treatment')
print(f"Equal variances: {not result['reject_null']}")

# 3. Compare distributions
print("\n=== Distribution Comparison ===")
result = est.kolmogorov_smirnov_2sample('control', 'treatment')
print(f"Same distribution: {not result['reject_null']}")

# Distribution Operations
do = DistributionOperations(df)

# 4. Compare central tendency and spread
print("\n=== Distribution Characteristics ===")
for col in ['control', 'treatment']:
    pct = do.percentiles(col, [25, 50, 75])
    mom = do.moments(col)
    print(f"\n{col}:")
    print(f"  Median: {pct['p50']:.1f}")
    print(f"  IQR: {pct['p75'] - pct['p25']:.1f}")
    print(f"  Skewness: {mom['standardized_moments']['skewness']:.3f}")
```

---

## API Reference

### ExtendedStatisticalTests

| Method | Description |
|--------|-------------|
| `kolmogorov_smirnov_1sample(column, dist)` | 1-sample KS test |
| `kolmogorov_smirnov_2sample(col1, col2)` | 2-sample KS test |
| `anderson_darling(column, dist)` | A-D goodness of fit |
| `runs_test(column, cutoff)` | Test for randomness |
| `sign_test(col1, col2)` | Paired sign test |
| `mood_median_test(*columns)` | Multi-group median test |
| `friedman_test(columns)` | Repeated measures |
| `bartlett_test(*columns)` | Equal variances (normal) |
| `brown_forsythe_test(*columns)` | Equal variances (robust) |

### DistributionOperations

| Method | Description |
|--------|-------------|
| `kernel_density_estimation(col, bw, n_points)` | KDE estimation |
| `percentiles(col, percentiles)` | Quantile calculation |
| `moments(col)` | Raw/central/standardized moments |
| `entropy(col, bins)` | Entropy measures |
| `distribution_sampling(col, dist, n)` | Sample from fitted dist |
| `probability_calculations(col, dist, values)` | CDF/PDF/SF values |
| `plot_kde_comparison(columns)` | Multi-KDE plot |

---

## See Also

- [Statistical Analysis Guide](docs/ANALYSIS_GUIDE.md) - Main statistical methods
- [Sensitivity Analysis Guide](docs/SENSITIVITY_ANALYSIS_GUIDE.md) - Parameter sensitivity
- [Tutorial](docs/TUTORIAL.md) - Complete walkthrough
