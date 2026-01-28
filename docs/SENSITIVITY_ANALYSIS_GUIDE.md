# Sensitivity Analysis Guide

## Overview

Sensitivity analysis helps understand how variations in model inputs affect outputs. This is essential for:

- **Factor Prioritization**: Identify which parameters most influence results
- **Model Simplification**: Fix unimportant parameters to reduce complexity
- **Uncertainty Analysis**: Focus uncertainty quantification on important factors
- **Experimental Design**: Determine where to invest measurement effort

## Available Methods

### 1. Morris Screening (Elementary Effects)

A computationally efficient screening method for identifying important factors.

```python
from data_toolkit import SensitivityAnalysis

# Define model function
def my_model(x):
    return 2*x[0] + 3*x[1] - x[2]**2

# Create analyzer
sa = SensitivityAnalysis(my_model)

# Define parameter bounds
bounds = {
    'param1': (0, 10),
    'param2': (0, 5),
    'param3': (-3, 3)
}

# Run Morris screening
results = sa.morris_screening(bounds, n_trajectories=20, seed=42)

# View results
print(f"Ranking: {results['ranking']}")
print(f"μ* values: {results['mu_star']}")
print(f"σ values: {results['sigma']}")
print(f"Classification: {results['classification']}")
```

**Interpretation:**
- **μ* (mu-star)**: Mean of absolute elementary effects - indicates overall importance
- **σ (sigma)**: Standard deviation - indicates nonlinearity or interactions
- High μ* + Low σ/μ* → Linear effect
- High μ* + High σ/μ* → Nonlinear or interactive effect
- Low μ* → Negligible factor

### 2. Sobol Sensitivity Indices

Variance-based method that decomposes output variance into contributions from inputs.

```python
# Run Sobol analysis
results = sa.sobol_indices(bounds, n_samples=500, seed=42)

print(f"First-order (S1): {results['S1']}")
print(f"Total-order (ST): {results['ST']}")
```

**Interpretation:**
- **S1 (First-order)**: Direct contribution of each parameter
- **ST (Total-order)**: Total contribution including all interactions
- ST >> S1 → Parameter has strong interaction effects
- Sum of S1 ≈ 1 for additive models

### 3. One-At-a-Time (OAT) Analysis

Simple local sensitivity by varying each parameter individually.

```python
# Run OAT analysis
results = sa.one_at_a_time(bounds, n_steps=20)

print(f"Gradients: {results['gradients']}")
print(f"Elasticities: {results['elasticities']}")
```

**Interpretation:**
- **Gradient**: Rate of change of output with respect to input
- **Elasticity**: Normalized sensitivity (% change output / % change input)

## DataFrame Convenience Function

For quick analysis on tabular data:

```python
from data_toolkit import analyze_dataframe_sensitivity
import pandas as pd

# Load data
df = pd.read_csv('my_data.csv')

# Run Morris screening
results = analyze_dataframe_sensitivity(
    df, 
    target='output_column',
    features=['x1', 'x2', 'x3'],  # Optional - auto-detected if None
    method='morris',  # or 'sobol', 'oat'
    n_samples=50,
    seed=42
)

print(f"Important features: {results['ranking']}")
```

## Visualization

```python
# Morris plot (μ* vs σ)
fig = sa.plot_morris()
fig.savefig('morris_plot.png')

# Sobol bar chart
fig = sa.plot_sobol()
fig.savefig('sobol_plot.png')

# OAT sweep plots
fig = sa.plot_oat()
fig.savefig('oat_plot.png')

# Summary table
summary = sa.summary_table()
print(summary)
```

## Using with Streamlit GUI

1. Load your data in the **Data** tab
2. Go to **Statistics → Sensitivity Analysis**
3. Select target variable and input parameters
4. Choose analysis method
5. Click "Run Sensitivity Analysis"
6. View interactive plots and ranking

## Test Data

Three test datasets are provided:

| File | Description | Use Case |
|------|-------------|----------|
| `sensitivity_linear.csv` | Linear model y = 3x₁ + 2x₂ - x₃ + 0.5x₄ + 0.1x₅ | Validate ranking matches coefficients |
| `sensitivity_nonlinear.csv` | y = x₁² + x₂x₃ + sin(x₄) + x₅ | Test detection of nonlinearity/interactions |
| `sensitivity_ishigami.csv` | Ishigami function (SA benchmark) | Validate against known analytical indices |

## Best Practices

1. **Start with Morris**: Fast screening to identify potentially important factors
2. **Use Sobol for important factors**: More accurate but computationally expensive
3. **Increase samples for precision**: More trajectories/samples = more accurate results
4. **Check for interactions**: Large ST-S1 gap indicates interaction effects
5. **Consider bounds carefully**: Results depend on parameter ranges

## References

- Morris, M.D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. *Technometrics*, 33(2), 161-174.
- Saltelli, A. et al. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*, 181(2), 259-270.
- Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large models. *Environmental Modelling & Software*, 22(10), 1509-1518.
