# Bayesian Prior Specification Files

This directory contains JSON files that encode **prior beliefs** for Bayesian
regression.  Each file can be loaded into the Bayesian Analysis module via
`BayesianAnalysis.load_prior("path/to/prior.json")` or selected in the
Streamlit UI.

## File Format

```json
{
  "name":               "Human-readable label",
  "description":        "Why this prior was chosen",
  "features":           ["predictor_1", "predictor_2", "predictor_3"],
  "target":             "response_homoscedastic",
  "prior_mean":         [5.0, 2.0, -1.5, 0.8],
  "prior_precision":    1.0,
  "noise_var_estimate": 4.0
}
```

| Field | Meaning |
|-------|---------|
| `prior_mean` | Expected coefficient values **[intercept, β₁, β₂, …]** before seeing data |
| `prior_precision` | Scalar controlling how tightly the prior constrains the coefficients (higher = stronger prior) |
| `noise_var_estimate` | Fixed noise variance σ². Use `null` to let the algorithm estimate it from the data (empirical Bayes) |

## Provided Presets

| File | Precision | Description |
|------|-----------|-------------|
| `weakly_informative.json` | 0.001 | Vague, lets data dominate |
| `informative_correct.json` | 1.0 | Matches the true data-generating coefficients |
| `strong_wrong.json` | 100.0 | Deliberately wrong — shows prior–data conflict |
| `domain_moderate.json` | 0.5 | Approximate domain knowledge |

## Creating Your Own

1. Copy any JSON file above as a template.
2. Set `prior_mean` to your expected coefficient values (intercept first).
3. Choose `prior_precision`:
   - **< 0.01** — very weak, data-driven
   - **0.01 – 1.0** — moderate
   - **> 1.0** — strong, prior-driven
4. Set `noise_var_estimate` to a known value or `null`.
5. Save as `.json` in this directory (or anywhere — pass the path to `load_prior`).

## Usage in Code

```python
from data_toolkit import BayesianAnalysis

ba = BayesianAnalysis(df)
prior = ba.load_prior("test_data/priors/informative_correct.json")

results = ba.bayesian_regression(
    features=prior['features'],
    target=prior['target'],
    prior_mean=prior['prior_mean'],
    prior_precision=prior['prior_precision'],
    noise_var_estimate=prior['noise_var_estimate'],
)
```
