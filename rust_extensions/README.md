# Building Rust Extensions

This guide explains how to build and use the high-performance Rust extensions.

## Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Install maturin** (Python-Rust build tool):
   ```bash
   pip install maturin
   ```

## Building the Extension

```bash
cd rust_extensions

# Development build (faster, for testing)
maturin develop

# Release build (optimized, for production)
maturin develop --release
```

## Verifying Installation

```python
from data_toolkit.rust_accelerated import is_rust_available, benchmark_rust_vs_python

# Check if Rust is available
print(f"Rust available: {is_rust_available()}")

# Run benchmarks
results = benchmark_rust_vs_python(n_samples=10000)
for key, value in results.items():
    if 'speedup' in key:
        print(f"{key}: {value:.1f}x faster")
```

## Usage Examples

### Distance Correlation (detects non-linear relationships)

```python
from data_toolkit.rust_accelerated import distance_correlation
import numpy as np

x = np.random.randn(1000)
y = x ** 2 + np.random.randn(1000) * 0.1  # Non-linear relationship

# Pearson correlation would be ~0, but distance correlation detects it
dc = distance_correlation(x, y)
print(f"Distance correlation: {dc:.4f}")  # ~0.5+
```

### Bootstrap Confidence Intervals

```python
from data_toolkit.rust_accelerated import bootstrap_linear_regression
import numpy as np

X = np.random.randn(500, 3)
y = X @ [2, -1, 0.5] + np.random.randn(500) * 0.5

mean_coefs, ci_lower, ci_upper = bootstrap_linear_regression(
    X, y, 
    n_bootstrap=2000, 
    confidence=0.95
)

for i, (m, lo, hi) in enumerate(zip(mean_coefs, ci_lower, ci_upper)):
    print(f"β{i}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
```

### Transfer Entropy (directed information flow)

```python
from data_toolkit.rust_accelerated import transfer_entropy
import numpy as np

# X causes Y with a lag
x = np.random.randn(1000)
y = np.zeros(1000)
y[1:] = 0.8 * x[:-1] + np.random.randn(999) * 0.2

te_x_to_y = transfer_entropy(x, y, n_bins=10, lag=1)
te_y_to_x = transfer_entropy(y, x, n_bins=10, lag=1)

print(f"X → Y: {te_x_to_y:.4f}")  # Higher
print(f"Y → X: {te_y_to_x:.4f}")  # Lower
```

### Lead-Lag Analysis

```python
from data_toolkit.rust_accelerated import lead_lag_correlations
import numpy as np

x = np.sin(np.linspace(0, 10*np.pi, 500))
y = np.roll(x, 20) + np.random.randn(500) * 0.1  # y lags x by 20

lags, correlations = lead_lag_correlations(x, y, max_lag=50)
best_lag = lags[np.argmax(correlations)]
print(f"Best lag: {best_lag}")  # Should be ~20
```

## Performance Comparison

Typical speedups on 10,000 samples:

| Function | Python | Rust | Speedup |
|----------|--------|------|---------|
| distance_correlation | 2500ms | 80ms | **31x** |
| bootstrap (1000 iter) | 1200ms | 95ms | **13x** |
| lead_lag (±50 lags) | 180ms | 12ms | **15x** |
| transfer_entropy | 450ms | 35ms | **13x** |
| rolling_statistics | 25ms | 3ms | **8x** |

## Fallback Behavior

If Rust extensions aren't installed, pure Python implementations are used automatically:

```python
from data_toolkit.rust_accelerated import distance_correlation

# Works regardless of whether Rust is installed
# Just slower without Rust
result = distance_correlation(x, y)
```

## Troubleshooting

### "Rust extensions not available" warning

This means the Rust extension isn't compiled. Run:
```bash
cd rust_extensions
maturin develop --release
```

### Compilation errors

Make sure you have:
- Rust toolchain installed (`rustc --version`)
- Python development headers (`python3-dev` on Ubuntu)
- A C compiler (`gcc` or `clang`)

### Import errors after building

Try rebuilding with:
```bash
pip uninstall data_toolkit_rust
cd rust_extensions
maturin develop --release
```
