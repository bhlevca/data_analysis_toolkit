#!/usr/bin/env python3
"""
Rust vs Python Backend Performance Benchmark
=============================================

This script demonstrates the speed difference between the Rust-accelerated
backend and the pure Python backend for computationally intensive operations.

Requirements:
- Run with Rust extensions compiled for full comparison
- If Rust not compiled, only Python times will be shown

Usage:
    python benchmark_rust_vs_python.py
"""

import numpy as np
import pandas as pd
import time
import sys
from typing import Callable, Tuple

# Add the package to path if running from test_data directory
sys.path.insert(0, '../src')

try:
    from data_toolkit.rust_accelerated import (
        AccelerationSettings,
        is_rust_available,
        distance_correlation,
        bootstrap_linear_regression,
        monte_carlo_predictions,
        lead_lag_correlations,
        rolling_statistics,
        detect_outliers_iqr,
        mutual_information,
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run: pip install -e . from the project root")
    IMPORTS_OK = False
    sys.exit(1)


def timer(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Time a function execution and return (elapsed_time, result)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def format_time(seconds: float) -> str:
    """Format time nicely"""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_result(name: str, python_time: float, rust_time: float = None):
    """Print benchmark result with speedup"""
    if rust_time is not None and rust_time > 0:
        speedup = python_time / rust_time
        bar_len = min(int(speedup * 2), 50)
        bar = "█" * bar_len
        print(f"\n{name}:")
        print(f"  Python: {format_time(python_time):>12}")
        print(f"  Rust:   {format_time(rust_time):>12}")
        print(f"  Speedup: {speedup:>10.1f}x  {bar}")
    else:
        print(f"\n{name}:")
        print(f"  Python: {format_time(python_time):>12}")
        print(f"  Rust:   {'N/A':>12} (not compiled)")


def benchmark_distance_correlation(n_samples: int = 2000):
    """Benchmark distance correlation calculation"""
    print_header(f"Distance Correlation (n={n_samples})")
    
    # Generate data with non-linear relationship
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n_samples)
    y = x**2 + np.random.normal(0, 1, n_samples)  # Quadratic relationship
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(distance_correlation, x, y)
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(distance_correlation, x, y)
        
        # Verify results match
        assert abs(python_result - rust_result) < 0.001, "Results don't match!"
    else:
        rust_time = None
    
    print_result("Distance Correlation", python_time, rust_time)
    print(f"  Result: {python_result:.4f}")
    
    return python_time, rust_time


def benchmark_bootstrap(n_samples: int = 1000, n_features: int = 5, n_bootstrap: int = 2000):
    """Benchmark bootstrap confidence intervals"""
    print_header(f"Bootstrap CI (n={n_samples}, features={n_features}, iterations={n_bootstrap})")
    
    # Generate regression data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.0, -1.5, 1.0, -0.5, 0.3])
    y = X @ true_coef + np.random.normal(0, 1, n_samples)
    
    confidence = 0.95
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(
        bootstrap_linear_regression, X, y, n_bootstrap, confidence
    )
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(
            bootstrap_linear_regression, X, y, n_bootstrap, confidence
        )
    else:
        rust_time = None
    
    print_result("Bootstrap CI", python_time, rust_time)
    
    return python_time, rust_time


def benchmark_monte_carlo(n_samples: int = 500, n_features: int = 3, n_simulations: int = 5000):
    """Benchmark Monte Carlo predictions"""
    print_header(f"Monte Carlo Simulation (n={n_samples}, simulations={n_simulations})")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    coefficients = np.array([1.5, -0.8, 0.5])
    intercept = 2.0
    residual_std = 1.0
    confidence = 0.95
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(
        monte_carlo_predictions, X, coefficients, intercept, 
        residual_std, n_simulations, confidence
    )
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(
            monte_carlo_predictions, X, coefficients, intercept,
            residual_std, n_simulations, confidence
        )
    else:
        rust_time = None
    
    print_result("Monte Carlo", python_time, rust_time)
    
    return python_time, rust_time


def benchmark_lead_lag(n_samples: int = 5000, max_lag: int = 20):
    """Benchmark lead-lag correlation analysis"""
    print_header(f"Lead-Lag Correlations (n={n_samples}, max_lag={max_lag})")
    
    # Generate time series with lead-lag relationship
    np.random.seed(42)
    base = np.cumsum(np.random.randn(n_samples + max_lag))
    x = base[:n_samples] + np.random.normal(0, 0.5, n_samples)
    y = base[5:n_samples+5] + np.random.normal(0, 0.5, n_samples)  # y lags x by 5
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(lead_lag_correlations, x, y, max_lag)
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(lead_lag_correlations, x, y, max_lag)
    else:
        rust_time = None
    
    print_result("Lead-Lag Correlations", python_time, rust_time)
    
    return python_time, rust_time


def benchmark_rolling_stats(n_samples: int = 100000, window: int = 50):
    """Benchmark rolling statistics"""
    print_header(f"Rolling Statistics (n={n_samples}, window={window})")
    
    # Generate time series
    np.random.seed(42)
    data = np.cumsum(np.random.randn(n_samples))
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(rolling_statistics, data, window)
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(rolling_statistics, data, window)
    else:
        rust_time = None
    
    print_result("Rolling Statistics", python_time, rust_time)
    
    return python_time, rust_time


def benchmark_outliers(n_samples: int = 50000, n_columns: int = 10):
    """Benchmark outlier detection"""
    print_header(f"Outlier Detection (n={n_samples}, columns={n_columns})")
    
    # Generate data with outliers
    np.random.seed(42)
    data = np.random.randn(n_samples, n_columns)
    # Add some outliers
    outlier_idx = np.random.choice(n_samples, 500, replace=False)
    data[outlier_idx, :] *= 5
    
    # Python backend
    AccelerationSettings.set_use_rust(False)
    python_time, python_result = timer(detect_outliers_iqr, data, 1.5)
    
    # Rust backend
    if is_rust_available():
        AccelerationSettings.set_use_rust(True)
        rust_time, rust_result = timer(detect_outliers_iqr, data, 1.5)
    else:
        rust_time = None
    
    print_result("Outlier Detection (IQR)", python_time, rust_time)
    
    return python_time, rust_time


def run_all_benchmarks():
    """Run all benchmarks and show summary"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " RUST vs PYTHON BACKEND PERFORMANCE BENCHMARK ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    rust_available = is_rust_available()
    print(f"\nRust Extensions: {'✅ AVAILABLE' if rust_available else '❌ NOT COMPILED'}")
    
    if not rust_available:
        print("\n⚠️  To see full comparison, compile Rust extensions:")
        print("    cd rust_extensions")
        print("    pip install maturin")
        print("    maturin develop --release")
    
    # Run benchmarks
    results = []
    
    results.append(("Distance Correlation", *benchmark_distance_correlation(2000)))
    results.append(("Bootstrap CI", *benchmark_bootstrap(1000, 5, 2000)))
    results.append(("Monte Carlo", *benchmark_monte_carlo(500, 3, 5000)))
    results.append(("Lead-Lag Correlations", *benchmark_lead_lag(5000, 20)))
    results.append(("Rolling Statistics", *benchmark_rolling_stats(100000, 50)))
    results.append(("Outlier Detection", *benchmark_outliers(50000, 10)))
    
    # Summary
    print_header("SUMMARY")
    
    print(f"\n{'Operation':<25} {'Python':>12} {'Rust':>12} {'Speedup':>12}")
    print("-" * 65)
    
    total_python = 0
    total_rust = 0
    
    for name, python_time, rust_time in results:
        total_python += python_time
        if rust_time:
            total_rust += rust_time
            speedup = f"{python_time/rust_time:.1f}x"
        else:
            speedup = "N/A"
        
        print(f"{name:<25} {format_time(python_time):>12} {format_time(rust_time) if rust_time else 'N/A':>12} {speedup:>12}")
    
    print("-" * 65)
    
    if rust_available and total_rust > 0:
        overall_speedup = total_python / total_rust
        print(f"{'TOTAL':<25} {format_time(total_python):>12} {format_time(total_rust):>12} {overall_speedup:.1f}x")
        
        print("\n" + "="*65)
        print(f"  🚀 OVERALL SPEEDUP: {overall_speedup:.1f}x faster with Rust!")
        print("="*65)
        
        # Visual comparison
        print("\n  Time Comparison (shorter = faster):")
        print(f"\n  Python: {'█' * 50} {format_time(total_python)}")
        rust_bar_len = max(1, int(50 * total_rust / total_python))
        print(f"  Rust:   {'█' * rust_bar_len} {format_time(total_rust)}")
    else:
        print(f"\n{'TOTAL':<25} {format_time(total_python):>12}")
        print("\n  Compile Rust extensions to see speedup comparison!")
    
    print("\n")


if __name__ == "__main__":
    run_all_benchmarks()
