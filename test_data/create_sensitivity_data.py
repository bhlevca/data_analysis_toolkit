"""
Create test data for sensitivity analysis.

Generates data with known parameter sensitivities for testing:
1. Linear model data (known coefficients)
2. Nonlinear model data (with interactions)
3. Ishigami function data (standard SA benchmark)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent


def create_linear_sensitivity_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Create data from a linear model with known sensitivities.
    
    Model: y = 3*x1 + 2*x2 - 1*x3 + 0.5*x4 + 0.1*x5 + noise
    
    Expected sensitivity ranking: x1 > x2 > x3 > x4 > x5
    """
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)
    x3 = np.random.uniform(0, 10, n_samples)
    x4 = np.random.uniform(0, 10, n_samples)
    x5 = np.random.uniform(0, 10, n_samples)
    
    # Known coefficients
    y = 3*x1 + 2*x2 - 1*x3 + 0.5*x4 + 0.1*x5 + np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
        'y': y
    })


def create_nonlinear_sensitivity_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Create data with nonlinear effects and interactions.
    
    Model: y = x1^2 + x2*x3 + sin(x4) + x5 + noise
    
    x1 has quadratic effect
    x2 and x3 interact
    x4 has periodic effect
    x5 is linear
    """
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(0, 5, n_samples)
    x3 = np.random.uniform(0, 5, n_samples)
    x4 = np.random.uniform(0, 2*np.pi, n_samples)
    x5 = np.random.uniform(0, 10, n_samples)
    
    y = x1**2 + x2*x3 + np.sin(x4) + x5 + np.random.normal(0, 0.5, n_samples)
    
    return pd.DataFrame({
        'x1_quadratic': x1,
        'x2_interact': x2,
        'x3_interact': x3,
        'x4_periodic': x4,
        'x5_linear': x5,
        'y': y
    })


def create_ishigami_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create data from the Ishigami function - standard SA benchmark.
    
    f(x) = sin(x1) + 7*sin(x2)^2 + 0.1*x3^4*sin(x1)
    
    Known analytical Sobol indices (for a=7, b=0.1):
    S1 = [0.3139, 0.4424, 0.0000]
    ST = [0.5576, 0.4424, 0.2437]
    """
    x1 = np.random.uniform(-np.pi, np.pi, n_samples)
    x2 = np.random.uniform(-np.pi, np.pi, n_samples)
    x3 = np.random.uniform(-np.pi, np.pi, n_samples)
    
    a, b = 7, 0.1
    y = np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)
    
    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3,
        'ishigami_output': y
    })


def main():
    print("🔧 Creating sensitivity analysis test data...\n")
    
    # 1. Linear sensitivity data
    df = create_linear_sensitivity_data()
    path = OUTPUT_DIR / 'sensitivity_linear.csv'
    df.to_csv(path, index=False)
    print(f"✅ {path.name}: {len(df)} rows (linear model with known coefficients)")
    
    # 2. Nonlinear sensitivity data
    df = create_nonlinear_sensitivity_data()
    path = OUTPUT_DIR / 'sensitivity_nonlinear.csv'
    df.to_csv(path, index=False)
    print(f"✅ {path.name}: {len(df)} rows (nonlinear with interactions)")
    
    # 3. Ishigami benchmark data
    df = create_ishigami_data()
    path = OUTPUT_DIR / 'sensitivity_ishigami.csv'
    df.to_csv(path, index=False)
    print(f"✅ {path.name}: {len(df)} rows (Ishigami function benchmark)")
    
    print("\n📁 All sensitivity analysis test data created!")
    print("\nExpected results:")
    print("  - Linear: x1 > x2 > x3 > x4 > x5 (by coefficient magnitude)")
    print("  - Nonlinear: High sigma for x1 (quadratic), x2/x3 (interaction)")
    print("  - Ishigami: Known analytical Sobol indices for validation")


if __name__ == '__main__':
    main()
