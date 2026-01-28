"""
Expand test data files that have insufficient rows.

Based on Analysis Report findings:
- ANOVA files need 100+ rows for statistical significance
- Time series files need 200+ data points for proper decomposition
- ML prediction files need 100+ samples
- Signal analysis needs multivariate signals

Usage:
    python test_data/expand_test_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent


def expand_oneway_anova(n_per_group: int = 50) -> pd.DataFrame:
    """Create one-way ANOVA data with 3 groups, 50 samples each = 150 rows."""
    groups = ['Control', 'Treatment_A', 'Treatment_B']
    effects = {'Control': 50, 'Treatment_A': 55, 'Treatment_B': 60}
    
    data = []
    for group in groups:
        for _ in range(n_per_group):
            value = effects[group] + np.random.normal(0, 5)
            data.append({'group': group, 'value': value, 'subject_id': len(data) + 1})
    
    return pd.DataFrame(data)


def expand_twoway_anova(n_per_cell: int = 25) -> pd.DataFrame:
    """Create two-way ANOVA data with 2x2 design, 25 per cell = 100 rows."""
    factors_A = ['Low', 'High']
    factors_B = ['Control', 'Treatment']
    
    # Effects
    base = 50
    effect_A = {'Low': 0, 'High': 10}
    effect_B = {'Control': 0, 'Treatment': 8}
    interaction = {('Low', 'Control'): 0, ('Low', 'Treatment'): 0,
                   ('High', 'Control'): 0, ('High', 'Treatment'): 5}  # synergy
    
    data = []
    subject = 0
    for a in factors_A:
        for b in factors_B:
            for _ in range(n_per_cell):
                subject += 1
                value = (base + effect_A[a] + effect_B[b] + 
                        interaction[(a, b)] + np.random.normal(0, 4))
                data.append({
                    'factor_A': a, 'factor_B': b, 
                    'response': value, 'subject': subject,
                    'covariate': np.random.normal(10, 2)
                })
    
    return pd.DataFrame(data)


def expand_repeated_measures_anova(n_subjects: int = 40) -> pd.DataFrame:
    """Create repeated measures data: 40 subjects × 4 time points = 160 rows."""
    time_points = ['T0', 'T1', 'T2', 'T3']
    time_effects = {'T0': 0, 'T1': 3, 'T2': 7, 'T3': 5}  # peak at T2, decline at T3
    
    data = []
    for subj in range(1, n_subjects + 1):
        # Subject random intercept
        subj_effect = np.random.normal(0, 3)
        baseline = 50 + subj_effect
        
        for t in time_points:
            value = baseline + time_effects[t] + np.random.normal(0, 2)
            data.append({'subject': subj, 'time': t, 'value': value})
    
    return pd.DataFrame(data)


def expand_seasonal_timeseries(n_points: int = 365) -> pd.DataFrame:
    """Create seasonal time series with 365 daily observations."""
    t = np.arange(n_points)
    
    # Components
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Annual cycle
    weekly = 3 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
    noise = np.random.normal(0, 2, n_points)
    
    # Create multiple related series
    base = 100 + trend + seasonal + weekly + noise
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'value': base,
        'temperature': 15 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_points),
        'sales': base * 1.2 + np.random.normal(0, 5, n_points),
        'inventory': 500 - 0.3 * base + np.random.normal(0, 10, n_points),
        'demand': base + np.random.poisson(10, n_points),
        'price': 50 + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1, n_points)
    })


def expand_ml_classification_predict(n_samples: int = 200) -> pd.DataFrame:
    """Create ML classification prediction data with 200 samples."""
    np.random.seed(123)  # Different seed for prediction set
    
    # Binary classification problem
    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0
    
    # Class 0 features
    X0 = np.random.multivariate_normal(
        mean=[2, 3, 1, 5, 2, 1, 0],
        cov=np.eye(7) * 0.8,
        size=n_class_0
    )
    
    # Class 1 features (shifted)
    X1 = np.random.multivariate_normal(
        mean=[4, 5, 3, 7, 4, 3, 2],
        cov=np.eye(7) * 0.8,
        size=n_class_1
    )
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_class_0 + [1] * n_class_1)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X = X[idx]
    y = y[idx]
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 8)])
    df['target'] = y
    
    return df


def expand_ml_regression_predict(n_samples: int = 200) -> pd.DataFrame:
    """Create ML regression prediction data with 200 samples."""
    np.random.seed(456)  # Different seed for prediction set
    
    # Generate features
    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(0, 5, n_samples)
    X3 = np.random.normal(5, 2, n_samples)
    X4 = np.random.exponential(2, n_samples)
    X5 = np.random.uniform(-3, 3, n_samples)
    X6 = np.random.normal(0, 1, n_samples)
    X7 = np.random.uniform(1, 10, n_samples)
    
    # Target with known relationship
    y = (2 * X1 + 3 * X2 - 0.5 * X3 + 0.8 * X4 + 
         1.2 * X5 - 0.3 * X6 + 0.5 * X7 + np.random.normal(0, 3, n_samples))
    
    return pd.DataFrame({
        'feature_1': X1, 'feature_2': X2, 'feature_3': X3,
        'feature_4': X4, 'feature_5': X5, 'feature_6': X6,
        'feature_7': X7, 'target': y
    })


def expand_signal_analysis(n_samples: int = 4000) -> pd.DataFrame:
    """Create multivariate signal data with 4000 samples and 6 channels."""
    fs = 1000  # Sampling frequency
    t = np.arange(n_samples) / fs  # Time in seconds
    
    # Base frequencies
    f1, f2, f3 = 10, 25, 50  # Hz
    
    # Channel 1: Clean sine waves
    ch1 = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Channel 2: Chirp signal (frequency sweep)
    ch2 = np.sin(2 * np.pi * (f1 + 20 * t) * t)
    
    # Channel 3: Amplitude modulated
    ch3 = (1 + 0.5 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * f3 * t)
    
    # Channel 4: Phase-locked to ch1 with noise
    ch4 = np.sin(2 * np.pi * f1 * t + np.pi/4) + np.random.normal(0, 0.3, n_samples)
    
    # Channel 5: Mixed frequencies with harmonics
    ch5 = (np.sin(2 * np.pi * f1 * t) + 
           0.5 * np.sin(2 * np.pi * 2 * f1 * t) + 
           0.25 * np.sin(2 * np.pi * 3 * f1 * t))
    
    # Channel 6: Random noise for comparison
    ch6 = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame({
        'time': t,
        'channel_1': ch1,
        'channel_2': ch2,
        'channel_3': ch3,
        'channel_4': ch4,
        'channel_5': ch5,
        'channel_6': ch6
    })


def main():
    print("🔧 Expanding test data files...\n")
    
    files_created = []
    
    # 1. One-way ANOVA (was 30 rows, now 150)
    df = expand_oneway_anova(n_per_group=50)
    path = OUTPUT_DIR / 'oneway_anova_data.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 30 → {len(df)} rows")
    
    # 2. Two-way ANOVA (was 30 rows, now 100)
    df = expand_twoway_anova(n_per_cell=25)
    path = OUTPUT_DIR / 'twoway_anova_data.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 30 → {len(df)} rows")
    
    # 3. Repeated measures ANOVA (was 40 rows, now 160)
    df = expand_repeated_measures_anova(n_subjects=40)
    path = OUTPUT_DIR / 'repeated_measures_anova_data.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 40 → {len(df)} rows")
    
    # 4. Seasonal time series (was 60 rows, now 365)
    df = expand_seasonal_timeseries(n_points=365)
    path = OUTPUT_DIR / 'seasonal_timeseries.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 60 → {len(df)} rows")
    
    # 5. ML classification predict (was 75 rows, now 200)
    df = expand_ml_classification_predict(n_samples=200)
    path = OUTPUT_DIR / 'ml_classification_predict.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 75 → {len(df)} rows")
    
    # 6. ML regression predict (was 50 rows, now 200)
    df = expand_ml_regression_predict(n_samples=200)
    path = OUTPUT_DIR / 'ml_regression_predict.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 50 → {len(df)} rows")
    
    # 7. Signal analysis (was 2 columns, now 7 with 4000 samples)
    df = expand_signal_analysis(n_samples=4000)
    path = OUTPUT_DIR / 'signal_analysis_sample.csv'
    df.to_csv(path, index=False)
    files_created.append((path.name, len(df)))
    print(f"✅ {path.name}: 2000×2 → {len(df)}×{len(df.columns)} (multivariate)")
    
    print(f"\n📁 Expanded {len(files_created)} test data files successfully!")
    print("\nSummary:")
    for name, rows in files_created:
        print(f"  • {name}: {rows} rows")


if __name__ == '__main__':
    main()
