"""
Create extended test data files for new toolkit features.

Generates:
1. multivariate_timeseries.csv - For VAR/VECM/DTW analysis
2. anova_factorial.csv - For two-way ANOVA
3. repeated_measures.csv - For repeated-measures ANOVA  
4. distribution_samples.csv - For probability distribution fitting
5. coherence_signals.csv - For coherence and cross-wavelet analysis
6. seasonal_timeseries.csv - For ARIMA/SARIMA forecasting

Usage:
    python test_data/create_extended_test_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def create_multivariate_timeseries(n_samples: int = 500) -> pd.DataFrame:
    """
    Create multivariate time series with known causal relationships.
    
    Structure:
    - X causes Y (with lag 2)
    - Y causes Z (with lag 1)
    - W is independent noise
    """
    # Generate time index
    t = np.arange(n_samples)
    
    # X: AR(1) process with trend
    X = np.zeros(n_samples)
    X[0] = np.random.normal(0, 1)
    for i in range(1, n_samples):
        X[i] = 0.7 * X[i-1] + 0.01 * i + np.random.normal(0, 0.5)
    
    # Y: Depends on lagged X
    Y = np.zeros(n_samples)
    for i in range(2, n_samples):
        Y[i] = 0.5 * X[i-2] + 0.3 * Y[i-1] + np.random.normal(0, 0.5)
    
    # Z: Depends on lagged Y
    Z = np.zeros(n_samples)
    for i in range(1, n_samples):
        Z[i] = 0.6 * Y[i-1] + 0.2 * Z[i-1] + np.random.normal(0, 0.5)
    
    # W: Independent process
    W = np.cumsum(np.random.normal(0, 0.5, n_samples))  # Random walk
    
    return pd.DataFrame({
        'time': t,
        'GDP': X + 100,  # Relabel for economic interpretation
        'Consumption': Y + 80,
        'Investment': Z + 50,
        'Noise_Control': W
    })


def create_anova_factorial(n_per_cell: int = 30) -> pd.DataFrame:
    """
    Create data for two-way factorial ANOVA.
    
    Factors:
    - Treatment: Control, Drug_A, Drug_B
    - Gender: Male, Female
    
    Design: 3x2 factorial with interaction effect
    """
    treatments = ['Control', 'Drug_A', 'Drug_B']
    genders = ['Male', 'Female']
    
    data = []
    
    # True effects (for generating data)
    treatment_effects = {'Control': 0, 'Drug_A': 5, 'Drug_B': 8}
    gender_effects = {'Male': 2, 'Female': 0}
    
    # Interaction: Drug_B works better for females
    interaction = {
        ('Control', 'Male'): 0, ('Control', 'Female'): 0,
        ('Drug_A', 'Male'): 0, ('Drug_A', 'Female'): 0,
        ('Drug_B', 'Male'): 0, ('Drug_B', 'Female'): 4  # Extra effect
    }
    
    baseline = 50  # Baseline response
    
    for treatment in treatments:
        for gender in genders:
            for _ in range(n_per_cell):
                response = (baseline + 
                           treatment_effects[treatment] + 
                           gender_effects[gender] +
                           interaction[(treatment, gender)] +
                           np.random.normal(0, 3))
                data.append({
                    'Subject_ID': len(data) + 1,
                    'Treatment': treatment,
                    'Gender': gender,
                    'Response': response
                })
    
    return pd.DataFrame(data)


def create_repeated_measures(n_subjects: int = 40) -> pd.DataFrame:
    """
    Create data for repeated-measures ANOVA.
    
    Design: Each subject measured at 4 time points (before, 1 week, 2 weeks, 4 weeks)
    """
    conditions = ['Baseline', 'Week_1', 'Week_2', 'Week_4']
    
    # True condition effects (improvement over time)
    condition_effects = {'Baseline': 0, 'Week_1': 3, 'Week_2': 6, 'Week_4': 8}
    
    data = []
    
    for subject in range(1, n_subjects + 1):
        # Subject-specific baseline (random intercept)
        subject_baseline = 50 + np.random.normal(0, 5)
        
        for condition in conditions:
            score = (subject_baseline + 
                    condition_effects[condition] + 
                    np.random.normal(0, 2))
            data.append({
                'Subject_ID': f'S{subject:03d}',
                'Timepoint': condition,
                'Score': score
            })
    
    return pd.DataFrame(data)


def create_distribution_samples(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create samples from various distributions for fitting tests.
    """
    return pd.DataFrame({
        'normal': np.random.normal(50, 10, n_samples),
        'lognormal': np.random.lognormal(3, 0.5, n_samples),
        'exponential': np.random.exponential(5, n_samples),
        'gamma': np.random.gamma(2, 5, n_samples),
        'weibull': np.random.weibull(1.5, n_samples) * 10,
        'uniform': np.random.uniform(0, 100, n_samples),
        'bimodal': np.concatenate([
            np.random.normal(30, 5, n_samples // 2),
            np.random.normal(70, 5, n_samples // 2)
        ]),
        't_dist': np.random.standard_t(5, n_samples) * 10 + 50,
    })


def create_coherence_signals(duration: float = 10.0, sr: float = 100.0) -> pd.DataFrame:
    """
    Create paired signals with known coherence at specific frequencies.
    
    Signal 1: Contains 5 Hz and 15 Hz components
    Signal 2: Contains 5 Hz (coherent with S1) and 25 Hz (independent)
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Shared 5 Hz component (high coherence at 5 Hz)
    shared_5hz = np.sin(2 * np.pi * 5 * t)
    
    # Signal 1: shared + independent 15 Hz
    signal1 = shared_5hz + 0.5 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0, 0.1, n_samples)
    
    # Signal 2: shared (with phase shift) + independent 25 Hz
    signal2 = 0.8 * np.sin(2 * np.pi * 5 * t + np.pi/4) + 0.6 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.1, n_samples)
    
    # Signal 3: Independent (control)
    signal3 = np.sin(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.2, n_samples)
    
    return pd.DataFrame({
        'time': t,
        'signal1': signal1,
        'signal2': signal2,
        'signal3_independent': signal3
    })


def create_seasonal_timeseries(n_years: int = 5) -> pd.DataFrame:
    """
    Create seasonal time series for ARIMA/SARIMA forecasting.
    
    Monthly data with:
    - Linear trend
    - Yearly seasonality
    - Autoregressive component
    - Random noise
    """
    n_months = n_years * 12
    t = np.arange(n_months)
    
    # Trend
    trend = 100 + 0.5 * t
    
    # Seasonality (monthly pattern)
    seasonality = 15 * np.sin(2 * np.pi * t / 12) + 5 * np.cos(4 * np.pi * t / 12)
    
    # AR component
    ar = np.zeros(n_months)
    for i in range(1, n_months):
        ar[i] = 0.4 * ar[i-1] + np.random.normal(0, 3)
    
    # Combine
    values = trend + seasonality + ar
    
    # Create date index
    dates = pd.date_range(start='2019-01-01', periods=n_months, freq='MS')
    
    return pd.DataFrame({
        'date': dates,
        'month': dates.month,
        'year': dates.year,
        'sales': values,
        'trend_component': trend,
        'seasonal_component': seasonality
    })


def main():
    """Generate all test data files."""
    output_dir = Path(__file__).parent
    
    print("Generating extended test data files...")
    
    # 1. Multivariate time series
    df = create_multivariate_timeseries()
    path = output_dir / 'multivariate_timeseries.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    # 2. Two-way ANOVA data
    df = create_anova_factorial()
    path = output_dir / 'anova_factorial.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    # 3. Repeated measures data
    df = create_repeated_measures()
    path = output_dir / 'repeated_measures.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    # 4. Distribution samples
    df = create_distribution_samples()
    path = output_dir / 'distribution_samples.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    # 5. Coherence signals
    df = create_coherence_signals()
    path = output_dir / 'coherence_signals.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    # 6. Seasonal time series
    df = create_seasonal_timeseries()
    path = output_dir / 'seasonal_timeseries.csv'
    df.to_csv(path, index=False)
    print(f"✅ Created {path.name} ({len(df)} rows)")
    
    print("\n📁 All test data files created successfully!")
    print("\nFiles can be used to demonstrate:")
    print("  - multivariate_timeseries.csv → VAR, VECM, Granger causality, DTW")
    print("  - anova_factorial.csv → Two-way ANOVA with interaction")
    print("  - repeated_measures.csv → Repeated-measures ANOVA")
    print("  - distribution_samples.csv → Distribution fitting (multiple types)")
    print("  - coherence_signals.csv → Coherence, cross-wavelet, wavelet coherence")
    print("  - seasonal_timeseries.csv → ARIMA, SARIMA, seasonal decomposition")


if __name__ == '__main__':
    main()
