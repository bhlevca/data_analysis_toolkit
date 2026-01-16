"""
Generate test data for Cross-Correlation Function (CCF) analysis.

This creates signals with known frequencies and lag relationships
to verify that CCF works correctly.
"""
import numpy as np
import pandas as pd

# Sampling parameters
fs = 500  # 500 Hz sampling rate
duration = 2  # 2 seconds
t = np.arange(0, duration, 1/fs)
n = len(t)

# Create signals
np.random.seed(42)

# 1. Pure 5 Hz sine wave
sine_5hz = np.sin(2 * np.pi * 5 * t)

# 2. Pure 15 Hz sine wave
sine_15hz = np.sin(2 * np.pi * 15 * t)

# 3. 5 Hz sine wave with 50ms lag (25 samples at 500Hz)
lag_samples = 25  # 50ms lag
sine_5hz_lagged = np.sin(2 * np.pi * 5 * (t - 0.05))

# 4. 5 Hz with Gaussian noise
sine_5hz_noisy = sine_5hz + 0.3 * np.random.randn(n)

# 5. Mixed 5 Hz + 15 Hz
mixed_signal = 0.7 * sine_5hz + 0.3 * sine_15hz

# 6. Cosine 5 Hz (90 degree phase shift from sine)
cosine_5hz = np.cos(2 * np.pi * 5 * t)

# 7. Correlated noise (signal1 + shifted noise)
correlated_noise = sine_5hz + 0.5 * np.roll(sine_5hz, 30) + 0.2 * np.random.randn(n)

# 8. Pure random noise (uncorrelated)
random_noise = np.random.randn(n)

# Create DataFrame
df = pd.DataFrame({
    'time': t,
    'sine_5hz': sine_5hz,
    'sine_15hz': sine_15hz,
    'sine_5hz_lag50ms': sine_5hz_lagged,
    'sine_5hz_noisy': sine_5hz_noisy,
    'mixed_5_15hz': mixed_signal,
    'cosine_5hz': cosine_5hz,
    'correlated_with_lag': correlated_noise,
    'random_noise': random_noise
})

# Save
df.to_csv('ccf_test_signals.csv', index=False)
print(f'Created ccf_test_signals.csv with {n} samples')
print('Columns:', list(df.columns))
print()
print('Expected CCF results:')
print('  - sine_5hz vs sine_15hz: Low correlation (different frequencies)')
print('  - sine_5hz vs sine_5hz_lag50ms: High correlation at lag=25 (50ms)')
print('  - sine_5hz vs cosine_5hz: Peak at lag=25 (90 degree = 1/4 period)')
print('  - sine_5hz vs random_noise: Near zero correlation')
print('  - sine_5hz vs sine_5hz_noisy: High correlation at lag=0')
