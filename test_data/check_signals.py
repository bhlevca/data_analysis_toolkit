import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

df = pd.read_csv('coherence_signals.csv')
fs = 100  # 100 Hz sampling

print('=== Signal Frequency Analysis ===\n')

for sig in ['signal1', 'signal2', 'signal3_independent']:
    y = df[sig].values
    n = len(y)
    freq = fftfreq(n, 1/fs)
    mag = np.abs(fft(y))
    idx = np.argmax(mag[1:n//2]) + 1
    dom_freq = abs(freq[idx])
    period_samples = fs / dom_freq
    print(f'{sig}:')
    print(f'  Dominant frequency: {dom_freq:.2f} Hz')
    print(f'  Period: {period_samples:.1f} samples')
    print()

print('=== Why MA(20) kills both 5 Hz AND 15 Hz ===\n')
print('At 100 Hz sampling rate:')
print('  5 Hz:  period = 20 samples  → MA(20) = 1 cycle  → avg = 0')
print(' 10 Hz:  period = 10 samples  → MA(20) = 2 cycles → avg = 0')
print(' 15 Hz:  period = 6.67 samples → MA(20) = 3 cycles → avg = 0!')
print(' 20 Hz:  period = 5 samples   → MA(20) = 4 cycles → avg = 0')
print(' 25 Hz:  period = 4 samples   → MA(20) = 5 cycles → avg = 0')
print()
print('MA(20) kills ANY frequency where 20 samples = integer number of cycles!')
print('This includes: 5, 10, 15, 20, 25, ... Hz (all multiples of 5 Hz)')
