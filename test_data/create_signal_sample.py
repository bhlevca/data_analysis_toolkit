"""Create a reproducible periodic test signal CSV for Fourier and Wavelet tests.

Generates a CSV with columns: time,value
Default: duration=5s, sampling rate=100 Hz, signal = sin(2π*5t) + 0.5*sin(2π*20t) + small noise

Usage:
    python test_data/create_signal_sample.py --duration 5 --sr 100 --out test_data/signal_analysis_sample.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate_signal(duration=5.0, sr=100.0, f1=5.0, f2=20.0, snr=0.1, seed=0):
    rng = np.random.RandomState(seed)
    N = int(np.round(duration * sr))
    t = np.arange(N) / float(sr)
    # Two distinct frequencies and amplitudes
    sig = 1.0 * np.sin(2 * np.pi * 3.0 * t) + 0.6 * np.sin(2 * np.pi * 15.0 * t)
    sig += snr * rng.normal(size=N)
    return t, sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=5.0, help='Duration in seconds')
    p.add_argument('--sr', type=float, default=100.0, help='Sampling rate in Hz')
    p.add_argument('--out', type=str, default='test_data/signal_analysis_sample.csv', help='Output CSV path')
    p.add_argument('--seed', type=int, default=0, help='Random seed for noise')
    args = p.parse_args()

    t, sig = generate_signal(duration=args.duration, sr=args.sr, seed=args.seed)
    df = pd.DataFrame({'time': t, 'value': sig})
    df.to_csv(args.out, index=False)
    print(f'Wrote {len(df)} samples to {args.out}')
    print('Signal: value = 1.0*sin(2π*3t) + 0.6*sin(2π*15t) + noise')


if __name__ == '__main__':
    main()
