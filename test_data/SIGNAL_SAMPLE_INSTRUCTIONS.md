Signal Analysis Sample

This file describes the sample signal helper for Fourier/Wavelet testing.

Files added:
- `create_signal_sample.py` — script to generate `signal_analysis_sample.csv` with `time,value` columns.

Default signal parameters:
- sampling rate: 100 Hz
- duration: 5 seconds
- components: 5 Hz (amplitude 1.0) and 20 Hz (amplitude 0.5)
- additive Gaussian noise with std=0.1 (seeded for reproducibility)

To generate:

```bash
python test_data/create_signal_sample.py --duration 5 --sr 100 --out test_data/signal_analysis_sample.csv
```

Suggestion:
- Consider renaming Fourier/Wavelet modules and UI to `signal_analysis` to allow grouping FFT, PSD, CWT, DWT, STFT, spectrograms, etc. I can implement that refactor when you confirm.
