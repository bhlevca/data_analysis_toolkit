import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Skip tests if pywt is not installed
pytest.importorskip('pywt')

from data_toolkit.timeseries_analysis import TimeSeriesAnalysis


def test_cwt_periods_and_coi_and_plot_return():
    # Try to load the sample CSV generated in test_data; fallback to synthetic if missing
    sr = 100.0
    sample_path = 'test_data/signal_analysis_sample.csv'
    try:
        df = pd.read_csv(sample_path)
        # Expect columns: time,value
        if 'value' in df.columns:
            column_name = 'value'
        else:
            numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            column_name = numeric_cols[0] if numeric_cols else 'value'
    except Exception:
        # fallback to synthetic signal: sum of two sines + noise
        N = 512
        t = np.arange(N) / sr
        signal = 1.0 * np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 20.0 * t)
        signal += 0.1 * np.random.RandomState(0).normal(size=N)
        df = pd.DataFrame({'x': signal})
        column_name = 'x'

    tsa = TimeSeriesAnalysis(df)

    scales = np.arange(1, 64)

    res = tsa.continuous_wavelet_transform(column_name, scales=scales, wavelet='morl', sampling_rate=sr)

    assert isinstance(res, dict)
    assert 'periods' in res, "CWT results must include 'periods'"
    assert 'coi' in res, "CWT results must include 'coi'"
    assert 'time' in res, "CWT results must include 'time'"
    assert res['periods'].shape[0] == scales.shape[0]
    assert res['coi'].ndim == 1 and res['coi'].shape[0] == res['time'].shape[0]

    fig = tsa.plot_wavelet_torrence(res, column_name)
    # Should return a matplotlib Figure
    assert isinstance(fig, matplotlib.figure.Figure)

    plt.close(fig)
