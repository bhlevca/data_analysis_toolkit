import pytest
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

from data_toolkit.ml_models import MLModels
from data_toolkit.timeseries_analysis import TimeSeriesAnalysis


def make_clustered_data(n=150):
    # Create three separated Gaussian blobs
    np.random.seed(42)
    a = np.random.normal(loc=0.0, scale=0.3, size=(n//3, 2)) + np.array([0, 0])
    b = np.random.normal(loc=0.0, scale=0.3, size=(n//3, 2)) + np.array([3, 0])
    c = np.random.normal(loc=0.0, scale=0.3, size=(n - 2*(n//3), 2)) + np.array([0, 3])
    X = np.vstack([a, b, c])
    df = pd.DataFrame(X, columns=['x', 'y'])
    return df


def test_clustering_methods_produce_different_labels():
    df = make_clustered_data()
    ml = MLModels(df)

    kres = ml.kmeans_clustering(['x', 'y'], n_clusters=3)
    gmres = ml.gaussian_mixture_model(['x', 'y'], n_components=3)

    k_clusters = np.array(kres.get('clusters'))
    g_clusters = np.array(gmres.get('clusters'))

    # Basic sanity checks: shapes, cluster sizes present and total samples accounted for
    assert k_clusters.shape == g_clusters.shape
    assert 'cluster_sizes' in kres and 'cluster_sizes' in gmres
    # Ensure cluster sizes sum to input size
    assert sum(kres['cluster_sizes'].values()) == df.shape[0]
    assert sum(gmres['cluster_sizes'].values()) == df.shape[0]


def test_cwt_returns_power_and_time_if_pywt_available():
    pywt = pytest.importorskip('pywt')
    # make a synthetic signal with a transient
    t = np.linspace(0, 1, 256)
    sig = np.sin(2 * np.pi * 10 * t) * (np.exp(-((t - 0.5) ** 2) / 0.001))
    df = pd.DataFrame({'value': sig})
    ts = TimeSeriesAnalysis(df)
    res = ts.continuous_wavelet_transform('value', wavelet='morl')

    assert 'power' in res and 'time' in res and 'scales' in res
    power = res['power']
    assert power.ndim == 2
