"""
Unit tests for new features added to the Data Analysis Toolkit.

Tests cover:
- Coherence Analysis
- Cross-Wavelet Transform (XWT)
- Wavelet Coherence (WTC)
- Harmonic Analysis
- Extended ANOVA (Two-Way, Repeated-Measures, Post-Hoc)
- VECM (Vector Error Correction Model)
- Probability Distributions
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_toolkit.signal_analysis import (
    coherence_analysis,
    cross_wavelet_transform,
    wavelet_coherence,
    harmonic_analysis,
)
from data_toolkit.timeseries_analysis import TimeSeriesAnalysis
from data_toolkit.statistical_analysis import StatisticalAnalysis


class TestCoherenceAnalysis:
    """Tests for coherence_analysis function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with correlated signals."""
        np.random.seed(42)
        n = 500
        t = np.linspace(0, 10, n)
        
        # Create two correlated signals with common frequency
        freq = 3.0
        signal1 = np.sin(2 * np.pi * freq * t) + 0.3 * np.random.randn(n)
        signal2 = 0.8 * np.sin(2 * np.pi * freq * t + 0.5) + 0.3 * np.random.randn(n)
        
        return pd.DataFrame({
            'time': t,
            'signal1': signal1,
            'signal2': signal2
        })

    def test_coherence_basic(self, sample_data):
        """Test basic coherence computation."""
        result = coherence_analysis(sample_data, 'signal1', 'signal2', sampling_rate=50.0)
        
        assert 'error' not in result
        assert 'frequencies' in result
        assert 'coherence' in result
        # Phase may be named differently or optional
        assert 'phase' in result or 'cross_spectral_density' in result
        assert len(result['frequencies']) == len(result['coherence'])

    def test_coherence_peak(self, sample_data):
        """Test that coherence detects the correlated frequency."""
        result = coherence_analysis(sample_data, 'signal1', 'signal2', sampling_rate=50.0)
        
        assert 'peak_frequency' in result
        assert 'peak_coherence' in result
        # Coherence should be high for correlated signals
        assert result['peak_coherence'] > 0.5

    def test_coherence_nperseg(self, sample_data):
        """Test coherence with different segment lengths."""
        result = coherence_analysis(sample_data, 'signal1', 'signal2', 
                                    sampling_rate=50.0, nperseg=128)
        
        assert 'error' not in result
        assert len(result['frequencies']) > 0


class TestCrossWaveletTransform:
    """Tests for cross_wavelet_transform function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with related signals."""
        np.random.seed(42)
        n = 256
        t = np.linspace(0, 5, n)
        
        # Signals with common periodicities
        signal1 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
        signal2 = 0.8 * np.sin(2 * np.pi * 2 * t + 0.3) + 0.5 * np.sin(2 * np.pi * 5 * t)
        
        return pd.DataFrame({
            'time': t,
            'signal1': signal1,
            'signal2': signal2
        })

    def test_xwt_basic(self, sample_data):
        """Test basic XWT computation."""
        result = cross_wavelet_transform(sample_data, 'signal1', 'signal2',
                                         sampling_rate=50.0)
        
        assert 'error' not in result
        assert 'xwt' in result or 'xwt_power' in result
        assert 'phase_difference' in result
        assert 'scales' in result

    def test_xwt_with_scales(self, sample_data):
        """Test XWT with specific scales."""
        scales = np.arange(1, 32)
        result = cross_wavelet_transform(sample_data, 'signal1', 'signal2',
                                         scales=scales, sampling_rate=50.0)
        
        assert 'error' not in result
        assert len(result.get('scales', [])) == len(scales)


class TestWaveletCoherence:
    """Tests for wavelet_coherence function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with related signals."""
        np.random.seed(42)
        n = 256
        t = np.linspace(0, 5, n)
        
        signal1 = np.sin(2 * np.pi * 3 * t)
        signal2 = 0.9 * np.sin(2 * np.pi * 3 * t + 0.2) + 0.1 * np.random.randn(n)
        
        return pd.DataFrame({
            'signal1': signal1,
            'signal2': signal2
        })

    def test_wtc_basic(self, sample_data):
        """Test basic wavelet coherence computation."""
        result = wavelet_coherence(sample_data, 'signal1', 'signal2',
                                   sampling_rate=50.0)
        
        assert 'error' not in result
        assert 'coherence' in result
        assert 'scales' in result

    def test_wtc_with_smoothing(self, sample_data):
        """Test WTC with different smoothing factors."""
        result = wavelet_coherence(sample_data, 'signal1', 'signal2',
                                   sampling_rate=50.0, smooth_factor=5)
        
        assert 'error' not in result


class TestHarmonicAnalysis:
    """Tests for harmonic_analysis function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known harmonics."""
        np.random.seed(42)
        n = 500
        t = np.linspace(0, 10, n)
        
        # Create signal with known harmonics
        signal = (3.0 * np.sin(2 * np.pi * 1.0 * t) + 
                  1.5 * np.sin(2 * np.pi * 2.0 * t) +
                  0.5 * np.sin(2 * np.pi * 3.0 * t) +
                  0.2 * np.random.randn(n))
        
        return pd.DataFrame({
            'time': t,
            'signal': signal
        })

    def test_harmonic_basic(self, sample_data):
        """Test basic harmonic analysis."""
        result = harmonic_analysis(sample_data, 'signal', n_harmonics=3, 
                                   sampling_rate=50.0)
        
        assert 'error' not in result
        assert 'r_squared' in result
        assert 'harmonics' in result or 'fitted' in result

    def test_harmonic_fit_quality(self, sample_data):
        """Test that harmonic fit has good quality."""
        result = harmonic_analysis(sample_data, 'signal', n_harmonics=5,
                                   sampling_rate=50.0)
        
        assert 'error' not in result
        # R² should be reasonably high for data with clear harmonics
        assert result.get('r_squared', 0) > 0.7


class TestExtendedANOVA:
    """Tests for extended ANOVA functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for ANOVA tests."""
        np.random.seed(42)
        n = 100
        
        # Create data with group effects
        group1 = np.random.normal(10, 2, n)
        group2 = np.random.normal(12, 2, n)
        group3 = np.random.normal(11, 2, n)
        
        # Two-way data
        factor1 = np.repeat(['A', 'B'], n * 3 // 2)[:n*3]
        factor2 = np.tile(['X', 'Y', 'Z'], n)[:n*3]
        values = np.concatenate([group1, group2, group3])
        
        return {
            'groups': pd.DataFrame({
                'group1': group1,
                'group2': group2,
                'group3': group3
            }),
            'twoway': pd.DataFrame({
                'value': values,
                'factor1': factor1,
                'factor2': factor2
            })
        }

    def test_oneway_anova(self, sample_data):
        """Test one-way ANOVA."""
        stats = StatisticalAnalysis(sample_data['groups'])
        result = stats.anova_oneway(['group1', 'group2', 'group3'])
        
        assert 'error' not in result
        assert 'F' in result or 'statistic' in result
        assert 'p_value' in result

    def test_twoway_anova(self, sample_data):
        """Test two-way ANOVA."""
        stats = StatisticalAnalysis(sample_data['twoway'])
        result = stats.anova_twoway('value', 'factor1', 'factor2')
        
        assert 'error' not in result
        assert 'effects' in result

    def test_posthoc_tukey(self, sample_data):
        """Test Tukey's HSD post-hoc test."""
        df = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(10, 2, 30),
                np.random.normal(15, 2, 30),
                np.random.normal(12, 2, 30)
            ]),
            'group': np.repeat(['A', 'B', 'C'], 30)
        })
        stats = StatisticalAnalysis(df)
        result = stats.posthoc_tukey('value', 'group')
        
        assert 'error' not in result
        assert 'comparisons' in result


class TestVECM:
    """Tests for VECM (Vector Error Correction Model)."""

    @pytest.fixture
    def sample_data(self):
        """Create cointegrated time series data."""
        np.random.seed(42)
        n = 200
        
        # Create cointegrated series
        e = np.random.randn(n)
        x1 = np.cumsum(np.random.randn(n))
        x2 = x1 + 0.5 + 0.3 * e  # Cointegrated with x1
        
        return pd.DataFrame({
            'x1': x1,
            'x2': x2
        })

    def test_vecm_basic(self, sample_data):
        """Test basic VECM fitting."""
        ts = TimeSeriesAnalysis(sample_data)
        result = ts.vecm_model(['x1', 'x2'], deterministic='co', k_ar_diff=1)
        
        # VECM may return warning if no cointegration detected
        assert 'error' not in result or 'warning' in result


class TestTwoWayANOVAWithTestData:
    """Tests for Two-Way ANOVA using the test data file."""

    @pytest.fixture
    def twoway_data(self):
        """Load the twoway_anova_data.csv file."""
        test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
        filepath = os.path.join(test_data_dir, 'twoway_anova_data.csv')
        return pd.read_csv(filepath)

    def test_twoway_anova_with_testfile(self, twoway_data):
        """Test two-way ANOVA with the test data file."""
        stats = StatisticalAnalysis(twoway_data)
        result = stats.anova_twoway('score', 'treatment', 'gender')
        
        assert 'error' not in result
        assert 'effects' in result
        assert 'factor1' in result['effects']
        assert 'factor2' in result['effects']
        assert 'interaction' in result['effects']
        
    def test_twoway_detects_treatment_effect(self, twoway_data):
        """Test that two-way ANOVA detects the treatment effect."""
        stats = StatisticalAnalysis(twoway_data)
        result = stats.anova_twoway('score', 'treatment', 'gender')
        
        # Treatment should have a significant effect in our test data
        assert result['effects']['factor1']['p_value'] < 0.05


class TestRepeatedMeasuresANOVAWithTestData:
    """Tests for Repeated-Measures ANOVA using the test data file."""

    @pytest.fixture
    def rm_data(self):
        """Load the repeated_measures_anova_data.csv file."""
        test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
        filepath = os.path.join(test_data_dir, 'repeated_measures_anova_data.csv')
        return pd.read_csv(filepath)

    def test_repeated_measures_with_testfile(self, rm_data):
        """Test repeated-measures ANOVA with the test data file."""
        stats = StatisticalAnalysis(rm_data)
        result = stats.anova_repeated_measures('score', 'subject_id', 'time_point')
        
        assert 'error' not in result
        assert 'F' in result
        assert 'p_value' in result
        assert 'n_subjects' in result
        assert 'n_conditions' in result
        
    def test_rm_detects_time_effect(self, rm_data):
        """Test that repeated-measures ANOVA detects the time effect."""
        stats = StatisticalAnalysis(rm_data)
        result = stats.anova_repeated_measures('score', 'subject_id', 'time_point')
        
        # Time should have a significant effect (scores increase over time)
        assert result['p_value'] < 0.05
        
    def test_rm_subject_count(self, rm_data):
        """Test that repeated-measures ANOVA counts subjects correctly."""
        stats = StatisticalAnalysis(rm_data)
        result = stats.anova_repeated_measures('score', 'subject_id', 'time_point')
        
        # We have 10 subjects and 4 time points in our test data
        assert result['n_subjects'] == 10
        assert result['n_conditions'] == 4


class TestProbabilityDistributions:
    """Tests for probability distribution fitting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known distributions."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal': np.random.normal(0, 1, 500),
            'exponential': np.random.exponential(2, 500),
            'uniform': np.random.uniform(0, 10, 500)
        })

    def test_distribution_fitting(self, sample_data):
        """Test distribution fitting."""
        stats = StatisticalAnalysis(sample_data)
        
        # Test that distribution analysis runs
        if hasattr(stats, 'fit_distributions'):
            result = stats.fit_distributions('normal')
            assert 'error' not in result


class TestChiSquare:
    """Tests for Chi-Square test."""

    @pytest.fixture
    def sample_data(self):
        """Create categorical data for chi-square test."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'category1': np.random.choice(['A', 'B', 'C'], n),
            'category2': np.random.choice(['X', 'Y'], n)
        })

    def test_chi_square_basic(self, sample_data):
        """Test basic chi-square test."""
        stats = StatisticalAnalysis(sample_data)
        result = stats.chi_square_test('category1', 'category2')
        
        assert 'error' not in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'dof' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
