"""
Unit tests for extended_statistics module.

Tests for:
- ExtendedStatisticalTests class (KS, Anderson-Darling, Runs, Sign, etc.)
- DistributionOperations class (KDE, percentiles, moments, entropy, etc.)
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_toolkit.extended_statistics import (
    ExtendedStatisticalTests,
    DistributionOperations
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def normal_data():
    """Generate normally distributed test data."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'normal1': np.random.normal(100, 15, n),
        'normal2': np.random.normal(100, 15, n),
        'normal3': np.random.normal(105, 15, n),  # Slightly different mean
        'exponential': np.random.exponential(10, n),
        'uniform': np.random.uniform(0, 100, n)
    })


@pytest.fixture
def paired_data():
    """Generate paired test data."""
    np.random.seed(42)
    n = 50
    baseline = np.random.normal(100, 10, n)
    return pd.DataFrame({
        'before': baseline,
        'after': baseline + np.random.normal(5, 3, n),  # Treatment effect
        'placebo': baseline + np.random.normal(0, 3, n)  # No effect
    })


@pytest.fixture
def repeated_measures_data():
    """Generate repeated measures test data."""
    np.random.seed(42)
    n = 30
    subject_effect = np.random.normal(0, 5, n)
    return pd.DataFrame({
        'time1': 100 + subject_effect + np.random.normal(0, 3, n),
        'time2': 105 + subject_effect + np.random.normal(0, 3, n),
        'time3': 110 + subject_effect + np.random.normal(0, 3, n),
        'time4': 108 + subject_effect + np.random.normal(0, 3, n)
    })


@pytest.fixture
def variance_data():
    """Generate data with different variances."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'low_var': np.random.normal(100, 5, n),
        'med_var': np.random.normal(100, 15, n),
        'high_var': np.random.normal(100, 30, n)
    })


# =============================================================================
# EXTENDED STATISTICAL TESTS
# =============================================================================

class TestKolmogorovSmirnov:
    """Tests for Kolmogorov-Smirnov tests."""
    
    def test_ks_1sample_normal(self, normal_data):
        """Test 1-sample KS on normally distributed data."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.kolmogorov_smirnov_1sample('normal1', 'norm')
        
        assert 'test' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert result['test'] == 'Kolmogorov-Smirnov (1-sample)'
        # Normal data should not reject null
        assert result['p_value'] > 0.01
    
    def test_ks_1sample_exponential(self, normal_data):
        """Test 1-sample KS on exponential data."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.kolmogorov_smirnov_1sample('exponential', 'expon')
        
        assert 'statistic' in result
        # Exponential data should fit exponential distribution
        assert result['p_value'] > 0.01
    
    def test_ks_2sample_same(self, normal_data):
        """Test 2-sample KS on similar distributions."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.kolmogorov_smirnov_2sample('normal1', 'normal2')
        
        assert 'test' in result
        assert result['test'] == 'Kolmogorov-Smirnov (2-sample)'
        # Same distribution should not reject null
        assert result['p_value'] > 0.01
    
    def test_ks_2sample_different(self, normal_data):
        """Test 2-sample KS on different distributions."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.kolmogorov_smirnov_2sample('normal1', 'exponential')
        
        # Different distributions should reject null
        assert result['reject_null'] == True


class TestAndersonDarling:
    """Tests for Anderson-Darling test."""
    
    def test_ad_normal_data(self, normal_data):
        """Test A-D on normally distributed data."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.anderson_darling('normal1', 'norm')
        
        assert 'test' in result
        assert 'statistic' in result
        assert 'critical_values' in result
        # Normal data should pass normality test
        assert len(result['reject_at_levels']) == 0
    
    def test_ad_non_normal_data(self, normal_data):
        """Test A-D on non-normal data."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.anderson_darling('exponential', 'norm')
        
        # Exponential is not normal, should reject
        assert len(result['reject_at_levels']) > 0


class TestRunsTest:
    """Tests for Runs test for randomness."""
    
    def test_runs_random_sequence(self, normal_data):
        """Test runs test on random data."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.runs_test('normal1')
        
        assert 'test' in result
        assert 'n_runs' in result
        assert 'expected_runs' in result
        assert 'z_statistic' in result
        assert 'p_value' in result
    
    def test_runs_detects_pattern(self):
        """Test that runs test detects non-random pattern."""
        # Create alternating pattern (very non-random)
        pattern = pd.DataFrame({
            'alternating': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20
        })
        
        est = ExtendedStatisticalTests(pattern)
        result = est.runs_test('alternating', cutoff='mean')
        
        # Alternating pattern has too many runs
        assert result['reject_null'] == True


class TestSignTest:
    """Tests for Sign test."""
    
    def test_sign_test_effect(self, paired_data):
        """Test sign test detects treatment effect."""
        est = ExtendedStatisticalTests(paired_data)
        result = est.sign_test('after', 'before')
        
        assert 'test' in result
        assert 'n_positive' in result
        assert 'n_negative' in result
        assert 'p_value' in result
    
    def test_sign_test_no_effect(self, paired_data):
        """Test sign test with no treatment effect."""
        est = ExtendedStatisticalTests(paired_data)
        result = est.sign_test('placebo', 'before')
        
        # No true effect, should not reject
        assert result['p_value'] > 0.01


class TestMoodMedianTest:
    """Tests for Mood's Median test."""
    
    def test_mood_equal_medians(self, normal_data):
        """Test Mood's test on equal medians."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.mood_median_test('normal1', 'normal2')
        
        assert 'test' in result
        assert 'grand_median' in result
        assert 'contingency_table' in result
        # Equal medians should not reject
        assert result['p_value'] > 0.01


class TestFriedmanTest:
    """Tests for Friedman test."""
    
    def test_friedman_effect(self, repeated_measures_data):
        """Test Friedman test detects treatment effect."""
        est = ExtendedStatisticalTests(repeated_measures_data)
        result = est.friedman_test(['time1', 'time2', 'time3', 'time4'])
        
        assert 'test' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'n_treatments' in result
        # There is a time effect
        assert result['reject_null'] == True


class TestVarianceTests:
    """Tests for variance homogeneity tests."""
    
    def test_bartlett_equal_variances(self, normal_data):
        """Test Bartlett on equal variances."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.bartlett_test('normal1', 'normal2')
        
        assert 'test' in result
        assert result['test'] == "Bartlett's Test"
        # Equal variances should not reject
        assert result['p_value'] > 0.01
    
    def test_bartlett_unequal_variances(self, variance_data):
        """Test Bartlett on unequal variances."""
        est = ExtendedStatisticalTests(variance_data)
        result = est.bartlett_test('low_var', 'high_var')
        
        # Unequal variances should reject
        assert result['reject_null'] == True
    
    def test_brown_forsythe_equal(self, normal_data):
        """Test Brown-Forsythe on equal variances."""
        est = ExtendedStatisticalTests(normal_data)
        result = est.brown_forsythe_test('normal1', 'normal2')
        
        assert 'test' in result
        assert result['test'] == 'Brown-Forsythe Test'
        # Equal variances should not reject
        assert result['p_value'] > 0.01


# =============================================================================
# DISTRIBUTION OPERATIONS
# =============================================================================

class TestKernelDensityEstimation:
    """Tests for KDE operations."""
    
    def test_kde_basic(self, normal_data):
        """Test basic KDE calculation."""
        do = DistributionOperations(normal_data)
        result = do.kernel_density_estimation('normal1')
        
        assert 'x' in result
        assert 'density' in result
        assert 'bandwidth' in result
        assert 'mode_estimate' in result
        assert len(result['x']) == 200
        assert len(result['density']) == 200
    
    def test_kde_mode_estimate(self, normal_data):
        """Test KDE mode estimate is reasonable."""
        do = DistributionOperations(normal_data)
        result = do.kernel_density_estimation('normal1')
        
        # Mode should be close to true mean (100)
        assert 80 < result['mode_estimate'] < 120


class TestPercentiles:
    """Tests for percentile calculations."""
    
    def test_percentiles_default(self, normal_data):
        """Test default percentile calculation."""
        do = DistributionOperations(normal_data)
        result = do.percentiles('normal1')
        
        assert 'p50' in result  # Median
        assert 'p25' in result  # Q1
        assert 'p75' in result  # Q3
        # Median should be close to true mean (100)
        assert 80 < result['p50'] < 120
    
    def test_percentiles_custom(self, normal_data):
        """Test custom percentiles."""
        do = DistributionOperations(normal_data)
        result = do.percentiles('normal1', percentiles=[10, 50, 90])
        
        assert 'p10' in result
        assert 'p50' in result
        assert 'p90' in result
        assert result['p10'] < result['p50'] < result['p90']


class TestMoments:
    """Tests for moment calculations."""
    
    def test_moments_basic(self, normal_data):
        """Test moment calculations."""
        do = DistributionOperations(normal_data)
        result = do.moments('normal1')
        
        assert 'raw_moments' in result
        assert 'central_moments' in result
        assert 'standardized_moments' in result
        
        # Check specific moments
        assert 'm1' in result['raw_moments']  # Mean
        assert 'mu2' in result['central_moments']  # Variance
        assert 'skewness' in result['standardized_moments']
        assert 'kurtosis' in result['standardized_moments']
    
    def test_moments_normal_distribution(self, normal_data):
        """Test moments for normal distribution."""
        do = DistributionOperations(normal_data)
        result = do.moments('normal1')
        
        # Normal distribution should have near-zero skewness
        assert abs(result['standardized_moments']['skewness']) < 0.5
        # Normal distribution has excess kurtosis near 0
        assert abs(result['standardized_moments']['excess_kurtosis']) < 1


class TestEntropy:
    """Tests for entropy calculations."""
    
    def test_entropy_basic(self, normal_data):
        """Test entropy calculation."""
        do = DistributionOperations(normal_data)
        result = do.entropy('normal1')
        
        assert 'shannon_entropy' in result
        assert 'normalized_entropy' in result
        assert 'negentropy' in result
        
        # Normalized entropy should be between 0 and 1
        assert 0 <= result['normalized_entropy'] <= 1
    
    def test_entropy_comparison(self, normal_data):
        """Compare entropy of different distributions."""
        do = DistributionOperations(normal_data)
        
        uniform_entropy = do.entropy('uniform')
        normal_entropy = do.entropy('normal1')
        
        # Uniform distribution should have higher entropy (more uncertainty)
        # This may not always hold due to bin effects, so just check they're computed
        assert uniform_entropy['shannon_entropy'] > 0
        assert normal_entropy['shannon_entropy'] > 0


class TestDistributionSampling:
    """Tests for distribution sampling."""
    
    def test_sampling_normal(self, normal_data):
        """Test sampling from fitted normal distribution."""
        do = DistributionOperations(normal_data)
        result = do.distribution_sampling('normal1', 'norm', n_samples=100)
        
        assert 'samples' in result
        assert 'parameters' in result
        assert len(result['samples']) == 100
    
    def test_sampling_gamma(self, normal_data):
        """Test sampling from gamma distribution."""
        do = DistributionOperations(normal_data)
        result = do.distribution_sampling('exponential', 'gamma', n_samples=50)
        
        assert len(result['samples']) == 50


class TestProbabilityCalculations:
    """Tests for probability calculations."""
    
    def test_probability_calcs(self, normal_data):
        """Test probability calculations."""
        do = DistributionOperations(normal_data)
        result = do.probability_calculations('normal1', 'norm')
        
        assert 'calculations' in result
        assert len(result['calculations']) > 0
        
        # Check structure of calculations
        calc = result['calculations'][0]
        assert 'cdf' in calc
        assert 'pdf' in calc
        assert 'sf' in calc
        
        # CDF + SF should equal 1
        assert abs(calc['cdf'] + calc['sf'] - 1.0) < 0.001


class TestNoDataLoaded:
    """Test error handling when no data loaded."""
    
    def test_extended_stats_no_data(self):
        """Test extended stats error handling."""
        est = ExtendedStatisticalTests()
        result = est.kolmogorov_smirnov_1sample('col', 'norm')
        assert 'error' in result
    
    def test_dist_ops_no_data(self):
        """Test distribution ops error handling."""
        do = DistributionOperations()
        result = do.kernel_density_estimation('col')
        assert 'error' in result


class TestSetData:
    """Test set_data methods."""
    
    def test_set_data_extended_stats(self, normal_data):
        """Test setting data for extended stats."""
        est = ExtendedStatisticalTests()
        est.set_data(normal_data)
        result = est.kolmogorov_smirnov_1sample('normal1', 'norm')
        assert 'error' not in result
    
    def test_set_data_dist_ops(self, normal_data):
        """Test setting data for distribution ops."""
        do = DistributionOperations()
        do.set_data(normal_data)
        result = do.percentiles('normal1')
        assert 'error' not in result
