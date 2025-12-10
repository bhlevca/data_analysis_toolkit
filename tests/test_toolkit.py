"""
Tests for the Data Toolkit package
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import modules to test
from data_toolkit import (
    DataLoader,
    StatisticalAnalysis,
    MLModels,
    BayesianAnalysis,
    UncertaintyAnalysis,
    NonLinearAnalysis,
    TimeSeriesAnalysis,
    CausalityAnalysis,
    VisualizationMethods,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing"""
    np.random.seed(42)
    n = 100
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise = np.random.randn(n) * 0.1
    y = 2 * x1 + 3 * x2 + noise
    
    return pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'target': y,
        'category': np.random.choice(['A', 'B', 'C'], n)
    })


@pytest.fixture
def sample_csv(sample_data):
    """Create temporary CSV file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f, index=False)
        return f.name


@pytest.fixture
def data_loader(sample_csv):
    """Create DataLoader with loaded data"""
    loader = DataLoader()
    loader.load_csv(sample_csv)
    os.unlink(sample_csv)  # Clean up
    return loader


# ============================================================================
# DataLoader Tests
# ============================================================================

class TestDataLoader:
    
    def test_load_csv(self, sample_csv):
        loader = DataLoader()
        success, msg = loader.load_csv(sample_csv)
        assert success is True
        assert loader.df is not None
        assert len(loader.df) == 100
    
    def test_get_columns(self, data_loader):
        cols = data_loader.get_columns()
        assert 'feature1' in cols
        assert 'feature2' in cols
        assert 'target' in cols
    
    def test_get_numeric_columns(self, data_loader):
        numeric = data_loader.get_numeric_columns()
        assert 'feature1' in numeric
        assert 'feature2' in numeric
        assert 'target' in numeric
        assert 'category' not in numeric
    
    def test_get_preview(self, data_loader):
        preview = data_loader.get_preview(10)
        assert len(preview) == 10
    
    def test_get_data_info(self, data_loader):
        info = data_loader.get_data_info()
        assert info['rows'] == 100
        assert info['columns'] == 4


# ============================================================================
# StatisticalAnalysis Tests
# ============================================================================

class TestStatisticalAnalysis:
    
    def test_descriptive_stats(self, sample_data):
        stats = StatisticalAnalysis(sample_data)
        result = stats.descriptive_stats(['feature1', 'feature2'])
        
        assert 'feature1' in result.columns
        assert 'mean' in result.index
        assert 'std' in result.index
    
    def test_correlation_matrix(self, sample_data):
        stats = StatisticalAnalysis(sample_data)
        result = stats.correlation_matrix(['feature1', 'feature2', 'target'])
        
        assert result.shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_almost_equal(np.diag(result), [1, 1, 1])
    
    def test_outlier_detection(self, sample_data):
        stats = StatisticalAnalysis(sample_data)
        result = stats.outlier_detection(['feature1', 'feature2'])
        
        assert 'feature1' in result
        assert 'n_outliers' in result['feature1']
        assert 'percentage' in result['feature1']


# ============================================================================
# MLModels Tests
# ============================================================================

class TestMLModels:
    
    def test_train_linear_regression(self, sample_data):
        ml = MLModels(sample_data)
        result = ml.train_model(
            features=['feature1', 'feature2'],
            target='target',
            model_name='Linear Regression'
        )
        
        assert 'r2' in result
        assert result['r2'] > 0.9  # Should fit well given data generation
        assert 'mse' in result
    
    def test_cross_validation(self, sample_data):
        ml = MLModels(sample_data)
        result = ml.cross_validation(
            features=['feature1', 'feature2'],
            target='target'
        )
        
        assert 'scores' in result
        assert len(result['scores']) == 5
        assert 'mean' in result
    
    def test_pca_analysis(self, sample_data):
        ml = MLModels(sample_data)
        result = ml.pca_analysis(['feature1', 'feature2', 'target'])
        
        assert 'explained_variance' in result
        assert len(result['explained_variance']) == 3
    
    def test_feature_importance(self, sample_data):
        ml = MLModels(sample_data)
        result = ml.feature_importance(
            features=['feature1', 'feature2'],
            target='target'
        )
        
        assert 'feature1' in result
        assert 'feature2' in result


# ============================================================================
# BayesianAnalysis Tests
# ============================================================================

class TestBayesianAnalysis:
    
    def test_bayesian_regression(self, sample_data):
        bayes = BayesianAnalysis(sample_data)
        result = bayes.bayesian_regression(
            features=['feature1', 'feature2'],
            target='target'
        )
        
        assert 'posterior_mean' in result
        assert 'credible_intervals_lower' in result
        assert 'credible_intervals_upper' in result
    
    def test_credible_intervals(self, sample_data):
        bayes = BayesianAnalysis(sample_data)
        result = bayes.credible_intervals(
            features=['feature1', 'feature2'],
            target='target'
        )
        
        assert 'coverage' in result
        assert 0 <= result['coverage'] <= 1


# ============================================================================
# UncertaintyAnalysis Tests
# ============================================================================

class TestUncertaintyAnalysis:
    
    def test_bootstrap_ci(self, sample_data):
        unc = UncertaintyAnalysis(sample_data)
        result = unc.bootstrap_ci(
            features=['feature1', 'feature2'],
            target='target',
            n_bootstrap=100  # Small for speed
        )
        
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert len(result['ci_lower']) == 2
    
    def test_residual_analysis(self, sample_data):
        unc = UncertaintyAnalysis(sample_data)
        result = unc.residual_analysis(
            features=['feature1', 'feature2'],
            target='target'
        )
        
        assert 'durbin_watson' in result
        assert 'residuals' in result


# ============================================================================
# TimeSeriesAnalysis Tests
# ============================================================================

class TestTimeSeriesAnalysis:
    
    def test_stationarity_test(self, sample_data):
        ts = TimeSeriesAnalysis(sample_data)
        result = ts.stationarity_test(['feature1'])
        
        assert 'feature1' in result
        assert 'adf_statistic' in result['feature1']
        assert 'p_value' in result['feature1']
    
    def test_rolling_statistics(self, sample_data):
        ts = TimeSeriesAnalysis(sample_data)
        result = ts.rolling_statistics('feature1', window=10)
        
        assert 'rolling_mean' in result
        assert 'rolling_std' in result


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
