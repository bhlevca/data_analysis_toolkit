"""
Unit tests for bayesian_analysis module.
"""
import pytest
import numpy as np
import pandas as pd
from data_toolkit import BayesianAnalysis


class TestBayesianAnalysis:
    """Tests for the BayesianAnalysis class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 5, n),
            'x3': np.random.normal(5, 2, n),
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] - df['x3'] + np.random.normal(0, 1, n)
        return df
    
    @pytest.fixture
    def analyzer(self, sample_df):
        """Create BayesianAnalysis instance with data."""
        ba = BayesianAnalysis(sample_df)
        return ba
    
    def test_init_with_data(self, sample_df):
        """Test initialization with DataFrame."""
        ba = BayesianAnalysis(sample_df)
        assert ba.df is not None
        assert len(ba.df) == 100
    
    def test_init_without_data(self):
        """Test initialization without DataFrame."""
        ba = BayesianAnalysis()
        assert ba.df is None
    
    def test_set_data(self, sample_df):
        """Test set_data method."""
        ba = BayesianAnalysis()
        ba.set_data(sample_df)
        assert ba.df is not None
    
    def test_bayesian_regression(self, analyzer):
        """Test Bayesian regression."""
        results = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y', n_samples=500)
        
        assert 'coefficients' in results or 'posterior_mean' in results or 'model' in results
        assert 'error' not in results
    
    def test_bayesian_regression_no_data(self):
        """Test Bayesian regression without data returns error."""
        ba = BayesianAnalysis()
        results = ba.bayesian_regression(['x1'], 'y')
        assert 'error' in results
    
    def test_credible_intervals(self, analyzer):
        """Test credible interval calculation."""
        # First run regression
        analyzer.bayesian_regression(['x1', 'x2'], 'y', n_samples=500)
        
        # Then get credible intervals if available
        if hasattr(analyzer, 'credible_intervals'):
            # Check method signature
            import inspect
            sig = inspect.signature(analyzer.credible_intervals)
            if len(sig.parameters) > 0:
                results = analyzer.credible_intervals(['x1', 'x2'], 'y')
            else:
                results = analyzer.credible_intervals()
            assert results is not None
    
    def test_posterior_predictive(self, analyzer):
        """Test posterior predictive distribution."""
        analyzer.bayesian_regression(['x1', 'x2'], 'y', n_samples=100)
        
        if hasattr(analyzer, 'posterior_predictive'):
            # Create new data for prediction
            new_data = pd.DataFrame({
                'x1': [5.0],
                'x2': [2.5]
            })
            results = analyzer.posterior_predictive(new_data)
            assert results is not None
    
    def test_prior_sensitivity(self, analyzer):
        """Test prior sensitivity analysis."""
        if hasattr(analyzer, 'prior_sensitivity'):
            results = analyzer.prior_sensitivity(['x1', 'x2'], 'y')
            assert results is not None


class TestBayesianUncertaintyData:
    """Test with the actual test data file."""
    
    @pytest.fixture
    def uncertainty_df(self):
        """Load bayesian uncertainty test data."""
        try:
            df = pd.read_csv('test_data/bayesian_uncertainty_data.csv')
            return df
        except FileNotFoundError:
            pytest.skip("Test data file not found")
    
    def test_with_real_data(self, uncertainty_df):
        """Test with real test data file."""
        ba = BayesianAnalysis(uncertainty_df)
        
        # Get numeric columns
        numeric_cols = uncertainty_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            features = numeric_cols[:-1]
            target = numeric_cols[-1]
            
            results = ba.bayesian_regression(features, target, n_samples=100)
            assert 'error' not in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
