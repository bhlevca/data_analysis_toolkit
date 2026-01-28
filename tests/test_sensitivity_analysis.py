"""
Unit tests for sensitivity analysis module.
"""
import pytest
import numpy as np
import pandas as pd
from data_toolkit import SensitivityAnalysis, analyze_dataframe_sensitivity


class TestSensitivityAnalysis:
    """Tests for the SensitivityAnalysis class."""
    
    @pytest.fixture
    def linear_model(self):
        """Simple linear model: f(x) = 2*x1 + 3*x2 + x3"""
        def model(x):
            return 2 * x[0] + 3 * x[1] + x[2]
        return model
    
    @pytest.fixture
    def nonlinear_model(self):
        """Nonlinear model with interactions: f(x) = x1*x2 + x3^2"""
        def model(x):
            return x[0] * x[1] + x[2]**2
        return model
    
    @pytest.fixture
    def param_bounds(self):
        """Standard parameter bounds for testing."""
        return {'x1': (0, 10), 'x2': (0, 10), 'x3': (0, 10)}
    
    def test_morris_screening_linear(self, linear_model, param_bounds):
        """Test Morris screening on linear model."""
        sa = SensitivityAnalysis(linear_model)
        results = sa.morris_screening(param_bounds, n_trajectories=20, seed=42)
        
        assert 'mu' in results
        assert 'mu_star' in results
        assert 'sigma' in results
        assert 'ranking' in results
        assert 'classification' in results
        
        # For linear model, x2 should have highest mu_star (coefficient 3)
        # followed by x1 (coefficient 2), then x3 (coefficient 1)
        ranking = results['ranking']
        assert ranking[0] == 'x2'  # Highest coefficient
        assert ranking[-1] == 'x3'  # Lowest coefficient
    
    def test_morris_screening_nonlinear(self, nonlinear_model, param_bounds):
        """Test Morris screening on nonlinear model."""
        sa = SensitivityAnalysis(nonlinear_model)
        results = sa.morris_screening(param_bounds, n_trajectories=30, seed=42)
        
        # Nonlinear model should show high sigma (indicating interactions)
        # x1 and x2 interact, so should have high sigma
        assert results['sigma']['x1'] > 0
        assert results['sigma']['x2'] > 0
    
    def test_sobol_indices_linear(self, linear_model, param_bounds):
        """Test Sobol indices on linear model."""
        sa = SensitivityAnalysis(linear_model)
        results = sa.sobol_indices(param_bounds, n_samples=500, seed=42)
        
        assert 'S1' in results
        assert 'ST' in results
        assert 'ranking' in results
        
        # For linear model, S1 ≈ ST (no interactions)
        for name in param_bounds.keys():
            # Allow some tolerance for sampling variance
            diff = abs(results['ST'][name] - results['S1'][name])
            assert diff < 0.2, f"{name}: S1={results['S1'][name]:.3f}, ST={results['ST'][name]:.3f}"
    
    def test_sobol_indices_nonlinear(self, nonlinear_model, param_bounds):
        """Test Sobol indices on nonlinear model with interactions."""
        sa = SensitivityAnalysis(nonlinear_model)
        results = sa.sobol_indices(param_bounds, n_samples=500, seed=42)
        
        # For model with x1*x2 interaction, ST > S1 for x1 and x2
        # (total effect includes interaction)
        assert results['ST']['x1'] >= results['S1']['x1'] * 0.8  # Allow some tolerance
        assert results['ST']['x2'] >= results['S1']['x2'] * 0.8
    
    def test_oat_analysis(self, linear_model, param_bounds):
        """Test One-At-a-Time analysis."""
        sa = SensitivityAnalysis(linear_model)
        results = sa.one_at_a_time(param_bounds, n_steps=20)
        
        assert 'sweeps' in results
        assert 'gradients' in results
        assert 'ranking' in results
        
        # Gradients should match coefficients for linear model
        assert abs(results['gradients']['x1'] - 2) < 0.5
        assert abs(results['gradients']['x2'] - 3) < 0.5
        assert abs(results['gradients']['x3'] - 1) < 0.5
    
    def test_no_model_error(self, param_bounds):
        """Test that analysis without model returns error."""
        sa = SensitivityAnalysis()  # No model
        results = sa.morris_screening(param_bounds)
        assert 'error' in results
    
    def test_plot_morris(self, linear_model, param_bounds):
        """Test Morris plot generation."""
        sa = SensitivityAnalysis(linear_model)
        sa.morris_screening(param_bounds, n_trajectories=10, seed=42)
        fig = sa.plot_morris()
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_sobol(self, linear_model, param_bounds):
        """Test Sobol plot generation."""
        sa = SensitivityAnalysis(linear_model)
        sa.sobol_indices(param_bounds, n_samples=100, seed=42)
        fig = sa.plot_sobol()
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_summary_table(self, linear_model, param_bounds):
        """Test summary table generation."""
        sa = SensitivityAnalysis(linear_model)
        sa.morris_screening(param_bounds, n_trajectories=10, seed=42)
        sa.sobol_indices(param_bounds, n_samples=100, seed=42)
        
        table = sa.summary_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3  # 3 parameters
        assert 'Morris μ*' in table.columns
        assert 'Sobol S1' in table.columns


class TestDataframeSensitivity:
    """Tests for the DataFrame convenience function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 5, n),
            'x3': np.random.normal(5, 2, n),
        })
        df['y'] = 3 * df['x1'] + 2 * df['x2'] - 0.5 * df['x3'] + np.random.normal(0, 1, n)
        return df
    
    def test_morris_method(self, sample_df):
        """Test Morris method on DataFrame."""
        results = analyze_dataframe_sensitivity(
            sample_df, 'y', method='morris', n_samples=20, seed=42
        )
        
        assert 'ranking' in results
        assert 'model_coefficients' in results
        # x1 should rank highest (coefficient 3)
        assert results['ranking'][0] == 'x1'
    
    def test_sobol_method(self, sample_df):
        """Test Sobol method on DataFrame."""
        results = analyze_dataframe_sensitivity(
            sample_df, 'y', method='sobol', n_samples=100, seed=42
        )
        
        assert 'S1' in results
        assert 'ST' in results
    
    def test_oat_method(self, sample_df):
        """Test OAT method on DataFrame."""
        results = analyze_dataframe_sensitivity(
            sample_df, 'y', method='oat', n_samples=20, seed=42
        )
        
        assert 'gradients' in results
        assert 'sweeps' in results
    
    def test_auto_feature_detection(self, sample_df):
        """Test automatic feature detection."""
        results = analyze_dataframe_sensitivity(
            sample_df, 'y', features=None, method='morris', n_samples=10, seed=42
        )
        
        # Should automatically detect x1, x2, x3 as features
        assert len(results['param_names']) == 3
    
    def test_invalid_method(self, sample_df):
        """Test invalid method returns error."""
        results = analyze_dataframe_sensitivity(
            sample_df, 'y', method='invalid', n_samples=10
        )
        assert 'error' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
