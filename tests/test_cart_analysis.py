"""
Unit tests for CART analysis module.

Tests for:
- CARTAnalysis class (regression and classification)
- Integration with Morris Screening
- Monte Carlo simulations
- Hypercube generation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_toolkit.cart_analysis import (
    CARTAnalysis,
    sensitivity_to_cart_workflow
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def regression_data():
    """Generate regression test data with known relationships."""
    np.random.seed(42)
    n = 200
    
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 10, n)
    x3 = np.random.uniform(0, 10, n)
    x4 = np.random.uniform(0, 10, n)  # Less important
    x5 = np.random.uniform(0, 10, n)  # Noise
    
    # y depends mainly on x1, x2, x3
    y = 3*x1 + 2*x2 - x3 + 0.1*x4 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
        'y': y
    })


@pytest.fixture
def classification_data():
    """Generate classification test data."""
    np.random.seed(42)
    n = 200
    
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 10, n)
    x3 = np.random.uniform(0, 10, n)
    
    # Class based on x1 and x2
    y = ((x1 > 5) & (x2 > 5)).astype(int)
    
    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3,
        'class': y
    })


# =============================================================================
# CART REGRESSION TESTS
# =============================================================================

class TestCARTRegression:
    """Tests for CART regression."""
    
    def test_basic_regression(self, regression_data):
        """Test basic CART regression."""
        cart = CARTAnalysis(regression_data)
        result = cart.cart_regression(
            ['x1', 'x2', 'x3', 'x4', 'x5'],
            'y',
            max_depth=5
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'feature_importance' in result
        assert result['metrics']['train_r2'] > 0.5
        assert result['metrics']['test_r2'] > 0
    
    def test_feature_importance_order(self, regression_data):
        """Test that important features are ranked higher."""
        cart = CARTAnalysis(regression_data)
        result = cart.cart_regression(
            ['x1', 'x2', 'x3', 'x4', 'x5'],
            'y',
            max_depth=5
        )
        
        # x1 has largest coefficient, should be among top features
        ranking = result['importance_ranking']
        top_features = [r[0] for r in ranking[:3]]
        assert 'x1' in top_features or 'x2' in top_features
    
    def test_cross_validation(self, regression_data):
        """Test that CV scores are reasonable."""
        cart = CARTAnalysis(regression_data)
        result = cart.cart_regression(
            ['x1', 'x2', 'x3'],
            'y',
            max_depth=5
        )
        
        assert 'cv_r2_mean' in result['metrics']
        assert 'cv_r2_std' in result['metrics']
        assert result['metrics']['cv_r2_mean'] > 0


class TestCARTClassification:
    """Tests for CART classification."""
    
    def test_basic_classification(self, classification_data):
        """Test basic CART classification."""
        cart = CARTAnalysis(classification_data)
        result = cart.cart_classification(
            ['x1', 'x2', 'x3'],
            'class',
            max_depth=5
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'classes' in result
        assert result['metrics']['train_accuracy'] > 0.7
    
    def test_classification_accuracy(self, classification_data):
        """Test classification accuracy is reasonable."""
        cart = CARTAnalysis(classification_data)
        result = cart.cart_classification(
            ['x1', 'x2'],  # Only the relevant features
            'class',
            max_depth=5
        )
        
        assert result['metrics']['test_accuracy'] > 0.7


# =============================================================================
# HYPERCUBE GENERATION
# =============================================================================

class TestHypercubeGeneration:
    """Tests for Latin Hypercube Sampling."""
    
    def test_hypercube_size(self, regression_data):
        """Test hypercube has correct dimensions."""
        cart = CARTAnalysis(regression_data)
        cart.feature_names = ['x1', 'x2', 'x3']
        
        hc = cart.generate_hypercube(n_samples=100)
        
        assert len(hc) == 100
        assert len(hc.columns) == 3
    
    def test_hypercube_bounds(self, regression_data):
        """Test hypercube respects bounds."""
        cart = CARTAnalysis(regression_data)
        
        bounds = {'x1': (0, 10), 'x2': (5, 15)}
        hc = cart.generate_hypercube(['x1', 'x2'], n_samples=100, bounds=bounds)
        
        assert hc['x1'].min() >= 0
        assert hc['x1'].max() <= 10
        assert hc['x2'].min() >= 5
        assert hc['x2'].max() <= 15
    
    def test_hypercube_coverage(self, regression_data):
        """Test LHS covers parameter space well."""
        cart = CARTAnalysis(regression_data)
        
        hc = cart.generate_hypercube(['x1', 'x2'], n_samples=1000, 
                                    bounds={'x1': (0, 10), 'x2': (0, 10)})
        
        # Check coverage - all quartiles should have samples
        for col in ['x1', 'x2']:
            q25 = (hc[col] < 2.5).sum()
            q50 = ((hc[col] >= 2.5) & (hc[col] < 5)).sum()
            q75 = ((hc[col] >= 5) & (hc[col] < 7.5)).sum()
            q100 = (hc[col] >= 7.5).sum()
            
            # Each quartile should have ~25% ± 10%
            assert 150 < q25 < 350
            assert 150 < q50 < 350
            assert 150 < q75 < 350
            assert 150 < q100 < 350


# =============================================================================
# MONTE CARLO SIMULATIONS
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo predictions."""
    
    def test_mc_predictions_regression(self, regression_data):
        """Test Monte Carlo predictions for regression."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=5)
        
        mc = cart.monte_carlo_predictions(n_samples=500)
        
        assert 'statistics' in mc
        assert 'predictions' in mc
        assert len(mc['predictions']) == 500
        assert 'mean' in mc['statistics']
        assert 'std' in mc['statistics']
    
    def test_mc_statistics(self, regression_data):
        """Test Monte Carlo statistics are reasonable."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=5)
        
        mc = cart.monte_carlo_predictions(n_samples=1000)
        stats = mc['statistics']
        
        # 5th percentile should be less than 95th
        assert stats['p5'] < stats['p95']
        # IQR should be positive
        assert stats['iqr'] > 0
        # CV should be non-negative
        assert stats['cv'] >= 0
    
    def test_mc_with_custom_hypercube(self, regression_data):
        """Test MC with pre-generated hypercube."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=5)
        
        hc = cart.generate_hypercube(['x1', 'x2', 'x3'], n_samples=200)
        mc = cart.monte_carlo_predictions(hypercube=hc)
        
        assert len(mc['predictions']) == 200
        assert 'hypercube' in mc


# =============================================================================
# MORRIS INTEGRATION
# =============================================================================

class TestMorrisIntegration:
    """Tests for Morris screening integration."""
    
    def test_from_morris_basic(self, regression_data):
        """Test building CART from Morris results."""
        # Create mock Morris results
        morris_results = {
            'parameter_ranking': [
                {'parameter': 'x1', 'mu_star': 3.0, 'sigma': 0.5},
                {'parameter': 'x2', 'mu_star': 2.0, 'sigma': 0.3},
                {'parameter': 'x3', 'mu_star': 1.0, 'sigma': 0.2},
                {'parameter': 'x4', 'mu_star': 0.1, 'sigma': 0.05},
                {'parameter': 'x5', 'mu_star': 0.05, 'sigma': 0.02}
            ]
        }
        
        cart = CARTAnalysis(regression_data)
        result = cart.from_morris_screening(morris_results, 'y', top_n=3)
        
        assert 'morris_integration' in result
        assert result['morris_integration']['parameters_selected'] == 3
        assert 'x1' in result['morris_integration']['selected_parameters']
    
    def test_from_morris_threshold(self, regression_data):
        """Test Morris selection by threshold."""
        morris_results = {
            'parameter_ranking': [
                {'parameter': 'x1', 'mu_star': 3.0, 'sigma': 0.5},
                {'parameter': 'x2', 'mu_star': 2.0, 'sigma': 0.3},
                {'parameter': 'x3', 'mu_star': 1.0, 'sigma': 0.2},
                {'parameter': 'x4', 'mu_star': 0.1, 'sigma': 0.05}
            ]
        }
        
        cart = CARTAnalysis(regression_data)
        result = cart.from_morris_screening(morris_results, 'y', mu_star_threshold=1.5)
        
        # Only x1 and x2 have mu_star >= 1.5
        assert result['morris_integration']['parameters_selected'] == 2


# =============================================================================
# TREE VISUALIZATION
# =============================================================================

class TestTreeVisualization:
    """Tests for tree visualization methods."""
    
    def test_get_tree_rules(self, regression_data):
        """Test tree rules extraction."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=3)
        
        rules = cart.get_tree_rules()
        
        assert isinstance(rules, str)
        assert len(rules) > 0
        assert '|' in rules or '<=' in rules  # Tree structure characters
    
    def test_plot_tree(self, regression_data):
        """Test tree plotting."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=3)
        
        fig = cart.plot_tree()
        
        assert fig is not None
    
    def test_plot_feature_importance(self, regression_data):
        """Test feature importance plotting."""
        cart = CARTAnalysis(regression_data)
        cart.cart_regression(['x1', 'x2', 'x3'], 'y', max_depth=3)
        
        fig = cart.plot_feature_importance()
        
        assert fig is not None


# =============================================================================
# COMPLETE WORKFLOW
# =============================================================================

class TestCompleteWorkflow:
    """Tests for sensitivity_to_cart_workflow function."""
    
    def test_complete_workflow(self, regression_data):
        """Test complete Morris → CART → MC workflow."""
        result = sensitivity_to_cart_workflow(
            regression_data,
            output_col='y',
            input_cols=['x1', 'x2', 'x3', 'x4', 'x5'],
            top_n_params=3,
            num_trajectories=10,
            cart_max_depth=4,
            mc_samples=100
        )
        
        assert 'morris_results' in result
        assert 'cart_results' in result
        assert 'monte_carlo_results' in result
        assert 'workflow_summary' in result
        
        summary = result['workflow_summary']
        assert summary['parameters_selected'] == 3
        assert 'mc_mean' in summary
        assert 'mc_std' in summary


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_no_data(self):
        """Test error when no data loaded."""
        cart = CARTAnalysis()
        result = cart.cart_regression(['x1'], 'y')
        assert 'error' in result
    
    def test_no_model_for_mc(self, regression_data):
        """Test error when running MC without model."""
        cart = CARTAnalysis(regression_data)
        result = cart.monte_carlo_predictions()
        assert 'error' in result
    
    def test_invalid_morris_results(self, regression_data):
        """Test error with invalid Morris results."""
        cart = CARTAnalysis(regression_data)
        result = cart.from_morris_screening({}, 'y', top_n=3)
        assert 'error' in result
