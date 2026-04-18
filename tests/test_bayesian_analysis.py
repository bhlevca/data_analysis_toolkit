"""
Unit tests for bayesian_analysis module.

Tests cover:
- Bayesian regression with conjugate Normal-Inverse-Gamma priors
- Prior file loading / saving / listing
- Credible intervals (Bayesian, not confidence intervals)
- Posterior predictive distribution
- Prior sensitivity analysis
- Integration with the regenerated heteroscedastic test data
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
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
        return BayesianAnalysis(sample_df)

    def test_init_with_data(self, sample_df):
        ba = BayesianAnalysis(sample_df)
        assert ba.df is not None
        assert len(ba.df) == 100

    def test_init_without_data(self):
        ba = BayesianAnalysis()
        assert ba.df is None

    def test_set_data(self, sample_df):
        ba = BayesianAnalysis()
        ba.set_data(sample_df)
        assert ba.df is not None

    # ── Bayesian regression ──────────────────────────────────────────

    def test_bayesian_regression_default_prior(self, analyzer):
        """Default (weakly informative) prior: zero mean, precision=0.01."""
        results = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y', n_samples=500)

        assert 'error' not in results
        assert 'posterior_mean' in results
        assert 'posterior_cov' in results
        assert 'posterior_samples' in results
        assert 'prior_info' in results
        assert 'ols_coefficients' in results
        assert 'shrinkage_factor' in results
        assert 'predictions' in results
        assert 'residuals' in results
        assert 'standardization' in results
        assert results['posterior_samples'].shape == (500, 4)  # intercept + 3 features

        # Coefficients should be close to true values:  ~[0, 2, 3, -1]
        pm = results['posterior_mean']
        assert abs(pm[1] - 2.0) < 0.5   # x1 coef near 2
        assert abs(pm[2] - 3.0) < 0.5   # x2 coef near 3

    def test_bayesian_regression_custom_prior(self, analyzer):
        """Custom prior_mean and precision are propagated correctly."""
        custom_mean = np.array([0.0, 2.0, 3.0, -1.0])
        results = analyzer.bayesian_regression(
            ['x1', 'x2', 'x3'], 'y',
            n_samples=300,
            prior_mean=custom_mean,
            prior_precision=1.0,
        )
        assert 'error' not in results
        info = results['prior_info']
        assert info['prior_precision'] == 1.0
        assert np.allclose(info['prior_mean'], custom_mean.tolist())

    def test_bayesian_regression_fixed_noise(self, analyzer):
        """Explicitly set noise_var_estimate."""
        results = analyzer.bayesian_regression(
            ['x1', 'x2', 'x3'], 'y',
            n_samples=200,
            noise_var_estimate=1.5,
        )
        assert 'error' not in results
        assert results['noise_variance'] == 1.5

    def test_bayesian_regression_no_data(self):
        ba = BayesianAnalysis()
        results = ba.bayesian_regression(['x1'], 'y')
        assert 'error' in results

    def test_bayesian_regression_credible_intervals(self, analyzer):
        """95% credible intervals from posterior samples."""
        results = analyzer.bayesian_regression(['x1', 'x2'], 'y', n_samples=1000)
        ci_lo = results['credible_intervals_lower']
        ci_hi = results['credible_intervals_upper']
        pm = results['posterior_mean']
        # CI should contain the posterior mean
        for i in range(len(pm)):
            assert ci_lo[i] <= pm[i] <= ci_hi[i]

    def test_bayesian_regression_r2(self, analyzer):
        results = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y', n_samples=200)
        r2 = results['r2_posterior_mean']
        assert 0.0 < r2 <= 1.0

    def test_bayesian_regression_standardization(self, analyzer):
        """Verify standardization info is returned and OLS matches."""
        results = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y', n_samples=200)
        std_info = results['standardization']
        assert 'x_means' in std_info
        assert 'x_stds' in std_info
        assert len(std_info['x_means']) == 3
        # OLS coefficients should be close to posterior mean with weak prior
        ols = results['ols_coefficients']
        pm = results['posterior_mean']
        for i in range(len(ols)):
            assert abs(ols[i] - pm[i]) < 0.5  # weak prior means little shrinkage

    def test_bayesian_regression_strong_prior_shrinks(self, analyzer):
        """Strong zero-mean prior should shrink coefficients toward zero."""
        weak = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y',
                                            n_samples=200, prior_precision=0.001)
        strong = analyzer.bayesian_regression(['x1', 'x2', 'x3'], 'y',
                                              n_samples=200, prior_precision=500.0)
        for i in range(4):
            assert abs(strong['posterior_mean'][i]) < abs(weak['posterior_mean'][i]) + 0.5

    # ── Credible intervals (prediction-level) ────────────────────────

    def test_credible_intervals(self, analyzer):
        results = analyzer.credible_intervals(['x1', 'x2'], 'y', confidence=0.95)
        assert 'error' not in results
        assert 'coverage' in results
        assert 'y_pred' in results
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        assert 'model_alpha' in results
        assert 'model_lambda' in results
        assert 0 < results['coverage'] <= 1.0

    def test_credible_intervals_different_levels(self, analyzer):
        r80 = analyzer.credible_intervals(['x1', 'x2'], 'y', confidence=0.80)
        r99 = analyzer.credible_intervals(['x1', 'x2'], 'y', confidence=0.99)
        assert r99['mean_ci_width'] > r80['mean_ci_width']

    # ── Posterior predictive ─────────────────────────────────────────

    def test_posterior_predictive_training(self, analyzer):
        """Posterior predictive on training data (new_data=None)."""
        results = analyzer.posterior_predictive(['x1', 'x2'], 'y', n_samples=200)
        assert 'error' not in results
        assert 'predictive_mean' in results
        assert 'credible_lower' in results
        assert len(results['predictive_mean']) == 100

    def test_posterior_predictive_new_data(self, analyzer):
        """Posterior predictive on new observations."""
        new_data = pd.DataFrame({'x1': [5.0, 7.0], 'x2': [2.5, 3.5]})
        results = analyzer.posterior_predictive(
            ['x1', 'x2'], 'y', new_data=new_data, n_samples=300,
        )
        assert 'error' not in results
        assert len(results['predictive_mean']) == 2
        assert results['predictive_samples'].shape == (2, 300)

    # ── Prior sensitivity ────────────────────────────────────────────

    def test_prior_sensitivity(self, analyzer):
        results = analyzer.prior_sensitivity(['x1', 'x2'], 'y')
        assert isinstance(results, list)
        assert len(results) >= 6
        # Each entry has prior_precision, posterior_mean, and ols reference
        for r in results:
            assert 'prior_precision' in r
            assert 'posterior_mean' in r
            assert 'r2' in r
            assert 'ols_coefficients' in r

    def test_prior_sensitivity_stability(self, analyzer):
        """With enough data, weak and moderate priors should give similar results."""
        results = analyzer.prior_sensitivity(['x1', 'x2', 'x3'], 'y')
        weak = next(r for r in results if r['prior_precision'] == 0.001)
        moderate = next(r for r in results if r['prior_precision'] == 0.1)
        # Posterior means should be close for weak/moderate priors
        for i in range(len(weak['posterior_mean'])):
            assert abs(weak['posterior_mean'][i] - moderate['posterior_mean'][i]) < 1.0

    def test_prior_sensitivity_strong_shrinks(self, analyzer):
        """Very strong prior (1000) should pull coefficients toward zero."""
        results = analyzer.prior_sensitivity(['x1', 'x2', 'x3'], 'y')
        weak = next(r for r in results if r['prior_precision'] == 0.001)
        strong = next(r for r in results if r['prior_precision'] == 1000.0)
        # Strong zero-mean prior should shrink coefficients toward 0
        for i in range(len(weak['posterior_mean'])):
            assert abs(strong['posterior_mean'][i]) < abs(weak['posterior_mean'][i])


class TestPriorFiles:
    """Test prior JSON file loading, saving, and listing."""

    @pytest.fixture
    def prior_dir(self):
        """Path to the test_data/priors directory."""
        d = Path('test_data/priors')
        if not d.is_dir():
            pytest.skip("test_data/priors directory not found")
        return d

    def test_load_weakly_informative(self, prior_dir):
        prior = BayesianAnalysis.load_prior(prior_dir / 'weakly_informative.json')
        assert isinstance(prior['prior_mean'], np.ndarray)
        assert prior['prior_precision'] == 0.001
        assert prior['noise_var_estimate'] is None
        assert len(prior['prior_mean']) == 4

    def test_load_informative_correct(self, prior_dir):
        prior = BayesianAnalysis.load_prior(prior_dir / 'informative_correct.json')
        assert np.allclose(prior['prior_mean'], [5.0, 2.0, -1.5, 0.8])
        assert prior['prior_precision'] == 1.0
        assert prior['noise_var_estimate'] == 4.0

    def test_load_strong_wrong(self, prior_dir):
        prior = BayesianAnalysis.load_prior(prior_dir / 'strong_wrong.json')
        assert prior['prior_precision'] == 100.0
        assert np.allclose(prior['prior_mean'], [0.0, 0.0, 0.0, 0.0])

    def test_load_domain_moderate(self, prior_dir):
        prior = BayesianAnalysis.load_prior(prior_dir / 'domain_moderate.json')
        assert prior['prior_precision'] == 0.5
        assert prior['prior_mean'][1] == 1.5  # expected x1 coef

    def test_save_and_reload(self, tmp_path):
        """Round-trip: save then load produces identical values."""
        path = tmp_path / 'test_prior.json'
        mean = [1.0, 2.0, 3.0]
        BayesianAnalysis.save_prior(
            path=path,
            prior_mean=mean,
            prior_precision=0.5,
            noise_var_estimate=2.0,
            features=['a', 'b'],
            target='y',
            name='Test prior',
        )
        assert path.exists()
        loaded = BayesianAnalysis.load_prior(path)
        assert np.allclose(loaded['prior_mean'], mean)
        assert loaded['prior_precision'] == 0.5
        assert loaded['noise_var_estimate'] == 2.0

    def test_save_creates_directory(self, tmp_path):
        path = tmp_path / 'subdir' / 'deep' / 'prior.json'
        BayesianAnalysis.save_prior(path, [0.0], 0.01)
        assert path.exists()

    def test_load_missing_field_raises(self, tmp_path):
        """A JSON missing required fields should raise ValueError."""
        bad_file = tmp_path / 'bad.json'
        bad_file.write_text(json.dumps({'name': 'bad'}))
        with pytest.raises(ValueError, match="missing 'prior_mean'"):
            BayesianAnalysis.load_prior(bad_file)

    def test_list_prior_presets(self, prior_dir):
        presets = BayesianAnalysis.list_prior_presets(prior_dir)
        assert len(presets) >= 4
        names = {p['name'] for p in presets}
        assert 'Informative prior (correct domain knowledge)' in names

    def test_list_prior_presets_empty_dir(self, tmp_path):
        presets = BayesianAnalysis.list_prior_presets(tmp_path)
        assert presets == []

    def test_regression_with_loaded_prior(self, prior_dir):
        """End-to-end: load prior file -> feed to bayesian_regression."""
        df = pd.read_csv('test_data/bayesian_uncertainty_data.csv')
        ba = BayesianAnalysis(df)
        prior = BayesianAnalysis.load_prior(prior_dir / 'informative_correct.json')

        results = ba.bayesian_regression(
            features=prior['features'],
            target=prior['target'],
            prior_mean=prior['prior_mean'],
            prior_precision=prior['prior_precision'],
            noise_var_estimate=prior['noise_var_estimate'],
            n_samples=500,
        )
        assert 'error' not in results
        # With a correct informative prior the R² should be high
        assert results['r2_posterior_mean'] > 0.7

    def test_regression_with_wrong_prior(self, prior_dir):
        """Strong wrong prior should still run but R² may be worse."""
        df = pd.read_csv('test_data/bayesian_uncertainty_data.csv')
        ba = BayesianAnalysis(df)
        prior = BayesianAnalysis.load_prior(prior_dir / 'strong_wrong.json')

        results = ba.bayesian_regression(
            features=prior['features'],
            target=prior['target'],
            prior_mean=prior['prior_mean'],
            prior_precision=prior['prior_precision'],
            noise_var_estimate=prior['noise_var_estimate'],
            n_samples=500,
        )
        assert 'error' not in results
        # Strong wrong prior should degrade the fit
        correct_prior = BayesianAnalysis.load_prior(prior_dir / 'informative_correct.json')
        correct_results = ba.bayesian_regression(
            features=correct_prior['features'],
            target=correct_prior['target'],
            prior_mean=correct_prior['prior_mean'],
            prior_precision=correct_prior['prior_precision'],
            noise_var_estimate=correct_prior['noise_var_estimate'],
            n_samples=500,
        )
        assert correct_results['r2_posterior_mean'] > results['r2_posterior_mean']


class TestBayesianUncertaintyData:
    """Test with the actual test data file."""

    @pytest.fixture
    def uncertainty_df(self):
        try:
            return pd.read_csv('test_data/bayesian_uncertainty_data.csv')
        except FileNotFoundError:
            pytest.skip("Test data file not found")

    def test_with_real_data(self, uncertainty_df):
        ba = BayesianAnalysis(uncertainty_df)
        features = ['predictor_1', 'predictor_2', 'predictor_3']
        target = 'response_homoscedastic'
        results = ba.bayesian_regression(features, target, n_samples=200)
        assert 'error' not in results
        # True coefficients: intercept≈5, x1≈2, x2≈-1.5, x3≈0.8
        pm = results['posterior_mean']
        assert abs(pm[1] - 2.0) < 0.5  # predictor_1 coef ~ 2
        assert pm[2] < 0               # predictor_2 coef is negative

    def test_heteroscedastic_data_has_varying_noise(self, uncertainty_df):
        """Verify the heteroscedastic response has dramatically increasing residual variance."""
        from sklearn.linear_model import LinearRegression
        X = uncertainty_df[['predictor_1', 'predictor_2', 'predictor_3']].values
        y = uncertainty_df['response_heteroscedastic'].values
        lr = LinearRegression().fit(X, y)
        resid = y - lr.predict(X)

        bins = np.percentile(X[:, 0], [0, 25, 50, 75, 100])
        stds = []
        for i in range(4):
            lo, hi = bins[i], bins[i + 1] if i < 3 else np.inf
            mask = (X[:, 0] >= lo) & (X[:, 0] < hi)
            stds.append(resid[mask].std())
        # Residual std should grow dramatically from Q1 to Q4
        assert stds[3] > stds[0] * 2.5  # at least 2.5x ratio


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
