"""
Unit tests for the MCMC sampler module.

Tests cover:
- Prior log_pdf correctness for each distribution
- Prior sample shapes and bounds
- Prior serialization (to_dict / from_dict)
- MCMCSampler.run() with default priors
- MCMCSampler.run() with Laplace priors (sparse)
- MCMCSampler.run() with Student-t priors (robust)
- Convergence diagnostics (R-hat, ESS)
- load_mcmc_prior / save_mcmc_prior round-trip
- BayesianAnalysis.mcmc_regression() convenience wrapper
- Strong prior shrinks coefficients toward prior mean
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from data_toolkit.mcmc import MCMCSampler, Prior, SUPPORTED_PRIORS


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_df():
    """DataFrame with known coefficients: y = 2*x1 + 3*x2 - x3 + noise."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'x1': np.random.uniform(0, 10, n),
        'x2': np.random.uniform(0, 5, n),
        'x3': np.random.normal(5, 2, n),
    })
    df['y'] = 2 * df['x1'] + 3 * df['x2'] - df['x3'] + np.random.normal(0, 1, n)
    return df


@pytest.fixture
def sampler(sample_df):
    return MCMCSampler(sample_df)


# =====================================================================
# Prior class tests
# =====================================================================

class TestPrior:
    """Tests for the Prior dataclass."""

    def test_supported_priors_set(self):
        assert "normal" in SUPPORTED_PRIORS
        assert "laplace" in SUPPORTED_PRIORS
        assert "student_t" in SUPPORTED_PRIORS
        assert "uniform" in SUPPORTED_PRIORS
        assert "cauchy" in SUPPORTED_PRIORS
        assert "half_normal" in SUPPORTED_PRIORS
        assert "half_cauchy" in SUPPORTED_PRIORS
        assert "inverse_gamma" in SUPPORTED_PRIORS

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown prior"):
            Prior(distribution="banana")

    def test_case_insensitive(self):
        p = Prior(distribution="Normal")
        assert p.distribution == "normal"
        p2 = Prior(distribution="  LAPLACE  ")
        assert p2.distribution == "laplace"

    # -- log_pdf correctness ------------------------------------

    def test_normal_log_pdf(self):
        p = Prior("normal", loc=1.0, scale=2.0)
        expected = sp_stats.norm.logpdf(3.0, loc=1.0, scale=2.0)
        assert np.isclose(p.log_pdf(3.0), expected)

    def test_laplace_log_pdf(self):
        p = Prior("laplace", loc=0.0, scale=1.0)
        expected = sp_stats.laplace.logpdf(1.5, loc=0.0, scale=1.0)
        assert np.isclose(p.log_pdf(1.5), expected)

    def test_student_t_log_pdf(self):
        p = Prior("student_t", loc=0.0, scale=2.0, df=5.0)
        expected = sp_stats.t.logpdf(1.0, df=5.0, loc=0.0, scale=2.0)
        assert np.isclose(p.log_pdf(1.0), expected)

    def test_uniform_log_pdf(self):
        p = Prior("uniform", a=0.0, b=10.0)
        expected = sp_stats.uniform.logpdf(5.0, loc=0.0, scale=10.0)
        assert np.isclose(p.log_pdf(5.0), expected)
        # Outside bounds
        assert p.log_pdf(-1.0) == -np.inf

    def test_cauchy_log_pdf(self):
        p = Prior("cauchy", loc=0.0, scale=1.0)
        expected = sp_stats.cauchy.logpdf(2.0, loc=0.0, scale=1.0)
        assert np.isclose(p.log_pdf(2.0), expected)

    def test_half_normal_log_pdf(self):
        p = Prior("half_normal", scale=2.0)
        expected = sp_stats.halfnorm.logpdf(1.5, loc=0, scale=2.0)
        assert np.isclose(p.log_pdf(1.5), expected)
        # Negative values should be -inf
        assert p.log_pdf(-0.1) == -np.inf

    def test_half_cauchy_log_pdf(self):
        p = Prior("half_cauchy", scale=1.0)
        expected = sp_stats.halfcauchy.logpdf(3.0, loc=0, scale=1.0)
        assert np.isclose(p.log_pdf(3.0), expected)
        assert p.log_pdf(-0.5) == -np.inf

    def test_inverse_gamma_log_pdf(self):
        p = Prior("inverse_gamma", a=2.0, b=3.0)
        expected = sp_stats.invgamma.logpdf(1.0, a=2.0, scale=3.0)
        assert np.isclose(p.log_pdf(1.0), expected)
        assert p.log_pdf(-1.0) == -np.inf
        assert p.log_pdf(0.0) == -np.inf

    # -- sample shapes ------------------------------------------

    def test_sample_shape_scalar(self):
        for dist in SUPPORTED_PRIORS:
            p = Prior(dist, loc=0, scale=2, df=3, a=1, b=5)
            s = p.sample(1)
            assert s.shape == (1,), f"Failed for {dist}"

    def test_sample_shape_array(self):
        for dist in SUPPORTED_PRIORS:
            p = Prior(dist, loc=0, scale=2, df=3, a=1, b=5)
            s = p.sample(100)
            assert s.shape == (100,), f"Failed for {dist}"

    def test_half_normal_samples_non_negative(self):
        p = Prior("half_normal", scale=5.0)
        s = p.sample(500)
        assert np.all(s >= 0)

    def test_inverse_gamma_samples_positive(self):
        p = Prior("inverse_gamma", a=2, b=1)
        s = p.sample(500)
        assert np.all(s > 0)

    # -- serialization ------------------------------------------

    def test_to_dict_round_trip(self):
        p = Prior("laplace", loc=1.0, scale=0.5)
        d = p.to_dict()
        p2 = Prior.from_dict(d)
        assert p2.distribution == "laplace"
        assert p2.loc == 1.0
        assert p2.scale == 0.5

    def test_from_dict_ignores_extra_keys(self):
        d = {"distribution": "normal", "loc": 0, "scale": 1, "extra_key": 999}
        p = Prior.from_dict(d)
        assert p.distribution == "normal"


# =====================================================================
# MCMCSampler tests
# =====================================================================

class TestMCMCSampler:
    """Tests for the MCMCSampler class."""

    def test_no_data_returns_error(self):
        sampler = MCMCSampler()
        result = sampler.run(['x1'], 'y')
        assert 'error' in result

    def test_set_data(self, sample_df):
        sampler = MCMCSampler()
        sampler.set_data(sample_df)
        assert sampler.df is not None

    def test_run_default_priors(self, sampler):
        """MCMC with default Normal priors should recover true coefficients."""
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, n_chains=2, seed=123,
        )
        assert 'error' not in result
        assert 'posterior_mean' in result
        assert 'posterior_std' in result
        assert 'credible_intervals_lower' in result
        assert 'credible_intervals_upper' in result
        assert 'chains' in result
        assert 'diagnostics' in result
        assert 'predictions' in result
        assert 'residuals' in result
        assert 'r2' in result
        assert 'priors_used' in result

        # Chains shape: (2, n_keep, 5)  (intercept + 3 features + sigma2)
        assert result['chains'].shape[0] == 2
        assert result['chains'].shape[2] == 5

        # Coefficients near true values: y = 2*x1 + 3*x2 - x3
        pm = result['posterior_mean']
        assert abs(pm[1] - 2.0) < 1.0, f"x1 coef {pm[1]} too far from 2.0"
        assert abs(pm[2] - 3.0) < 1.0, f"x2 coef {pm[2]} too far from 3.0"
        assert abs(pm[3] - (-1.0)) < 1.0, f"x3 coef {pm[3]} too far from -1.0"

    def test_run_laplace_priors(self, sampler):
        """Laplace priors should produce valid MCMC results (sparsity-inducing)."""
        priors = {
            'Intercept': Prior('normal', loc=0, scale=10),
            'x1': Prior('laplace', loc=0, scale=1),
            'x2': Prior('laplace', loc=0, scale=1),
            'x3': Prior('laplace', loc=0, scale=1),
            'sigma2': Prior('inverse_gamma', a=1, b=1),
        }
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            priors=priors, n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        assert 'error' not in result
        assert result['r2'] > 0.5

    def test_run_student_t_priors(self, sampler):
        """Student-t priors should produce valid MCMC results (robust)."""
        priors = {
            'Intercept': Prior('student_t', loc=0, scale=10, df=3),
            'x1': Prior('student_t', loc=2, scale=2, df=3),
            'x2': Prior('student_t', loc=3, scale=2, df=3),
            'x3': Prior('student_t', loc=-1, scale=2, df=3),
            'sigma2': Prior('inverse_gamma', a=2, b=2),
        }
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            priors=priors, n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        assert 'error' not in result
        assert result['r2'] > 0.5

    def test_partial_priors_defaults_fill_in(self, sampler):
        """Only specifying some priors — the rest should get defaults."""
        priors = {
            'x1': Prior('laplace', loc=0, scale=1),
        }
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            priors=priors, n_iter=1500, n_warmup=500, n_chains=1, seed=7,
        )
        assert 'error' not in result
        # Verify the specified prior was used and the rest are normal defaults
        assert result['priors_used']['x1']['distribution'] == 'laplace'
        assert result['priors_used']['x2']['distribution'] == 'normal'
        assert result['priors_used']['sigma2']['distribution'] == 'inverse_gamma'

    def test_acceptance_rate_reasonable(self, sampler):
        """Acceptance rate should be between 5% and 95%."""
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        assert 0.05 < result['acceptance_rate'] < 0.95

    def test_predictions_length_matches_data(self, sampler, sample_df):
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert len(result['predictions']) == len(sample_df)
        assert len(result['residuals']) == len(sample_df)
        assert len(result['y_actual']) == len(sample_df)

    def test_credible_intervals_contain_mean(self, sampler):
        """95% CI should contain the posterior mean."""
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        for i in range(len(result['posterior_mean'])):
            lo = result['credible_intervals_lower'][i]
            hi = result['credible_intervals_upper'][i]
            pm = result['posterior_mean'][i]
            assert lo <= pm <= hi, f"Param {i}: mean {pm} not in [{lo}, {hi}]"

    def test_thinning(self, sampler):
        """Thinning should reduce the number of post-warmup samples."""
        r1 = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, thin=1, n_chains=1, seed=42,
        )
        r5 = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, thin=5, n_chains=1, seed=42,
        )
        n1 = r1['chains'].shape[1]
        n5 = r5['chains'].shape[1]
        assert n5 < n1
        # With thin=5, should be roughly 1/5 of samples
        assert abs(n5 - n1 / 5) <= 2

    def test_param_names(self, sampler):
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert result['param_names'] == ['Intercept', 'x1', 'x2', 'x3', 'sigma2']
        assert result['feature_names'] == ['Intercept', 'x1', 'x2', 'x3']

    def test_noise_variance_positive(self, sampler):
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert result['noise_variance'] > 0


# =====================================================================
# Diagnostics tests
# =====================================================================

class TestDiagnostics:
    """Tests for MCMC convergence diagnostics."""

    def test_rhat_close_to_one_when_converged(self, sampler):
        """Well-behaved chains should have R-hat near 1.0."""
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=3000, n_warmup=1000, n_chains=2, seed=42,
        )
        diag = result['diagnostics']
        assert 'r_hat' in diag
        assert 'ess' in diag
        # All R-hat should be < 1.2 for well-behaved linear regression
        for j, rh in enumerate(diag['r_hat']):
            assert rh < 1.2, f"R-hat for param {j} = {rh:.3f} (>1.2)"

    def test_ess_positive(self, sampler):
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        diag = result['diagnostics']
        for e in diag['ess']:
            assert e > 0

    def test_diagnostics_with_single_chain(self, sampler):
        """With 1 chain, R-hat should be 1.0 (not computable)."""
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        diag = result['diagnostics']
        assert all(r == 1.0 for r in diag['r_hat'])

    def test_convergence_warnings_structure(self, sampler):
        result = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=2000, n_warmup=500, n_chains=2, seed=42,
        )
        diag = result['diagnostics']
        assert 'warnings' in diag
        assert 'converged' in diag
        assert isinstance(diag['warnings'], list)
        assert isinstance(diag['converged'], bool)


# =====================================================================
# Strong prior effects
# =====================================================================

class TestPriorEffects:
    """Test that priors actually influence the posterior."""

    def test_strong_laplace_shrinks_toward_zero(self, sample_df):
        """Tight Laplace(0, 0.1) should shrink coefficients toward 0."""
        sampler = MCMCSampler(sample_df)
        weak = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            priors=None,  # default Normal(0,10)
            n_iter=2000, n_warmup=500, n_chains=1, seed=42,
        )
        tight = sampler.run(
            ['x1', 'x2', 'x3'], 'y',
            priors={
                'Intercept': Prior('normal', loc=0, scale=10),
                'x1': Prior('laplace', loc=0, scale=0.1),
                'x2': Prior('laplace', loc=0, scale=0.1),
                'x3': Prior('laplace', loc=0, scale=0.1),
                'sigma2': Prior('inverse_gamma', a=1, b=1),
            },
            n_iter=2000, n_warmup=500, n_chains=1, seed=42,
        )
        # Tight Laplace should pull coefficients closer to 0
        for i in range(1, 4):  # skip intercept
            assert abs(tight['posterior_mean'][i]) < abs(weak['posterior_mean'][i]) + 0.5

    def test_informative_prior_shifts_posterior(self):
        """Strong prior at extreme value should visibly shift posterior vs weak prior."""
        # Use small dataset so prior has real influence
        np.random.seed(99)
        n = 30
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 5, n),
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] + np.random.normal(0, 1, n)

        sampler = MCMCSampler(df)
        weak = sampler.run(
            ['x1', 'x2'], 'y',
            priors=None,  # default Normal(0,10)
            n_iter=2000, n_warmup=500, n_chains=1, seed=42,
        )
        # Prior pulls x1 toward 0 very strongly (scale=0.1 in standardised space)
        strong_zero = sampler.run(
            ['x1', 'x2'], 'y',
            priors={
                'Intercept': Prior('normal', loc=0, scale=10),
                'x1': Prior('normal', loc=0.0, scale=0.1),  # very strong pull toward 0
                'x2': Prior('normal', loc=0, scale=10),
                'sigma2': Prior('inverse_gamma', a=1, b=1),
            },
            n_iter=2000, n_warmup=500, n_chains=1, seed=42,
        )
        # Strong zero-prior should pull x1 coef toward 0 vs weak prior
        assert abs(strong_zero['posterior_mean'][1]) < abs(weak['posterior_mean'][1])


# =====================================================================
# File I/O tests
# =====================================================================

class TestMCMCPriorFiles:
    """Test save/load of MCMC prior JSON files."""

    def test_save_and_load_round_trip(self, tmp_path):
        priors = {
            'Intercept': Prior('normal', loc=0, scale=10),
            'x1': Prior('laplace', loc=0, scale=1),
            'x2': Prior('student_t', loc=1, scale=2, df=5),
            'sigma2': Prior('inverse_gamma', a=2, b=3),
        }
        path = tmp_path / 'test_mcmc_prior.json'
        MCMCSampler.save_mcmc_prior(
            path, priors,
            features=['x1', 'x2'], target='y',
            name='Test MCMC prior',
        )
        assert path.exists()

        loaded = MCMCSampler.load_mcmc_prior(path)
        assert set(loaded.keys()) == {'Intercept', 'x1', 'x2', 'sigma2'}
        assert loaded['x1'].distribution == 'laplace'
        assert loaded['x2'].distribution == 'student_t'
        assert loaded['x2'].df == 5.0
        assert loaded['sigma2'].a == 2.0
        assert loaded['sigma2'].b == 3.0

    def test_save_creates_directory(self, tmp_path):
        path = tmp_path / 'subdir' / 'deep' / 'prior.json'
        priors = {'Intercept': Prior('normal', loc=0, scale=1)}
        MCMCSampler.save_mcmc_prior(path, priors)
        assert path.exists()

    def test_load_missing_priors_key_raises(self, tmp_path):
        bad_file = tmp_path / 'bad.json'
        bad_file.write_text(json.dumps({"name": "bad file"}))
        with pytest.raises(ValueError, match="missing 'priors'"):
            MCMCSampler.load_mcmc_prior(bad_file)

    def test_load_preset_laplace(self):
        """Load the shipped mcmc_sparse_laplace.json preset."""
        path = Path('test_data/priors/mcmc_sparse_laplace.json')
        if not path.exists():
            pytest.skip("Preset file not found")
        priors = MCMCSampler.load_mcmc_prior(path)
        assert 'Intercept' in priors
        assert priors['predictor_1'].distribution == 'laplace'
        assert 'sigma2' in priors

    def test_load_preset_student_t(self):
        """Load the shipped mcmc_robust_student_t.json preset."""
        path = Path('test_data/priors/mcmc_robust_student_t.json')
        if not path.exists():
            pytest.skip("Preset file not found")
        priors = MCMCSampler.load_mcmc_prior(path)
        assert priors['predictor_1'].distribution == 'student_t'
        assert priors['predictor_1'].df == 3.0

    def test_loaded_preset_works_in_sampler(self):
        """End-to-end: load preset → run MCMC on real data."""
        csv_path = Path('test_data/bayesian_uncertainty_data.csv')
        prior_path = Path('test_data/priors/mcmc_sparse_laplace.json')
        if not csv_path.exists() or not prior_path.exists():
            pytest.skip("Test data or preset file not found")

        df = pd.read_csv(csv_path)
        priors = MCMCSampler.load_mcmc_prior(prior_path)
        sampler = MCMCSampler(df)
        result = sampler.run(
            ['predictor_1', 'predictor_2', 'predictor_3'],
            'response_homoscedastic',
            priors=priors,
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert 'error' not in result
        assert result['r2'] > 0.3


# =====================================================================
# BayesianAnalysis wrapper test
# =====================================================================

class TestMCMCWrapper:
    """Test the mcmc_regression() convenience method on BayesianAnalysis."""

    def test_mcmc_regression_via_wrapper(self, sample_df):
        from data_toolkit import BayesianAnalysis
        ba = BayesianAnalysis(sample_df)
        result = ba.mcmc_regression(
            ['x1', 'x2', 'x3'], 'y',
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert 'error' not in result
        assert 'posterior_mean' in result
        assert 'chains' in result
        assert result['r2'] > 0.5

    def test_mcmc_regression_with_priors(self, sample_df):
        from data_toolkit import BayesianAnalysis
        ba = BayesianAnalysis(sample_df)
        priors = {
            'Intercept': Prior('normal', loc=0, scale=10),
            'x1': Prior('laplace', loc=0, scale=2),
            'x2': Prior('laplace', loc=0, scale=2),
            'x3': Prior('laplace', loc=0, scale=2),
            'sigma2': Prior('inverse_gamma', a=1, b=1),
        }
        result = ba.mcmc_regression(
            ['x1', 'x2', 'x3'], 'y',
            priors=priors,
            n_iter=1500, n_warmup=500, n_chains=1, seed=42,
        )
        assert 'error' not in result
        assert result['priors_used']['x1']['distribution'] == 'laplace'

    def test_mcmc_regression_no_data(self):
        from data_toolkit import BayesianAnalysis
        ba = BayesianAnalysis()
        result = ba.mcmc_regression(['x1'], 'y')
        assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
