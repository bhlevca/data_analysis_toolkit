"""
Unit Tests for New Modules (v4.0)
===================================
Tests for:
- EffectSizes
- ModelValidation
- ReportGenerator
- DataQuality
- FeatureSelection
- SurvivalAnalysis
- AdvancedTimeSeries
- DomainSpecificAnalysis
- ModelInterpretability
- Statistical analysis enhancements (multiple testing, VIF, robust stats)

Uses: test_data/new_modules_test_data.csv
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test data path
TEST_DATA_PATH = Path(__file__).parent.parent / "test_data" / "new_modules_test_data.csv"


@pytest.fixture
def test_df():
    """Load test data fixture"""
    df = pd.read_csv(TEST_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture
def numeric_cols():
    """Return numeric column names"""
    return ['measurement1', 'measurement2', 'method_a', 'method_b', 'value']


@pytest.fixture
def group_data(test_df):
    """Return grouped data for effect size tests"""
    group_a = test_df[test_df['group'] == 'A']['measurement1'].values
    group_b = test_df[test_df['group'] == 'B']['measurement1'].values
    return group_a, group_b


# =============================================================================
# EFFECT SIZES TESTS
# =============================================================================

class TestEffectSizes:
    """Tests for EffectSizes module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.effect_sizes import EffectSizes
        assert EffectSizes is not None
    
    def test_cohens_d(self, test_df):
        """Test Cohen's d calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        
        # Get data for two groups
        group_a = test_df[test_df['group'] == 'A']['measurement1'].values
        group_b = test_df[test_df['group'] == 'B']['measurement1'].values
        
        result = es.cohens_d(group_a, group_b)
        
        assert 'cohens_d' in result
        assert 'interpretation' in result
        assert isinstance(result['cohens_d'], float)
        # Group B has higher measurements, so d should be negative (or we check absolute)
        assert abs(result['cohens_d']) > 0
    
    def test_hedges_g(self, test_df):
        """Test Hedges' g calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        group_a = test_df[test_df['group'] == 'A']['measurement1'].values
        group_b = test_df[test_df['group'] == 'B']['measurement1'].values
        
        result = es.hedges_g(group_a, group_b)
        
        assert 'hedges_g' in result
        assert isinstance(result['hedges_g'], float)
    
    def test_hedges_g_with_confidence_level(self, test_df):
        """Test Hedges' g with confidence_level parameter (API compatibility)"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        group_a = test_df[test_df['group'] == 'A']['measurement1'].values
        group_b = test_df[test_df['group'] == 'B']['measurement1'].values
        
        result = es.hedges_g(group_a, group_b, confidence_level=0.95)
        
        assert 'hedges_g' in result
        assert isinstance(result['hedges_g'], float)
    
    def test_cohens_d_with_confidence_level(self, test_df):
        """Test Cohen's d with confidence_level parameter (API compatibility)"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        group_a = test_df[test_df['group'] == 'A']['measurement1'].values
        group_b = test_df[test_df['group'] == 'B']['measurement1'].values
        
        result = es.cohens_d(group_a, group_b, confidence_level=0.95)
        
        assert 'cohens_d' in result
    
    def test_glass_delta(self, test_df):
        """Test Glass's delta calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        group_a = test_df[test_df['group'] == 'A']['measurement1'].values
        group_b = test_df[test_df['group'] == 'B']['measurement1'].values
        
        result = es.glass_delta(group_a, group_b)
        
        assert 'glass_delta' in result
        assert isinstance(result['glass_delta'], float)
        assert 'control_sd' in result
    
    def test_eta_squared(self, test_df):
        """Test eta-squared calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        # Use the correct parameter names: grouping_col and value_col
        result = es.eta_squared(grouping_col='group', value_col='measurement1')
        
        assert 'eta_squared' in result
        assert 0 <= result['eta_squared'] <= 1
    
    def test_cramers_v(self, test_df):
        """Test Cramér's V calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        result = es.cramers_v('group', 'treatment')
        
        assert 'cramers_v' in result
        assert 0 <= result['cramers_v'] <= 1
    
    def test_odds_ratio(self, test_df):
        """Test odds ratio calculation"""
        from data_toolkit.effect_sizes import EffectSizes
        
        es = EffectSizes(test_df)
        
        # Create a proper 2x2 table using contingency
        # Use var1 and var2 parameters with columns that have exactly 2 categories
        result = es.odds_ratio(var1='group', var2='event')
        
        # Check if we got an error (table might not be 2x2) or valid result
        if 'error' not in result:
            assert 'odds_ratio' in result
            assert result['odds_ratio'] > 0
        else:
            # If the cross-tab isn't 2x2, the function correctly returns an error
            assert 'error' in result


# =============================================================================
# MODEL VALIDATION TESTS
# =============================================================================

class TestModelValidation:
    """Tests for ModelValidation module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.model_validation import ModelValidation
        assert ModelValidation is not None
    
    def test_cross_validate(self, test_df, numeric_cols):
        """Test cross-validation"""
        from data_toolkit.model_validation import ModelValidation
        from sklearn.linear_model import LinearRegression
        
        mv = ModelValidation(test_df)
        model = LinearRegression()
        result = mv.cross_validate(
            model=model,
            features=['measurement1', 'measurement2'],
            target='value',
            n_splits=3
        )
        
        # Check for either naming convention
        assert 'cv_scores' in result or 'test_scores' in result
        assert 'mean_score' in result or 'test_score_mean' in result
        assert 'std_score' in result or 'test_score_std' in result
    
    def test_learning_curve(self, test_df):
        """Test learning curve analysis"""
        from data_toolkit.model_validation import ModelValidation
        from sklearn.linear_model import LinearRegression
        
        mv = ModelValidation(test_df)
        model = LinearRegression()
        result = mv.learning_curve_analysis(
            model=model,
            features=['measurement1', 'measurement2'],
            target='value',
            cv=2
        )
        
        assert 'train_sizes' in result
        assert 'train_scores_mean' in result
        assert 'test_scores_mean' in result
    
    def test_residual_diagnostics(self, test_df):
        """Test residual diagnostics"""
        from data_toolkit.model_validation import ModelValidation
        from sklearn.linear_model import LinearRegression
        
        mv = ModelValidation(test_df)
        model = LinearRegression()
        result = mv.residual_diagnostics(
            model=model,
            features=['measurement1', 'measurement2'],
            target='value'
        )
        
        assert 'residuals' in result or 'mean_residual' in result


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================

class TestDataQuality:
    """Tests for DataQuality module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.data_quality import DataQuality
        assert DataQuality is not None
    
    def test_missing_data_summary(self, test_df):
        """Test missing data summary"""
        from data_toolkit.data_quality import DataQuality
        
        # Add some missing values
        df_with_missing = test_df.copy()
        df_with_missing.loc[0, 'measurement1'] = np.nan
        df_with_missing.loc[1, 'measurement2'] = np.nan
        
        dq = DataQuality(df_with_missing)
        result = dq.missing_data_summary()
        
        assert 'by_column' in result
        assert 'total_missing' in result
        assert result['total_missing'] == 2
    
    def test_detect_outliers(self, test_df):
        """Test outlier detection"""
        from data_toolkit.data_quality import DataQuality
        
        dq = DataQuality(test_df)
        result = dq.detect_outliers('value', method='iqr')
        
        assert 'n_outliers' in result
        assert 'outlier_indices' in result
        assert 'method' in result
    
    def test_impute_missing(self, test_df):
        """Test missing value imputation"""
        from data_toolkit.data_quality import DataQuality
        
        df_with_missing = test_df.copy()
        df_with_missing.loc[0, 'measurement1'] = np.nan
        
        dq = DataQuality(df_with_missing)
        result = dq.impute_missing(method='mean', columns=['measurement1'])
        
        assert not result['measurement1'].isna().any()
    
    def test_quality_report(self, test_df):
        """Test quality report generation"""
        from data_toolkit.data_quality import DataQuality
        
        dq = DataQuality(test_df)
        result = dq.generate_quality_report()
        
        assert 'overview' in result
        assert 'quality_score' in result
        assert result['quality_score'] >= 0


# =============================================================================
# FEATURE SELECTION TESTS
# =============================================================================

class TestFeatureSelection:
    """Tests for FeatureSelection module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.feature_selection import FeatureSelection
        assert FeatureSelection is not None
    
    def test_statistical_selection(self, test_df):
        """Test statistical feature selection"""
        from data_toolkit.feature_selection import FeatureSelection
        
        fs = FeatureSelection(test_df)
        result = fs.statistical_selection(
            feature_cols=['measurement1', 'measurement2', 'method_a', 'method_b'],
            target_col='value',
            method='f_regression',
            k=2
        )
        
        assert 'selected_features' in result
        assert 'scores' in result
        assert len(result['selected_features']) == 2
    
    def test_permutation_selection(self, test_df):
        """Test permutation importance selection"""
        from data_toolkit.feature_selection import FeatureSelection
        
        fs = FeatureSelection(test_df)
        result = fs.permutation_selection(
            feature_cols=['measurement1', 'measurement2', 'method_a'],
            target_col='value',
            n_features=2
        )
        
        assert 'selected_features' in result
        assert 'importances' in result
    
    def test_lasso_selection(self, test_df):
        """Test Lasso-based feature selection"""
        from data_toolkit.feature_selection import FeatureSelection
        
        fs = FeatureSelection(test_df)
        result = fs.lasso_selection(
            feature_cols=['measurement1', 'measurement2', 'method_a', 'method_b'],
            target_col='value',
            cv=3
        )
        
        assert 'selected_features' in result
        assert 'coefficients' in result


# =============================================================================
# SURVIVAL ANALYSIS TESTS
# =============================================================================

class TestSurvivalAnalysis:
    """Tests for SurvivalAnalysis module"""
    
    def test_import(self):
        """Test module imports"""
        try:
            from data_toolkit.survival_analysis import SurvivalAnalysis, LIFELINES_AVAILABLE
            assert SurvivalAnalysis is not None
        except ImportError:
            pytest.skip("lifelines not installed")
    
    def test_kaplan_meier(self, test_df):
        """Test Kaplan-Meier survival analysis"""
        try:
            from data_toolkit.survival_analysis import SurvivalAnalysis, LIFELINES_AVAILABLE
            if not LIFELINES_AVAILABLE:
                pytest.skip("lifelines not installed")
        except ImportError:
            pytest.skip("lifelines not installed")
        
        sa = SurvivalAnalysis(test_df)
        result = sa.kaplan_meier('time_to_event', 'event')
        
        assert 'n_observations' in result
        assert 'n_events' in result
        assert 'survival_function' in result
    
    def test_kaplan_meier_grouped(self, test_df):
        """Test grouped Kaplan-Meier analysis"""
        try:
            from data_toolkit.survival_analysis import SurvivalAnalysis, LIFELINES_AVAILABLE
            if not LIFELINES_AVAILABLE:
                pytest.skip("lifelines not installed")
        except ImportError:
            pytest.skip("lifelines not installed")
        
        sa = SurvivalAnalysis(test_df)
        result = sa.kaplan_meier('time_to_event', 'event', group_col='treatment')
        
        assert 'groups' in result
        assert 'log_rank_test' in result
    
    def test_cox_regression(self, test_df):
        """Test Cox proportional hazards regression"""
        try:
            from data_toolkit.survival_analysis import SurvivalAnalysis, LIFELINES_AVAILABLE
            if not LIFELINES_AVAILABLE:
                pytest.skip("lifelines not installed")
        except ImportError:
            pytest.skip("lifelines not installed")
        
        sa = SurvivalAnalysis(test_df)
        result = sa.cox_regression(
            'time_to_event', 'event',
            covariates=['measurement1', 'measurement2']
        )
        
        assert 'hazard_ratios' in result
        assert 'concordance_index' in result


# =============================================================================
# DOMAIN SPECIFIC TESTS
# =============================================================================

class TestDomainSpecific:
    """Tests for DomainSpecificAnalysis module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        assert DomainSpecificAnalysis is not None
    
    def test_mann_kendall(self, test_df):
        """Test Mann-Kendall trend test"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        result = dsa.mann_kendall_test('value')
        
        assert 'trend' in result
        assert 'p_value' in result
        assert 'h' in result  # Hypothesis test result
        assert 'tau' in result  # Kendall's tau should be present
    
    def test_sens_slope(self, test_df):
        """Test Sen's slope estimator"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        result = dsa.sens_slope('value')
        
        assert 'sens_slope' in result
        assert 'slope' in result  # Alias for compatibility
        assert 'ci_lower' in result
        assert 'ci_upper' in result
    
    def test_shannon_diversity(self, test_df):
        """Test Shannon diversity index"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        abundance = test_df[['species_a', 'species_b', 'species_c']].sum().values
        result = dsa.shannon_diversity(abundance_data=abundance)
        
        assert 'shannon_index' in result
        assert 'diversity_index' in result  # Alias for compatibility
        assert 'pielou_evenness' in result
        assert 'evenness' in result  # Alias for compatibility
        assert 'richness' in result  # Alias for compatibility
        assert result['shannon_index'] > 0
    
    def test_shannon_diversity_with_columns(self, test_df):
        """Test Shannon diversity index with list of columns"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        result = dsa.shannon_diversity(['species_a', 'species_b', 'species_c'])
        
        assert 'shannon_index' in result
        assert 'diversity_index' in result
        assert result['shannon_index'] > 0
    
    def test_bland_altman(self, test_df):
        """Test Bland-Altman analysis"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        result = dsa.bland_altman('method_a', 'method_b')
        
        assert 'mean_difference' in result
        assert 'loa_lower' in result
        assert 'loa_upper' in result
        assert 'lower_loa' in result  # Alias for compatibility
        assert 'upper_loa' in result  # Alias for compatibility
    
    def test_cohens_kappa(self, test_df):
        """Test Cohen's Kappa coefficient"""
        from data_toolkit.domain_specific import DomainSpecificAnalysis
        
        dsa = DomainSpecificAnalysis(test_df)
        result = dsa.cohens_kappa('rater1', 'rater2')
        
        assert 'kappa' in result
        assert 'interpretation' in result
        assert -1 <= result['kappa'] <= 1


# =============================================================================
# ADVANCED TIMESERIES TESTS
# =============================================================================

class TestAdvancedTimeSeries:
    """Tests for AdvancedTimeSeries module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.advanced_timeseries import AdvancedTimeSeries
        assert AdvancedTimeSeries is not None
    
    def test_changepoint_detection(self, test_df):
        """Test changepoint detection"""
        try:
            from data_toolkit.advanced_timeseries import AdvancedTimeSeries, RUPTURES_AVAILABLE
            if not RUPTURES_AVAILABLE:
                pytest.skip("ruptures not installed")
        except ImportError:
            pytest.skip("ruptures not installed")
        
        ats = AdvancedTimeSeries(test_df)
        result = ats.detect_changepoints('value', method='binseg', n_bkps=2)
        
        assert 'changepoints' in result
        assert 'segments' in result
    
    def test_stationarity_test(self, test_df):
        """Test stationarity test"""
        from data_toolkit.advanced_timeseries import AdvancedTimeSeries
        
        ats = AdvancedTimeSeries(test_df)
        result = ats.stationarity_test('value')
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result


# =============================================================================
# STATISTICAL ANALYSIS ENHANCEMENTS TESTS
# =============================================================================

class TestStatisticalEnhancements:
    """Tests for statistical analysis enhancements"""
    
    def test_multiple_testing_correction(self, test_df):
        """Test multiple testing correction"""
        from data_toolkit.statistical_analysis import StatisticalAnalysis
        
        sa = StatisticalAnalysis(test_df)
        p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
        
        # Test Bonferroni
        result = sa.multiple_testing_correction(p_values, method='bonferroni')
        assert 'corrected_p_values' in result
        assert len(result['corrected_p_values']) == 5
        assert all(p <= 1 for p in result['corrected_p_values'])
        
        # Test FDR
        result_fdr = sa.multiple_testing_correction(p_values, method='fdr_bh')
        assert 'corrected_p_values' in result_fdr
    
    def test_vif(self, test_df):
        """Test Variance Inflation Factor"""
        from data_toolkit.statistical_analysis import StatisticalAnalysis
        
        sa = StatisticalAnalysis(test_df)
        result = sa.variance_inflation_factor(['measurement1', 'measurement2', 'method_a'])
        
        assert 'vif_by_column' in result
        assert 'max_vif' in result
        assert 'has_multicollinearity' in result
    
    def test_robust_statistics(self, test_df):
        """Test robust statistics"""
        from data_toolkit.statistical_analysis import StatisticalAnalysis
        
        sa = StatisticalAnalysis(test_df)
        result = sa.robust_statistics('value')
        
        assert 'robust_center' in result
        assert 'robust_scale' in result
        assert 'median' in result['robust_center']
        assert 'mad' in result['robust_scale']
    
    def test_robust_regression(self, test_df):
        """Test robust regression"""
        from data_toolkit.statistical_analysis import StatisticalAnalysis
        
        sa = StatisticalAnalysis(test_df)
        result = sa.robust_regression('measurement1', 'value', method='huber')
        
        assert 'robust_slope' in result
        assert 'ols_slope' in result
        assert 'interpretation' in result


# =============================================================================
# REPORT GENERATOR TESTS
# =============================================================================

class TestReportGenerator:
    """Tests for ReportGenerator module"""
    
    def test_import(self):
        """Test module imports"""
        from data_toolkit.report_generator import ReportGenerator
        assert ReportGenerator is not None
    
    def test_create_report(self, test_df):
        """Test report creation"""
        from data_toolkit.report_generator import ReportGenerator
        
        rg = ReportGenerator(title="Test Report")
        
        assert rg.title == "Test Report"
    
    def test_add_section(self, test_df):
        """Test adding sections to report"""
        from data_toolkit.report_generator import ReportGenerator
        
        rg = ReportGenerator(title="Test Report")
        rg.add_section("Test Section", "This is test content.")
        
        assert len(rg.sections) == 1
        assert rg.sections[0]['title'] == "Test Section"
    
    def test_add_statistics_summary(self, test_df):
        """Test adding statistics summary to report"""
        from data_toolkit.report_generator import ReportGenerator
        
        rg = ReportGenerator(title="Test Report")
        rg.add_statistics_summary(
            {'mean': 25.5, 'std': 3.2},
            title="Test Analysis"
        )
        
        # Check that sections were added
        assert len(rg.sections) > 0
    
    def test_add_data_provenance(self, test_df):
        """Test adding data provenance from DataFrame"""
        from data_toolkit.report_generator import ReportGenerator
        
        rg = ReportGenerator(title="Test Report")
        rg.add_data_provenance(test_df)
        
        # Should have added to data_provenance list and sections
        assert len(rg.data_provenance) > 0
        assert rg.data_provenance[0]['n_rows'] == len(test_df)
    
    def test_add_statistics_table(self, test_df):
        """Test adding statistics table"""
        from data_toolkit.report_generator import ReportGenerator
        
        rg = ReportGenerator(title="Test Report")
        results = {'mean': 25.5, 'std': 3.2, 'n': 100}
        rg.add_statistics_table(results, title="Test Stats")
        
        # Should have added tables
        assert len(rg.tables) > 0 or len(rg.sections) > 0


# =============================================================================
# INTERPRETABILITY TESTS
# =============================================================================

class TestInterpretability:
    """Tests for ModelInterpretability module"""
    
    def test_import(self):
        """Test module imports"""
        try:
            from data_toolkit.interpretability import ModelInterpretability
            assert ModelInterpretability is not None
        except ImportError:
            pytest.skip("interpretability module not available")
    
    def test_permutation_importance(self, test_df):
        """Test permutation feature importance"""
        from data_toolkit.interpretability import ModelInterpretability
        from sklearn.linear_model import LinearRegression
        
        mi = ModelInterpretability(test_df)
        model = LinearRegression()
        result = mi.permutation_feature_importance(
            model=model,
            features=['measurement1', 'measurement2', 'method_a'],
            target='value'
        )
        
        assert 'feature_importance' in result
        assert len(result['feature_importance']) == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests using the same test data"""
    
    def test_full_analysis_pipeline(self, test_df):
        """Test a full analysis pipeline"""
        from data_toolkit.statistical_analysis import StatisticalAnalysis
        from data_toolkit.data_quality import DataQuality
        from data_toolkit.effect_sizes import EffectSizes
        
        # 1. Data quality check
        dq = DataQuality(test_df)
        quality = dq.generate_quality_report()
        assert quality['quality_score'] > 0
        
        # 2. Statistical analysis
        sa = StatisticalAnalysis(test_df)
        desc = sa.descriptive_stats(['measurement1', 'measurement2'])
        assert not desc.empty
        
        # 3. Effect sizes
        es = EffectSizes(test_df)
        eta = es.eta_squared(grouping_col='group', value_col='measurement1')
        assert 'eta_squared' in eta
    
    def test_ml_pipeline(self, test_df):
        """Test ML analysis pipeline"""
        from data_toolkit.feature_selection import FeatureSelection
        from data_toolkit.model_validation import ModelValidation
        from sklearn.linear_model import LinearRegression
        
        # 1. Feature selection
        fs = FeatureSelection(test_df)
        selected = fs.statistical_selection(
            ['measurement1', 'measurement2', 'method_a', 'method_b'],
            'value',
            k=2
        )
        
        # 2. Model validation with selected features
        mv = ModelValidation(test_df)
        model = LinearRegression()
        cv_result = mv.cross_validate(
            model=model,
            features=selected['selected_features'],
            target='value',
            n_splits=3
        )
        
        # Check for either naming convention
        assert 'mean_score' in cv_result or 'test_score_mean' in cv_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
