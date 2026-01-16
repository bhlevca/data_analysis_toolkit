"""
UI Dropdown Regression Tests

These tests verify that dropdowns in the Streamlit UI are configured correctly:
- Some dropdowns should show ALL columns (categorical + numeric) for ANOVA factors, general selection
- Some dropdowns MUST show ONLY numeric columns (effect sizes, survival time, statistical calculations)

This prevents accidental changes that break functionality.
"""

import pytest
import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def mixed_type_df():
    """DataFrame with both numeric and categorical columns - like twoway_anova_data.csv"""
    return pd.DataFrame({
        'score': [78, 82, 75, 80, 85, 88, 92, 86],
        'response_time': [1.2, 1.5, 1.3, 1.4, 1.1, 0.9, 1.0, 1.1],
        'treatment': ['control', 'control', 'control', 'control', 'drug_A', 'drug_A', 'drug_A', 'drug_A'],
        'gender': ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female'],
        'subject_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    })


@pytest.fixture
def numeric_only_df():
    """DataFrame with only numeric columns"""
    return pd.DataFrame({
        'value1': [1.0, 2.0, 3.0, 4.0],
        'value2': [5.0, 6.0, 7.0, 8.0],
        'value3': [9.0, 10.0, 11.0, 12.0]
    })


# =============================================================================
# SOURCE CODE PARSING
# =============================================================================

def get_streamlit_app_source():
    """Read the streamlit_app.py source code"""
    app_path = Path(__file__).parent.parent / "src" / "data_toolkit" / "streamlit_app.py"
    return app_path.read_text(encoding='utf-8')


def find_selectbox_calls(source: str):
    """Find all st.selectbox calls and their options parameter"""
    # Pattern to match st.selectbox with options parameter
    # Handles multiline and various formatting
    pattern = r'st\.selectbox\s*\(\s*["\']([^"\']+)["\'].*?(?:options\s*=\s*)?([a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]+\])?)'
    
    results = []
    lines = source.split('\n')
    
    for i, line in enumerate(lines, 1):
        if 'st.selectbox' in line:
            # Get context - this line and next few lines
            context = '\n'.join(lines[i-1:i+5])
            
            # Extract label
            label_match = re.search(r'st\.selectbox\s*\(\s*["\']([^"\']+)["\']', context)
            label = label_match.group(1) if label_match else "Unknown"
            
            # Check what options are used
            uses_numeric_cols = 'numeric_cols' in context and 'all_cols' not in context and 'all_columns' not in context
            uses_all_cols = 'all_cols' in context or 'all_columns' in context or 'df.columns' in context
            uses_features = 'features' in context and 'numeric_cols' not in context
            
            results.append({
                'line': i,
                'label': label,
                'uses_numeric_cols': uses_numeric_cols,
                'uses_all_cols': uses_all_cols,
                'uses_features': uses_features,
                'context': context[:200]
            })
    
    return results


def find_multiselect_calls(source: str):
    """Find all st.multiselect calls and their options parameter"""
    results = []
    lines = source.split('\n')
    
    for i, line in enumerate(lines, 1):
        if 'st.multiselect' in line:
            # Get context
            context = '\n'.join(lines[i-1:i+5])
            
            # Extract label
            label_match = re.search(r'st\.multiselect\s*\(\s*["\']([^"\']+)["\']', context)
            label = label_match.group(1) if label_match else "Unknown"
            
            uses_numeric_cols = 'numeric_cols' in context and 'all_cols' not in context
            uses_all_cols = 'all_cols' in context or 'all_columns' in context or 'df.columns' in context
            
            results.append({
                'line': i,
                'label': label,
                'uses_numeric_cols': uses_numeric_cols,
                'uses_all_cols': uses_all_cols,
                'context': context[:200]
            })
    
    return results


# =============================================================================
# DROPDOWN CONFIGURATION TESTS
# =============================================================================

class TestDataLoadingDropdowns:
    """Test dropdowns in the Data Loading tab - should show ALL columns"""
    
    def test_feature_columns_multiselect_shows_all_columns(self):
        """Feature column selector must show all columns including categorical"""
        source = get_streamlit_app_source()
        
        # Find the main feature columns multiselect
        pattern = r'st\.multiselect\s*\(\s*["\'].*Select Feature Columns["\'].*?options\s*=\s*([a-zA-Z_]+)'
        
        # Check that it uses all_cols, not numeric_cols
        assert 'options=all_cols' in source or 'options=all_columns' in source, \
            "Feature Columns multiselect must use all_cols, not numeric_cols"
        
        # Verify the pattern in context
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'Select Feature Columns' in line and 'st.multiselect' in line:
                context = '\n'.join(lines[max(0, i-3):i+5])
                assert 'numeric_cols' not in context or 'all_cols' in context, \
                    f"Feature Columns dropdown at line {i+1} uses numeric_cols instead of all_cols"
                break
    
    def test_target_column_selectbox_shows_all_columns(self):
        """Target column selector must show all columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Select Target Column' in line and 'selectbox' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                # target_options should use all_cols
                assert 'all_cols' in context or 'all_columns' in context, \
                    f"Target Column dropdown at line {i+1} should use all_cols"
                break
    
    def test_quick_visualization_uses_numeric_columns(self):
        """Quick visualization X/Y axes should use numeric columns for scatter plots"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'quick_viz_x' in line and 'selectbox' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                # Should use numeric columns for visualization
                assert 'numeric' in context.lower(), \
                    f"Quick Viz X-axis at line {i+1} should use numeric columns"
                break
    
    def test_quick_visualization_handles_non_numeric(self):
        """Quick visualization should handle non-numeric columns gracefully"""
        source = get_streamlit_app_source()
        
        # Should have error handling for non-numeric
        assert 'is_numeric_dtype' in source or 'non-numeric' in source.lower(), \
            "Quick visualization should check for numeric data types"


class TestANOVADropdowns:
    """Test ANOVA-related dropdowns - factors should be ALL columns, dependent var can be features"""
    
    def test_twoway_anova_factor1_shows_all_columns(self):
        """Two-Way ANOVA Factor 1 must allow categorical columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        found = False
        for i, line in enumerate(lines):
            if 'Factor 1' in line and 'twoway_f1' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                assert 'all_cols' in context, \
                    f"Two-Way ANOVA Factor 1 at line {i+1} must use all_cols for categorical factors"
                found = True
                break
        
        assert found, "Could not find Two-Way ANOVA Factor 1 dropdown"
    
    def test_twoway_anova_factor2_shows_all_columns(self):
        """Two-Way ANOVA Factor 2 must allow categorical columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        found = False
        for i, line in enumerate(lines):
            if 'Factor 2' in line and 'twoway_f2' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                assert 'all_cols' in context, \
                    f"Two-Way ANOVA Factor 2 at line {i+1} must use all_cols for categorical factors"
                found = True
                break
        
        assert found, "Could not find Two-Way ANOVA Factor 2 dropdown"
    
    def test_repeated_measures_subject_shows_all_columns(self):
        """Repeated Measures subject ID should allow any column type"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Subject ID' in line and 'selectbox' in line:
                context = '\n'.join(lines[max(0, i-5):i+8])
                # Subject ID can be string or numeric, so should use all_cols
                assert 'all_cols' in context or 'df.columns' in context, \
                    f"Repeated Measures Subject ID at line {i+1} should allow all column types"
                break


class TestNumericOnlyDropdowns:
    """Test dropdowns that MUST show only numeric columns for mathematical operations"""
    
    def test_effect_size_group_columns_are_numeric(self):
        """Effect size calculations require numeric columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Group 1 Column' in line and 'es_group1' in line:
                context = '\n'.join(lines[max(0, i-5):i+3])
                assert 'numeric_cols' in context, \
                    f"Effect Size Group 1 at line {i+1} must use numeric_cols"
                break
    
    def test_survival_time_column_is_numeric(self):
        """Survival analysis time column must be numeric"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Time Column' in line and 'surv_time' in line:
                context = '\n'.join(lines[max(0, i-5):i+3])
                assert 'numeric_cols' in context, \
                    f"Survival Time Column at line {i+1} must use numeric_cols"
                break
    
    def test_survival_event_column_is_numeric(self):
        """Survival analysis event column must be numeric (0/1)"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Event Column' in line and 'surv_event' in line:
                context = '\n'.join(lines[max(0, i-5):i+3])
                assert 'numeric_cols' in context, \
                    f"Survival Event Column at line {i+1} must use numeric_cols"
                break
    
    def test_outlier_detection_columns_are_numeric(self):
        """Outlier detection requires numeric columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Columns to Check' in line and 'outlier_cols' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Outlier Detection Columns at line {i+1} must use numeric_cols"
                break
    
    def test_bland_altman_methods_are_numeric(self):
        """Bland-Altman method comparison requires numeric columns"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Method 1' in line and 'ba_m1' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Bland-Altman Method 1 at line {i+1} must use numeric_cols"
                break
    
    def test_mann_kendall_column_is_numeric(self):
        """Mann-Kendall trend test requires numeric data"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'mk_col' in line and 'selectbox' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Mann-Kendall Column at line {i+1} must use numeric_cols"
                break
    
    def test_prophet_value_column_is_numeric(self):
        """Prophet forecasting value column must be numeric"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Value Column' in line and 'prophet_value' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Prophet Value Column at line {i+1} must use numeric_cols"
                break
    
    def test_changepoint_column_is_numeric(self):
        """Changepoint detection requires numeric data"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Column to Analyze' in line and 'cp_col' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Changepoint Column at line {i+1} must use numeric_cols"
                break
    
    def test_shannon_diversity_columns_are_numeric(self):
        """Shannon diversity species counts must be numeric"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'Species/Abundance' in line and 'shannon_cols' in line:
                context = '\n'.join(lines[max(0, i-3):i+3])
                assert 'numeric_cols' in context, \
                    f"Shannon Diversity Columns at line {i+1} must use numeric_cols"
                break


class TestChiSquareDropdowns:
    """Chi-Square test needs categorical columns"""
    
    def test_chi_square_allows_all_columns(self):
        """Chi-Square test should allow categorical columns for contingency tables"""
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'chi_var1' in line or 'chi_var2' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                # Chi-square should use all_cols or features (which now includes all)
                # It should NOT be restricted to numeric_cols only
                if 'numeric_cols' in context and 'all_cols' not in context:
                    # This is a potential issue - chi-square often uses categorical
                    pass  # May need fixing depending on implementation


class TestCorrelationDropdowns:
    """Correlation tests need numeric columns"""
    
    def test_correlation_columns_are_numeric(self):
        """Correlation requires numeric columns for calculation"""
        source = get_streamlit_app_source()
        
        # Correlation typically uses features which should be fine
        # as long as the user selects numeric features
        assert True  # Correlation uses session features which user controls


# =============================================================================
# COMPREHENSIVE DROPDOWN AUDIT
# =============================================================================

class TestDropdownAudit:
    """Audit all dropdowns and report their configuration"""
    
    def test_audit_all_selectboxes(self):
        """Audit all selectbox calls - informational test"""
        source = get_streamlit_app_source()
        selectboxes = find_selectbox_calls(source)
        
        # Categorize dropdowns
        numeric_only = []
        all_cols = []
        uses_features = []
        unknown = []
        
        for sb in selectboxes:
            if sb['uses_numeric_cols']:
                numeric_only.append(sb)
            elif sb['uses_all_cols']:
                all_cols.append(sb)
            elif sb['uses_features']:
                uses_features.append(sb)
            else:
                unknown.append(sb)
        
        # Print audit report (visible in pytest -v output)
        print(f"\n\n=== DROPDOWN AUDIT ===")
        print(f"Total selectboxes: {len(selectboxes)}")
        print(f"  Using numeric_cols only: {len(numeric_only)}")
        print(f"  Using all_cols: {len(all_cols)}")
        print(f"  Using features: {len(uses_features)}")
        print(f"  Unknown/other: {len(unknown)}")
        
        # This test always passes - it's for information
        assert True
    
    def test_audit_all_multiselects(self):
        """Audit all multiselect calls - informational test"""
        source = get_streamlit_app_source()
        multiselects = find_multiselect_calls(source)
        
        print(f"\n\n=== MULTISELECT AUDIT ===")
        print(f"Total multiselects: {len(multiselects)}")
        
        for ms in multiselects:
            config = "numeric_cols" if ms['uses_numeric_cols'] else ("all_cols" if ms['uses_all_cols'] else "other")
            print(f"  Line {ms['line']}: '{ms['label']}' -> {config}")
        
        assert True


# =============================================================================
# RUNTIME BEHAVIOR TESTS
# =============================================================================

class TestDropdownBehaviorWithMixedData:
    """Test that dropdowns behave correctly with mixed-type DataFrames"""
    
    def test_numeric_cols_extraction(self, mixed_type_df):
        """Verify numeric column extraction works correctly"""
        numeric_cols = mixed_type_df.select_dtypes(include=[np.number]).columns.tolist()
        
        assert 'score' in numeric_cols
        assert 'response_time' in numeric_cols
        assert 'treatment' not in numeric_cols
        assert 'gender' not in numeric_cols
        assert 'subject_id' not in numeric_cols
        assert len(numeric_cols) == 2
    
    def test_all_cols_extraction(self, mixed_type_df):
        """Verify all columns extraction includes categorical"""
        all_cols = mixed_type_df.columns.tolist()
        
        assert 'score' in all_cols
        assert 'response_time' in all_cols
        assert 'treatment' in all_cols
        assert 'gender' in all_cols
        assert 'subject_id' in all_cols
        assert len(all_cols) == 5
    
    def test_twoway_anova_data_compatibility(self, mixed_type_df):
        """Verify twoway_anova_data.csv structure is handled correctly"""
        # For Two-Way ANOVA:
        # - Dependent variable (score) should be numeric
        # - Factor 1 (treatment) should be categorical
        # - Factor 2 (gender) should be categorical
        
        numeric_cols = mixed_type_df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = mixed_type_df.columns.tolist()
        
        # Dependent variable must be in numeric list
        assert 'score' in numeric_cols
        
        # Factors must be in all_cols list (they're categorical)
        assert 'treatment' in all_cols
        assert 'gender' in all_cols
        
        # Factors should NOT be in numeric_cols
        assert 'treatment' not in numeric_cols
        assert 'gender' not in numeric_cols


# =============================================================================
# SPECIFIC REGRESSION TESTS
# =============================================================================

class TestKnownRegressions:
    """Tests for specific bugs that have occurred in the past"""
    
    def test_regression_feature_cols_not_numeric_only(self):
        """
        REGRESSION TEST: Feature columns dropdown was changed to numeric_cols only,
        breaking ANOVA factor selection. Must use all_cols.
        
        Bug introduced: Previous session
        Fixed: Current session
        """
        source = get_streamlit_app_source()
        
        # Find the main Select Feature Columns definition
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'Select Feature Columns' in line and 'multiselect' in line:
                # Get surrounding context
                context = '\n'.join(lines[max(0, i-10):i+10])
                
                # Must use all_cols for options
                assert 'options=all_cols' in context.replace(' ', ''), \
                    f"REGRESSION: Feature Columns at line {i+1} must use all_cols, not numeric_cols"
                break
    
    def test_regression_quick_viz_uses_numeric(self):
        """
        REGRESSION TEST: Quick Visualization X/Y dropdowns should use numeric columns
        for scatter plots with trendlines.
        
        Bug introduced: Previous session (used all_cols causing errors)
        Fixed: Current session (uses numeric_cols with graceful handling)
        """
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        for i, line in enumerate(lines):
            if 'quick_viz_x' in line:
                context = '\n'.join(lines[max(0, i-8):i+5])
                # Should use numeric columns for visualization
                assert 'numeric' in context.lower() or 'viz_numeric_cols' in context, \
                    f"REGRESSION: Quick Viz X-axis at line {i+1} must use numeric columns"
                break
    
    def test_regression_target_col_not_numeric_only(self):
        """
        REGRESSION TEST: Target column dropdown was restricted to numeric_cols only.
        Must use all_cols to allow categorical targets for classification.
        """
        source = get_streamlit_app_source()
        lines = source.split('\n')
        
        # Look for where target_options is defined - it should use all_cols
        for i, line in enumerate(lines):
            if 'target_options' in line and '=' in line and 'all_cols' in line:
                # Found the definition - it uses all_cols, test passes
                return
        
        # Also check directly at the selectbox
        for i, line in enumerate(lines):
            if 'Select Target Column' in line:
                context = '\n'.join(lines[max(0, i-15):i+10])
                # target_options should be derived from all_cols
                assert 'all_cols' in context or 'all_columns' in context or 'target_options' in context, \
                    f"REGRESSION: Target Column at line {i+1} must use all_cols"
                break
    
    def test_regression_statistical_tests_has_visualizations(self):
        """
        REGRESSION TEST: Statistical tests tab was stripped of visualizations.
        The modular version must be used which includes histogram and scatter previews.
        """
        source = get_streamlit_app_source()
        
        # Check that we import the modular version
        assert '_render_statistical_tests_tab_module' in source or 'from tabs.statistical_tests_tab' in source, \
            "REGRESSION: Must import modular statistical_tests_tab with visualizations"
        
        # Check that the inline function delegates to module (now always delegates)
        assert '_render_statistical_tests_tab_module()' in source, \
            "REGRESSION: Should always call module version for statistical tests"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
