"""
Comprehensive project analysis script
"""
import pandas as pd
import os
import ast
import re
from collections import defaultdict

# ============================================================================
# 1. TEST DATA ANALYSIS
# ============================================================================
print("=" * 80)
print("1. TEST DATA ANALYSIS - Checking for inadequate demo data")
print("=" * 80)

test_dir = 'test_data'
files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]

inadequate_files = []
adequate_files = []

for f in sorted(files):
    path = os.path.join(test_dir, f)
    try:
        df = pd.read_csv(path)
        rows, cols = len(df), len(df.columns)
        
        issues = []
        if rows < 50:
            issues.append(f"Too few rows ({rows}) for statistical significance")
        if rows < 100 and 'timeseries' in f.lower():
            issues.append(f"Time series needs more data points ({rows})")
        if rows < 200 and ('ml_' in f or 'neural' in f):
            issues.append(f"ML training needs more samples ({rows})")
        if cols < 3:
            issues.append(f"Too few columns ({cols}) for multivariate analysis")
            
        if issues:
            inadequate_files.append((f, rows, cols, issues))
        else:
            adequate_files.append((f, rows, cols))
    except Exception as e:
        print(f"  Error reading {f}: {e}")

print("\n⚠️ INADEQUATE TEST DATA FILES:")
for f, rows, cols, issues in inadequate_files:
    print(f"\n  {f} ({rows} rows, {cols} cols):")
    for issue in issues:
        print(f"    - {issue}")

print(f"\n✅ Adequate files: {len(adequate_files)}")
print(f"⚠️ Inadequate files: {len(inadequate_files)}")

# ============================================================================
# 2. MODULE COVERAGE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. MODULE COVERAGE - Which modules lack test data?")
print("=" * 80)

src_dir = 'src/data_toolkit'
modules = [f[:-3] for f in os.listdir(src_dir) if f.endswith('.py') and not f.startswith('__')]

# Map test data to modules
test_data_mapping = {
    'statistical_analysis': ['anova_factorial', 'twoway_anova', 'oneway_anova', 'distribution_samples', 'regression_data'],
    'timeseries_analysis': ['timeseries_data', 'seasonal_timeseries', 'timeseries_lstm', 'multivariate_timeseries'],
    'signal_analysis': ['signal_analysis_sample', 'coherence_signals', 'wavelet', 'ccf_test_signals'],
    'ml_models': ['ml_classification', 'ml_regression', 'clustering_data'],
    'neural_networks': ['neural_network_train', 'neural_network_predict'],
    'causality_analysis': ['causality_data'],
    'bayesian_analysis': ['bayesian_uncertainty_data'],
    'nonlinear_analysis': ['nonlinear_data'],
    'survival_analysis': ['survival_demo_data'],
    'data_quality': ['data_quality_demo_data'],
    'effect_sizes': ['effect_sizes_demo_data'],
    'feature_selection': ['feature_selection_demo_data'],
    'interpretability': ['interpretability_demo_data'],
    'domain_specific': ['domain_clinical', 'domain_ecology'],
    'model_validation': ['model_validation_demo_data'],
    'advanced_timeseries': ['advanced_timeseries_demo_data'],
    'image_models': ['digits'],
    'biomass_segmentation': [],  # Needs image data
    'pca_visualization': ['general_analysis_data'],
    'uncertainty_analysis': ['bayesian_uncertainty_data'],
}

print("\nModules without dedicated test data:")
for module in sorted(modules):
    if module not in test_data_mapping or not test_data_mapping.get(module, []):
        if module not in ['streamlit_app', 'utils', '__init__', 'plugin_system', 
                          'comprehensive_tutorial', 'data_loading_methods', 
                          'visualization_methods', 'rust_accelerated', 'image_data']:
            print(f"  ⚠️ {module}.py - No dedicated test data")

# ============================================================================
# 3. UNIT TEST COVERAGE
# ============================================================================
print("\n" + "=" * 80)
print("3. UNIT TEST COVERAGE ANALYSIS")
print("=" * 80)

tests_dir = 'tests'
test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]

print(f"\nExisting test files: {len(test_files)}")
for tf in test_files:
    print(f"  - {tf}")

# Check which modules have tests
tested_modules = set()
for tf in test_files:
    path = os.path.join(tests_dir, tf)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Find imports
        for module in modules:
            if module in content:
                tested_modules.add(module)

untested = set(modules) - tested_modules - {'streamlit_app', '__init__', 'utils', 'tabs'}
print(f"\nModules without test coverage:")
for m in sorted(untested):
    print(f"  ⚠️ {m}.py")

# ============================================================================
# 4. STREAMLIT UI COVERAGE
# ============================================================================
print("\n" + "=" * 80)
print("4. STREAMLIT UI - Functions and tabs without tests")
print("=" * 80)

# Read streamlit_app.py and find all render_ functions
streamlit_path = os.path.join(src_dir, 'streamlit_app.py')
with open(streamlit_path, 'r', encoding='utf-8') as f:
    content = f.read()

render_functions = re.findall(r'def (render_\w+)\(', content)
print(f"\nTotal render functions in streamlit_app.py: {len(render_functions)}")

# Check test_ui_dropdowns.py coverage
ui_test_path = os.path.join(tests_dir, 'test_ui_dropdowns.py')
if os.path.exists(ui_test_path):
    with open(ui_test_path, 'r', encoding='utf-8') as f:
        test_content = f.read()
    
    tested_renders = []
    untested_renders = []
    for func in render_functions:
        if func in test_content:
            tested_renders.append(func)
        else:
            untested_renders.append(func)
    
    print(f"Tested: {len(tested_renders)}, Untested: {len(untested_renders)}")
    if untested_renders:
        print("\nUntested render functions:")
        for func in untested_renders[:20]:  # Show first 20
            print(f"  - {func}")
        if len(untested_renders) > 20:
            print(f"  ... and {len(untested_renders) - 20} more")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
