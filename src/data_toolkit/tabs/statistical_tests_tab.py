"""
Tab module for the Data Analysis Toolkit
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from statistical_analysis import StatisticalAnalysis

PLOTLY_TEMPLATE = "plotly_white"

def render_statistical_tests_tab():
    """Render advanced statistical tests tab"""
    st.header("🧪 Statistical Hypothesis Tests")
    st.caption("t-tests, ANOVA, Chi-square, normality tests (Shapiro-Wilk), and correlation significance tests")

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first.")
        return

    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select feature columns.")
        return

    df = st.session_state.df
    features = st.session_state.feature_cols
    stats = StatisticalAnalysis(df)

    st.subheader("Test Distributions & PDFs")

    col1, col2 = st.columns(2)

    st.markdown("---")

    st.subheader("Hypothesis Tests")

    test_type = st.selectbox(
        "Test Type",
        ["Compare 2 Groups", "Compare 3+ Groups (ANOVA)", "Two-Way ANOVA", "Repeated-Measures ANOVA", 
         "Post-Hoc Tests", "Chi-Square", "Normality", "Correlation"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if test_type == "Compare 2 Groups":
            col1_test = st.selectbox("Column 1", features, key='col1_test')
            col2_test = st.selectbox("Column 2", [f for f in features if f != col1_test], key='col2_test')
            test_subtype = st.radio("Test", ["Independent t-test", "Paired t-test", "Mann-Whitney U"])

            if st.button("🧪 Run Test", width='stretch'):
                st.info(f"⏳ Running {test_subtype}...")
                with st.spinner(f"Running {test_subtype}..."):
                    try:
                        if test_subtype == "Independent t-test":
                            results = stats.ttest_independent(col1_test, col2_test)
                        elif test_subtype == "Paired t-test":
                            results = stats.ttest_paired(col1_test, col2_test)
                        else:
                            results = stats.mann_whitney_u(col1_test, col2_test)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.success(f"✅ {test_subtype} completed! See results on the right.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Compare 3+ Groups (ANOVA)":
            st.markdown("**One-Way ANOVA**")
            st.caption("Compare means across 3 or more independent groups")
            
            anova_cols = st.multiselect(
                "Select Groups (3+ columns)", 
                features, 
                default=features[:3] if len(features) >= 3 else features,
                key='anova_cols'
            )
            
            if st.button("🧪 Run One-Way ANOVA", width='stretch'):
                if len(anova_cols) >= 3:
                    st.info("⏳ Starting One-Way ANOVA calculation...")
                    with st.spinner("Running One-Way ANOVA... This may take a moment."):
                        try:
                            results = stats.anova_oneway(anova_cols)
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'oneway'
                            st.success("✅ One-Way ANOVA completed! See results on the right.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Select at least 3 groups for ANOVA")
        
        elif test_type == "Two-Way ANOVA":
            st.markdown("**Two-Way Factorial ANOVA**")
            st.caption("Test main effects and interaction of two factors on a continuous outcome.")
            
            st.info("""📊 **Data Structure Required:**
- **Dependent Variable**: The numeric measurement you're analyzing (e.g., 'score', 'response_time')
- **Factor 1**: First categorical grouping variable (e.g., 'treatment': control/drug_A/drug_B)
- **Factor 2**: Second categorical grouping variable (e.g., 'gender': male/female)

📁 Example file: `test_data/twoway_anova_data.csv`""")
            
            all_cols = list(df.columns)
            
            data_col = st.selectbox(
                "📈 Dependent Variable (numeric outcome to analyze)", 
                features, 
                key='twoway_data',
                help="The continuous measurement you want to compare across groups")
            factor1 = st.selectbox(
                "🏷️ Factor 1 (categorical grouping)", 
                [c for c in all_cols if c != data_col], 
                key='twoway_f1',
                help="First categorical variable (e.g., treatment type, condition)")
            factor2 = st.selectbox(
                "🏷️ Factor 2 (categorical grouping)", 
                [c for c in all_cols if c not in [data_col, factor1]], 
                key='twoway_f2',
                help="Second categorical variable (e.g., gender, age group)")
            
            # Show group counts
            if data_col and factor1 and factor2:
                groups = df.groupby([factor1, factor2])[data_col].count()
                st.caption(f"Groups: {len(groups)} combinations, {len(df[factor1].unique())} levels in Factor 1, {len(df[factor2].unique())} levels in Factor 2")
            
            if st.button("🧪 Run Two-Way ANOVA", width='stretch'):
                st.info("⏳ Starting Two-Way ANOVA calculation...")
                with st.spinner("Running Two-Way ANOVA... Computing main effects and interactions."):
                    try:
                        results = stats.anova_twoway(data_col, factor1, factor2)
                        if 'error' in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'twoway'
                            st.success("✅ Two-Way ANOVA completed! See results on the right.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Repeated-Measures ANOVA":
            st.markdown("**Repeated-Measures (Within-Subjects) ANOVA**")
            st.caption("Compare conditions when same subjects are measured multiple times.")
            
            st.info("""📊 **Data Structure Required (Long Format):**
- **Dependent Variable**: The numeric measurement (e.g., 'score')
- **Subject ID Column**: Unique identifier for each subject (e.g., 'subject_id', 'participant')
- **Within-Subjects Factor**: The condition/time point column (e.g., 'time_point': baseline/week_4/week_8)

⚠️ **Each subject must have a measurement for EVERY condition!**
📁 Example file: `test_data/repeated_measures_anova_data.csv`""")
            
            all_cols = list(df.columns)
            
            data_col = st.selectbox(
                "📈 Dependent Variable (numeric measurement)", 
                features, 
                key='rm_data',
                help="The continuous outcome measured at each time point/condition")
            subject_col = st.selectbox(
                "👤 Subject/ID Column (identifies each participant)", 
                [c for c in all_cols if c != data_col], 
                key='rm_subject',
                help="Column that uniquely identifies each subject (e.g., 'subj_01', 'participant_A')")
            within_factor = st.selectbox(
                "⏱️ Within-Subjects Factor (time/condition)", 
                [c for c in all_cols if c not in [data_col, subject_col]], 
                key='rm_within',
                help="Column with conditions/time points (e.g., 'baseline', 'week_4', 'week_8')")
            
            # Show data structure info
            if data_col and subject_col and within_factor:
                n_subjects = df[subject_col].nunique()
                n_conditions = df[within_factor].nunique()
                conditions_list = df[within_factor].unique().tolist()
                st.caption(f"Found: {n_subjects} subjects, {n_conditions} conditions: {conditions_list}")
                
                # Check for complete data
                complete = df.groupby(subject_col)[within_factor].nunique()
                complete_subjects = (complete == n_conditions).sum()
                if complete_subjects < n_subjects:
                    st.warning(f"⚠️ Only {complete_subjects}/{n_subjects} subjects have data for all conditions.")
            
            if st.button("🧪 Run Repeated-Measures ANOVA", width='stretch'):
                st.info("⏳ Starting Repeated-Measures ANOVA calculation...")
                with st.spinner("Running Repeated-Measures ANOVA... Checking data completeness and computing."):
                    try:
                        results = stats.anova_repeated_measures(data_col, subject_col, within_factor)
                        if 'error' in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.session_state.analysis_results['hypothesis_test'] = results
                            st.session_state.analysis_results['anova_type'] = 'repeated'
                            st.success("✅ Repeated-Measures ANOVA completed! See results on the right.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Post-Hoc Tests":
            st.markdown("**Post-Hoc Pairwise Comparisons**")
            st.caption("Follow-up tests after significant ANOVA")
            
            all_cols = list(df.columns)
            
            data_col = st.selectbox("Dependent Variable (numeric)", features, key='posthoc_data')
            group_col = st.selectbox("Grouping Variable", 
                                     [c for c in all_cols if c != data_col], 
                                     key='posthoc_group')
            
            posthoc_type = st.radio("Post-Hoc Method", 
                                    ["Tukey's HSD", "Bonferroni Correction"],
                                    key='posthoc_type')
            
            if st.button("🧪 Run Post-Hoc Tests", width='stretch'):
                st.info("⏳ Computing pairwise comparisons...")
                with st.spinner("Running Post-Hoc tests... Computing pairwise comparisons."):
                    try:
                        if posthoc_type == "Tukey's HSD":
                            results = stats.posthoc_tukey(data_col, group_col)
                        else:
                            results = stats.posthoc_bonferroni(data_col, group_col)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'posthoc'
                        st.success(f"✅ {posthoc_type} completed! See results on the right.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Chi-Square":
            st.markdown("**Chi-Square Test of Independence**")
            st.caption("Tests association between two categorical variables")
            
            all_cols = list(df.columns)
            
            cat_col1 = st.selectbox("First Categorical Variable", all_cols, key='chi_col1',
                                    help="Select the first categorical column")
            cat_col2 = st.selectbox("Second Categorical Variable", 
                                    [c for c in all_cols if c != cat_col1], 
                                    key='chi_col2',
                                    help="Select the second categorical column")
            
            # Show contingency table preview
            if cat_col1 and cat_col2:
                st.markdown("**Contingency Table Preview:**")
                contingency = pd.crosstab(df[cat_col1], df[cat_col2])
                st.dataframe(contingency, width='stretch')
            
            if st.button("🧪 Run Chi-Square Test", width='stretch'):
                with st.spinner("Computing Chi-Square test..."):
                    try:
                        results = stats.chi_square_test(cat_col1, cat_col2)
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'chi_square'
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Normality":
            st.markdown("**Normality Tests**")
            st.caption("Test if data follows a normal distribution (Shapiro-Wilk, D'Agostino-Pearson)")
            
            norm_col = st.selectbox("Select Column to Test", features, key='norm_col',
                                    help="Select a numeric column to test for normality")
            
            norm_test_type = st.radio("Test Type", 
                                      ["Shapiro-Wilk", "D'Agostino-Pearson", "Both"],
                                      key='norm_test_type',
                                      help="Shapiro-Wilk is best for n<5000, D'Agostino for larger samples")
            
            # Show histogram preview
            if norm_col:
                data = df[norm_col].dropna()
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data, name='Data', nbinsx=30))
                fig.update_layout(title=f'Distribution of {norm_col}', 
                                  template=PLOTLY_TEMPLATE, height=300)
                st.plotly_chart(fig, width='stretch')
            
            if st.button("🧪 Run Normality Test", width='stretch'):
                with st.spinner("Testing normality..."):
                    try:
                        from scipy import stats as scipy_stats
                        data = df[norm_col].dropna().values
                        
                        results = {'test': 'Normality Tests', 'column': norm_col}
                        
                        if norm_test_type in ["Shapiro-Wilk", "Both"]:
                            if len(data) <= 5000:
                                stat, p_val = scipy_stats.shapiro(data)
                                results['shapiro_statistic'] = float(stat)
                                results['shapiro_p_value'] = float(p_val)
                                results['shapiro_normal'] = p_val > 0.05
                            else:
                                results['shapiro_warning'] = 'Sample too large for Shapiro-Wilk (n>5000)'
                        
                        if norm_test_type in ["D'Agostino-Pearson", "Both"]:
                            if len(data) >= 20:
                                stat, p_val = scipy_stats.normaltest(data)
                                results['dagostino_statistic'] = float(stat)
                                results['dagostino_p_value'] = float(p_val)
                                results['dagostino_normal'] = p_val > 0.05
                            else:
                                results['dagostino_warning'] = 'Sample too small for D\'Agostino (n<20)'
                        
                        # Add skewness and kurtosis
                        results['skewness'] = float(scipy_stats.skew(data))
                        results['kurtosis'] = float(scipy_stats.kurtosis(data))
                        results['n_samples'] = len(data)
                        
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'normality'
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif test_type == "Correlation":
            st.markdown("**Correlation Significance Tests**")
            st.caption("Test if correlation between two variables is statistically significant")
            
            corr_col1 = st.selectbox("First Variable", features, key='corr_col1')
            corr_col2 = st.selectbox("Second Variable", 
                                     [f for f in features if f != corr_col1], 
                                     key='corr_col2')
            
            corr_method = st.radio("Correlation Method", 
                                   ["Pearson", "Spearman", "Kendall"],
                                   key='corr_method',
                                   help="Pearson for linear, Spearman/Kendall for monotonic relationships")
            
            # Show scatter plot preview
            if corr_col1 and corr_col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[corr_col1], y=df[corr_col2], 
                                         mode='markers', opacity=0.6))
                fig.update_layout(title=f'{corr_col1} vs {corr_col2}',
                                  xaxis_title=corr_col1, yaxis_title=corr_col2,
                                  template=PLOTLY_TEMPLATE, height=300)
                st.plotly_chart(fig, width='stretch')
            
            if st.button("🧪 Run Correlation Test", width='stretch'):
                st.info("⏳ Computing correlation...")
                with st.spinner("Computing correlation..."):
                    try:
                        from scipy import stats as scipy_stats
                        data1 = df[corr_col1].dropna()
                        data2 = df[corr_col2].dropna()
                        
                        # Align data
                        common_idx = data1.index.intersection(data2.index)
                        data1 = data1.loc[common_idx].values
                        data2 = data2.loc[common_idx].values
                        
                        if len(data1) < 3:
                            st.error("Error: Need at least 3 data points for correlation test")
                            return
                        
                        if corr_method == "Pearson":
                            r, p_val = scipy_stats.pearsonr(data1, data2)
                        elif corr_method == "Spearman":
                            r, p_val = scipy_stats.spearmanr(data1, data2)
                        else:  # Kendall
                            r, p_val = scipy_stats.kendalltau(data1, data2)
                        
                        # Interpret strength
                        abs_r = abs(r)
                        if abs_r < 0.1:
                            strength = "negligible"
                        elif abs_r < 0.3:
                            strength = "weak"
                        elif abs_r < 0.5:
                            strength = "moderate"
                        elif abs_r < 0.7:
                            strength = "strong"
                        else:
                            strength = "very strong"
                        
                        direction = "positive" if r > 0 else "negative"
                        
                        results = {
                            'test': f'{corr_method} Correlation',
                            'column1': corr_col1,
                            'column2': corr_col2,
                            'statistic': float(r),
                            'p_value': float(p_val),
                            'significant': p_val < 0.05,
                            'strength': strength,
                            'direction': direction,
                            'n_samples': len(data1),
                            'interpretation': f'{strength.capitalize()} {direction} correlation (r={r:.3f}, p={p_val:.4f})'
                        }
                        
                        st.session_state.analysis_results['hypothesis_test'] = results
                        st.session_state.analysis_results['anova_type'] = 'correlation'
                        st.success(f"✅ {corr_method} Correlation completed! See results on the right.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    st.markdown("---")

    if 'distributions' in st.session_state.analysis_results:
        st.subheader("Distribution Fitting Results")
        dist_results = st.session_state.analysis_results['distributions']

        if 'distributions' in dist_results:
            for dist_name, dist_data in dist_results['distributions'].items():
                with st.expander(f"**{dist_name.upper()}**"):
                    col1, col2 = st.columns(2)
                    col1.metric("Parameters", str(dist_data.get('params', {}))[:50])
                    col2.metric("KS Statistic", f"{dist_data.get('ks_statistic', 0):.4f}")

    if 'hypothesis_test' in st.session_state.analysis_results:
        st.subheader("Test Results")
        test_results = st.session_state.analysis_results['hypothesis_test']
        anova_type = st.session_state.analysis_results.get('anova_type', None)
        
        if 'error' in test_results and test_results['error']:
            st.error(f"Error: {test_results['error']}")
        elif anova_type == 'twoway':
            # Two-Way ANOVA results
            st.markdown(f"**{test_results.get('test', 'Two-Way ANOVA')}**")
            
            col1, col2 = st.columns(2)
            col1.metric("R²", f"{test_results.get('r_squared', 0):.4f}")
            col2.metric("Adj. R²", f"{test_results.get('adj_r_squared', 0):.4f}")
            
            st.markdown("**Effects:**")
            effects = test_results.get('effects', {})
            
            effects_data = []
            for effect_name, effect_info in effects.items():
                effects_data.append({
                    'Effect': effect_info.get('name', effect_name),
                    'Sum of Squares': f"{effect_info.get('sum_sq', 0):.4f}",
                    'df': effect_info.get('df', 0),
                    'F': f"{effect_info.get('F', 0):.4f}",
                    'p-value': f"{effect_info.get('p_value', 0):.4f}",
                    'Significant': '✅' if effect_info.get('significant') else '❌'
                })
            
            if effects_data:
                st.dataframe(pd.DataFrame(effects_data), width='stretch')
            
            st.info(f"📊 {test_results.get('interpretation', '')}")
            
            # === VISUALIZATION: Two-Way ANOVA Interaction Plot ===
            st.markdown("---")
            st.markdown("### 📊 Interaction Plot")
            
            # Get column names from session state
            data_col = st.session_state.get('twoway_data')
            factor1 = st.session_state.get('twoway_f1')
            factor2 = st.session_state.get('twoway_f2')
            
            if data_col and factor1 and factor2:
                if all(c in df.columns for c in [data_col, factor1, factor2]):
                    # Compute means and SE for interaction plot
                    grouped = df.groupby([factor1, factor2])[data_col].agg(['mean', 'sem', 'count']).reset_index()
                    grouped.columns = [factor1, factor2, 'mean', 'sem', 'n']
                    
                    # Create interaction plot
                    fig = go.Figure()
                    
                    # Get unique levels of factor2 for coloring
                    factor2_levels = df[factor2].unique()
                    colors = px.colors.qualitative.Set1[:len(factor2_levels)]
                    
                    for i, level in enumerate(factor2_levels):
                        subset = grouped[grouped[factor2] == level]
                        fig.add_trace(go.Scatter(
                            x=subset[factor1],
                            y=subset['mean'],
                            error_y=dict(type='data', array=subset['sem']),
                            mode='lines+markers',
                            name=f"{factor2}={level}",
                            marker=dict(size=10),
                            line=dict(width=2, color=colors[i % len(colors)])
                        ))
                    
                    fig.update_layout(
                        title=f'Interaction Plot: {data_col} by {factor1} × {factor2}',
                        xaxis_title=factor1,
                        yaxis_title=f'Mean {data_col} (±SE)',
                        template=PLOTLY_TEMPLATE,
                        height=450,
                        legend_title=factor2
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Add grouped bar chart option
                    with st.expander("📊 Grouped Bar Chart"):
                        fig_bar = px.bar(grouped, x=factor1, y='mean', color=factor2,
                                         barmode='group',
                                         error_y='sem',
                                         title=f'{data_col} by {factor1} and {factor2}',
                                         template=PLOTLY_TEMPLATE)
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, width='stretch')
                    
                    # Add main effects box plot
                    with st.expander("📦 Main Effects Box Plots"):
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = px.box(df, x=factor1, y=data_col, color=factor1,
                                         title=f'Effect of {factor1}',
                                         template=PLOTLY_TEMPLATE, points='outliers')
                            fig1.update_layout(height=350, showlegend=False)
                            st.plotly_chart(fig1, width='stretch')
                        with col2:
                            fig2 = px.box(df, x=factor2, y=data_col, color=factor2,
                                         title=f'Effect of {factor2}',
                                         template=PLOTLY_TEMPLATE, points='outliers')
                            fig2.update_layout(height=350, showlegend=False)
                            st.plotly_chart(fig2, width='stretch')
        
        elif anova_type == 'chi_square':
            # Chi-Square test results
            st.markdown(f"**{test_results.get('test', 'Chi-Square Test')}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("χ² Statistic", f"{test_results.get('statistic', 0):.4f}")
            col2.metric("p-value", f"{test_results.get('p_value', 0):.6f}")
            col3.metric("Degrees of Freedom", test_results.get('dof', 0))
            
            # Show effect size (Cramer's V)
            if 'cramers_v' in test_results:
                st.metric("Cramér's V (effect size)", f"{test_results.get('cramers_v', 0):.4f}")
            
            if test_results.get('significant'):
                st.success(f"✅ {test_results.get('interpretation', 'Variables are significantly associated')}")
            else:
                st.info(f"❌ {test_results.get('interpretation', 'No significant association')}")
            
            # === VISUALIZATION: Chi-Square Heatmap ===
            st.markdown("---")
            st.markdown("### 📊 Association Visualizations")
            
            cat_col1 = st.session_state.get('chi_col1')
            cat_col2 = st.session_state.get('chi_col2')
            
            if cat_col1 and cat_col2 and cat_col1 in df.columns and cat_col2 in df.columns:
                # Contingency table heatmap
                contingency = pd.crosstab(df[cat_col1], df[cat_col2])
                
                fig = px.imshow(contingency,
                               labels=dict(x=cat_col2, y=cat_col1, color="Count"),
                               title=f'Contingency Table Heatmap: {cat_col1} × {cat_col2}',
                               color_continuous_scale='Blues',
                               text_auto=True,
                               template=PLOTLY_TEMPLATE)
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
                
                # Stacked/grouped bar chart
                with st.expander("📊 Bar Chart View"):
                    bar_type = st.radio("Bar Type", ["Stacked", "Grouped", "Percent Stacked"], 
                                       horizontal=True, key='chi_bar_type')
                    
                    if bar_type == "Percent Stacked":
                        # Normalize by row
                        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                        fig_bar = px.bar(contingency_pct.reset_index().melt(id_vars=cat_col1),
                                        x=cat_col1, y='value', color=cat_col2,
                                        title=f'Distribution of {cat_col2} within each {cat_col1} (%)',
                                        template=PLOTLY_TEMPLATE,
                                        labels={'value': 'Percentage'})
                    else:
                        fig_bar = px.bar(contingency.reset_index().melt(id_vars=cat_col1),
                                        x=cat_col1, y='value', color=cat_col2,
                                        barmode='stack' if bar_type == "Stacked" else 'group',
                                        title=f'{cat_col1} by {cat_col2}',
                                        template=PLOTLY_TEMPLATE,
                                        labels={'value': 'Count'})
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, width='stretch')
                
                # Expected vs Observed
                with st.expander("📋 Expected vs Observed Frequencies"):
                    from scipy.stats import chi2_contingency
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    
                    st.markdown("**Observed Frequencies:**")
                    st.dataframe(contingency, width='stretch')
                    
                    st.markdown("**Expected Frequencies (under independence):**")
                    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
                    st.dataframe(expected_df.round(2), width='stretch')
                    
                    st.markdown("**Residuals (Observed - Expected):**")
                    residuals = contingency - expected_df
                    st.dataframe(residuals.round(2), width='stretch')
        
        elif anova_type == 'normality':
            # Normality test results
            st.markdown(f"**{test_results.get('test', 'Normality Tests')} - {test_results.get('column', '')}**")
            
            col1, col2 = st.columns(2)
            col1.metric("Skewness", f"{test_results.get('skewness', 0):.4f}")
            col2.metric("Kurtosis", f"{test_results.get('kurtosis', 0):.4f}")
            
            st.metric("N samples", test_results.get('n_samples', 0))
            
            # Shapiro-Wilk results
            if 'shapiro_statistic' in test_results:
                st.markdown("**Shapiro-Wilk Test:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("W Statistic", f"{test_results.get('shapiro_statistic', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('shapiro_p_value', 0):.6f}")
                if test_results.get('shapiro_normal'):
                    col3.success("✅ Normal")
                else:
                    col3.warning("❌ Non-normal")
            elif 'shapiro_warning' in test_results:
                st.warning(test_results['shapiro_warning'])
            
            # D'Agostino results
            if 'dagostino_statistic' in test_results:
                st.markdown("**D'Agostino-Pearson Test:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("K² Statistic", f"{test_results.get('dagostino_statistic', 0):.4f}")
                col2.metric("p-value", f"{test_results.get('dagostino_p_value', 0):.6f}")
                if test_results.get('dagostino_normal'):
                    col3.success("✅ Normal")
                else:
                    col3.warning("❌ Non-normal")
            elif 'dagostino_warning' in test_results:
                st.warning(test_results['dagostino_warning'])
        
        elif anova_type == 'correlation':
            # Correlation test results
            st.markdown(f"**{test_results.get('test', 'Correlation Test')}**")
            st.markdown(f"Testing: **{test_results.get('column1', '')}** vs **{test_results.get('column2', '')}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Correlation (r)", f"{test_results.get('statistic', 0):.4f}")
            col2.metric("p-value", f"{test_results.get('p_value', 0):.6f}")
            col3.metric("N samples", test_results.get('n_samples', 0))
            
            col1, col2 = st.columns(2)
            col1.metric("Strength", test_results.get('strength', '').capitalize())
            col2.metric("Direction", test_results.get('direction', '').capitalize())
            
            if test_results.get('significant'):
                st.success(f"✅ {test_results.get('interpretation', 'Significant correlation')}")
            else:
                st.info(f"❌ Correlation not statistically significant (p ≥ 0.05)")
            
            # === VISUALIZATION: Regression/Correlation Plot with Statistics ===
            st.markdown("---")
            st.markdown("### 📊 Regression Plot with Statistics")
            
            col1_name = test_results.get('column1')
            col2_name = test_results.get('column2')
            
            if col1_name in df.columns and col2_name in df.columns:
                from scipy import stats as scipy_stats
                
                # Get clean data
                plot_df = df[[col1_name, col2_name]].dropna()
                x_data = plot_df[col1_name].values
                y_data = plot_df[col2_name].values
                
                # Compute regression
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
                r_squared = r_value ** 2
                
                # Create scatter plot
                fig = go.Figure()
                
                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='markers',
                    marker=dict(size=8, opacity=0.6, color='steelblue'),
                    name='Data Points'
                ))
                
                # Add regression line
                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                y_line = slope * x_line + intercept
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color='crimson', width=2),
                    name='Regression Line'
                ))
                
                # Add confidence interval (95%)
                n = len(x_data)
                x_mean = np.mean(x_data)
                se_y = std_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_data - x_mean)**2))
                t_val = scipy_stats.t.ppf(0.975, n - 2)
                ci_upper = y_line + t_val * se_y
                ci_lower = y_line - t_val * se_y
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(220, 53, 69, 0.15)',
                    line=dict(color='rgba(220, 53, 69, 0)'),
                    name='95% CI'
                ))
                
                # Format equation and stats
                sign = '+' if intercept >= 0 else ''
                eq_text = f"y = {slope:.4f}x {sign}{intercept:.4f}"
                stats_text = f"R² = {r_squared:.4f}<br>r = {r_value:.4f}<br>p = {p_value:.4g}<br>n = {n}"
                
                # Add annotation with equation and stats
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=f"<b>{eq_text}</b><br>{stats_text}",
                    showarrow=False,
                    align='left',
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=6,
                    font=dict(size=12, family='monospace')
                )
                
                fig.update_layout(
                    title=f'Linear Regression: {col2_name} vs {col1_name}',
                    xaxis_title=col1_name,
                    yaxis_title=col2_name,
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    showlegend=True,
                    legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99)
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Show additional regression info in expander
                with st.expander("📋 Detailed Regression Statistics"):
                    st.markdown(f"""
                    | Statistic | Value |
                    |-----------|-------|
                    | **Slope** | {slope:.6f} |
                    | **Intercept** | {intercept:.6f} |
                    | **R² (coefficient of determination)** | {r_squared:.6f} |
                    | **Standard Error of Slope** | {std_err:.6f} |
                    | **t-statistic for slope** | {slope/std_err:.4f} |
                    | **p-value (slope ≠ 0)** | {p_value:.6g} |
                    """)
                    
                    if p_value < 0.001:
                        sig_text = "highly significant (p < 0.001)"
                    elif p_value < 0.01:
                        sig_text = "very significant (p < 0.01)"
                    elif p_value < 0.05:
                        sig_text = "significant (p < 0.05)"
                    else:
                        sig_text = "not significant (p ≥ 0.05)"
                    
                    st.info(f"📊 The regression slope is **{sig_text}**. "
                            f"For every 1-unit increase in {col1_name}, {col2_name} changes by {slope:.4f} units on average.")
            
        elif anova_type == 'repeated':
            # Repeated-Measures ANOVA results
            st.markdown(f"**{test_results.get('test', 'Repeated-Measures ANOVA')}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("F-statistic", f"{test_results.get('F', 0):.4f}")
            col2.metric("p-value", f"{test_results.get('p_value', 0):.4f}")
            col3.metric("η² (effect size)", f"{test_results.get('partial_eta_squared', 0):.4f}")
            
            col1, col2 = st.columns(2)
            col1.metric("N subjects", test_results.get('n_subjects', 0))
            col2.metric("N conditions", test_results.get('n_conditions', 0))
            
            if test_results.get('sphericity_concern'):
                st.warning("⚠️ Sphericity may be violated - consider Greenhouse-Geisser correction")
            
            if test_results.get('p_value', 1) < 0.05:
                st.success(f"✅ {test_results.get('interpretation', 'Significant')}")
            else:
                st.info(f"❌ {test_results.get('interpretation', 'Not significant')}")
            
            # === VISUALIZATION: Repeated Measures Line Plot ===
            st.markdown("---")
            st.markdown("### 📊 Repeated Measures Visualization")
            
            # Get the column info from session state
            data_col = st.session_state.get('rm_data')
            subject_col = st.session_state.get('rm_subject')
            within_factor = st.session_state.get('rm_within')
            
            if data_col and subject_col and within_factor:
                if all(c in df.columns for c in [data_col, subject_col, within_factor]):
                    # Create line plot showing each subject's trajectory
                    fig = px.line(df, x=within_factor, y=data_col, color=subject_col,
                                 markers=True,
                                 title=f'Repeated Measures: {data_col} across {within_factor}',
                                 template=PLOTLY_TEMPLATE)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Add mean plot with error bars
                    with st.expander("📈 Show Mean ± SE Plot"):
                        means = df.groupby(within_factor)[data_col].agg(['mean', 'sem']).reset_index()
                        fig_mean = go.Figure()
                        fig_mean.add_trace(go.Scatter(
                            x=means[within_factor], y=means['mean'],
                            error_y=dict(type='data', array=means['sem']),
                            mode='lines+markers',
                            name='Mean ± SE',
                            marker=dict(size=10),
                            line=dict(width=2)
                        ))
                        fig_mean.update_layout(
                            title=f'Mean {data_col} by {within_factor} (±SE)',
                            xaxis_title=within_factor,
                            yaxis_title=data_col,
                            template=PLOTLY_TEMPLATE,
                            height=400
                        )
                        st.plotly_chart(fig_mean, width='stretch')
            
        elif anova_type == 'posthoc':
            # Post-hoc test results
            st.markdown(f"**{test_results.get('test', 'Post-Hoc Tests')}**")
            
            col1, col2 = st.columns(2)
            col1.metric("Number of Groups", test_results.get('n_groups', 0))
            col2.metric("Comparisons Made", test_results.get('n_comparisons', 0))
            
            # Group means
            st.markdown("**Group Means:**")
            means_df = pd.DataFrame({
                'Group': list(test_results.get('group_means', {}).keys()),
                'Mean': list(test_results.get('group_means', {}).values()),
                'N': list(test_results.get('group_sizes', {}).values())
            })
            st.dataframe(means_df, width='stretch')
            
            # Pairwise comparisons
            st.markdown("**Pairwise Comparisons:**")
            comparisons = test_results.get('comparisons', [])
            if comparisons:
                comp_df = pd.DataFrame(comparisons)
                # Format for display
                display_cols = ['group1', 'group2', 'mean_diff', 'p_value', 'significant']
                if 'p_adjusted' in comp_df.columns:
                    display_cols = ['group1', 'group2', 'mean_diff', 'p_adjusted', 'significant', 'effect_size']
                comp_df_display = comp_df[display_cols].copy()
                comp_df_display['significant'] = comp_df_display['significant'].map({True: '✅', False: '❌'})
                st.dataframe(comp_df_display, width='stretch')
            
            st.info(f"📊 {test_results.get('interpretation', '')}")
            
            # === VISUALIZATION: Post-Hoc Comparison Chart ===
            st.markdown("---")
            st.markdown("### 📊 Post-Hoc Visualizations")
            
            data_col = st.session_state.get('posthoc_data')
            group_col = st.session_state.get('posthoc_group')
            
            if data_col and group_col and data_col in df.columns and group_col in df.columns:
                # Box plot with individual points
                fig = px.box(df, x=group_col, y=data_col, color=group_col,
                            title=f'Group Distributions: {data_col} by {group_col}',
                            template=PLOTLY_TEMPLATE,
                            points='all')
                fig.update_layout(height=450, showlegend=False)
                
                # Add mean markers
                means = df.groupby(group_col)[data_col].mean()
                for i, (group, mean_val) in enumerate(means.items()):
                    fig.add_trace(go.Scatter(
                        x=[group], y=[mean_val],
                        mode='markers',
                        marker=dict(symbol='diamond', size=12, color='red', line=dict(width=2, color='darkred')),
                        name='Group Mean',
                        showlegend=(i == 0)
                    ))
                
                st.plotly_chart(fig, width='stretch')
                
                # Comparison matrix heatmap
                if comparisons:
                    with st.expander("📊 Pairwise Comparison Matrix"):
                        groups = list(test_results.get('group_means', {}).keys())
                        n_groups = len(groups)
                        
                        # Create p-value matrix
                        p_matrix = np.ones((n_groups, n_groups))
                        for comp in comparisons:
                            g1, g2 = comp.get('group1'), comp.get('group2')
                            p_val = comp.get('p_adjusted', comp.get('p_value', 1))
                            if g1 in groups and g2 in groups:
                                i1, i2 = groups.index(g1), groups.index(g2)
                                p_matrix[i1, i2] = p_val
                                p_matrix[i2, i1] = p_val
                        
                        # Create significance annotation
                        sig_text = [['' for _ in range(n_groups)] for _ in range(n_groups)]
                        for i in range(n_groups):
                            for j in range(n_groups):
                                if i != j:
                                    p = p_matrix[i, j]
                                    if p < 0.001:
                                        sig_text[i][j] = '***'
                                    elif p < 0.01:
                                        sig_text[i][j] = '**'
                                    elif p < 0.05:
                                        sig_text[i][j] = '*'
                                    else:
                                        sig_text[i][j] = 'ns'
                        
                        fig_matrix = go.Figure(data=go.Heatmap(
                            z=p_matrix,
                            x=groups,
                            y=groups,
                            text=sig_text,
                            texttemplate='%{text}',
                            colorscale='RdYlGn_r',
                            zmin=0, zmax=0.1,
                            colorbar=dict(title='p-value')
                        ))
                        
                        fig_matrix.update_layout(
                            title='Pairwise Comparison p-values (* p<.05, ** p<.01, *** p<.001)',
                            template=PLOTLY_TEMPLATE,
                            height=400
                        )
                        st.plotly_chart(fig_matrix, width='stretch')
                
                # Mean difference forest plot
                if comparisons:
                    with st.expander("🌲 Mean Difference Forest Plot"):
                        comp_df = pd.DataFrame(comparisons)
                        comp_df['comparison'] = comp_df['group1'] + ' vs ' + comp_df['group2']
                        comp_df['color'] = comp_df['significant'].map({True: 'Significant', False: 'Not Significant'})
                        
                        fig_forest = go.Figure()
                        
                        # Add mean differences with CI if available
                        for idx, row in comp_df.iterrows():
                            color = 'green' if row['significant'] else 'gray'
                            fig_forest.add_trace(go.Scatter(
                                x=[row['mean_diff']],
                                y=[row['comparison']],
                                mode='markers',
                                marker=dict(size=10, color=color),
                                showlegend=False
                            ))
                        
                        # Add vertical line at 0
                        fig_forest.add_vline(x=0, line_dash='dash', line_color='red')
                        
                        fig_forest.update_layout(
                            title='Mean Differences (green = significant)',
                            xaxis_title='Mean Difference',
                            template=PLOTLY_TEMPLATE,
                            height=max(300, len(comparisons) * 40)
                        )
                        st.plotly_chart(fig_forest, width='stretch')
            
        else:
            # Standard test results (t-test, one-way ANOVA, etc.)
            col1, col2, col3 = st.columns(3)
            col1.metric("Statistic", f"{test_results.get('statistic', 0):.4f}")
            col2.metric("p-value", f"{test_results.get('p_value', 0):.4f}")

            if test_results.get('p_value', 1) < 0.05:
                col3.success("✅ Significant (p < 0.05)")
            else:
                col3.info("❌ Not Significant (p ≥ 0.05)")
            
            if 'interpretation' in test_results:
                st.info(f"📊 {test_results['interpretation']}")
            
            # === VISUALIZATION for t-tests and ANOVA ===
            st.markdown("---")
            st.markdown("### 📊 Visualization")
            
            test_name = test_results.get('test', '')
            
            # Box plot for 2-group comparisons (t-test, Mann-Whitney)
            if 'column1' in test_results and 'column2' in test_results:
                col1_name = test_results.get('column1')
                col2_name = test_results.get('column2')
                
                if col1_name in df.columns and col2_name in df.columns:
                    # Create comparison box plot
                    plot_data = pd.DataFrame({
                        'Value': pd.concat([df[col1_name], df[col2_name]]),
                        'Group': [col1_name] * len(df[col1_name]) + [col2_name] * len(df[col2_name])
                    })
                    
                    fig = px.box(plot_data, x='Group', y='Value', color='Group',
                                title=f'Distribution Comparison: {col1_name} vs {col2_name}',
                                template=PLOTLY_TEMPLATE,
                                points='all')
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
            
            # Box plot for One-Way ANOVA (multiple groups)
            elif 'groups' in test_results or anova_type == 'oneway':
                # Try to get the columns that were tested
                anova_cols = st.session_state.get('anova_cols', [])
                if anova_cols and len(anova_cols) >= 2:
                    # Create melted data for box plot
                    valid_cols = [c for c in anova_cols if c in df.columns]
                    if valid_cols:
                        plot_data = df[valid_cols].melt(var_name='Group', value_name='Value')
                        
                        fig = px.box(plot_data, x='Group', y='Value', color='Group',
                                    title='One-Way ANOVA: Group Distributions',
                                    template=PLOTLY_TEMPLATE,
                                    points='outliers')
                        fig.update_layout(height=450, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                        
                        # Add violin plot option
                        with st.expander("🎻 Show Violin Plot"):
                            fig_violin = px.violin(plot_data, x='Group', y='Value', color='Group',
                                                   box=True, points='all',
                                                   title='One-Way ANOVA: Violin Plot with Box',
                                                   template=PLOTLY_TEMPLATE)
                            fig_violin.update_layout(height=450, showlegend=False)
                            st.plotly_chart(fig_violin, width='stretch')


