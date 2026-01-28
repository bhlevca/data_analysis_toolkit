"""
Effect Size Calculations Module
================================
Comprehensive effect size measures for scientific research including:
- Cohen's d, Hedges' g (group comparisons)
- Glass's delta (control group reference)
- Eta-squared, Omega-squared, Partial eta-squared (ANOVA)
- Cramér's V, Phi coefficient (categorical)
- Odds ratios, Risk ratios (epidemiology)
- Correlation-based effect sizes (r, r², adjusted r²)
- Confidence intervals for all measures

Version: 1.0
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as t_dist

warnings.filterwarnings('ignore')


class EffectSizes:
    """Effect size calculations for statistical analyses"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to analyze"""
        self.df = df
    
    # =========================================================================
    # GROUP COMPARISON EFFECT SIZES
    # =========================================================================
    
    def cohens_d(self, group1: Union[str, np.ndarray], group2: Union[str, np.ndarray] = None,
                 grouping_col: str = None, value_col: str = None,
                 pooled: bool = True, confidence: float = 0.95,
                 confidence_level: float = None) -> Dict[str, Any]:
        # Support both confidence and confidence_level parameter names
        if confidence_level is not None:
            confidence = confidence_level
        """
        Calculate Cohen's d effect size for two groups
        
        Cohen's d = (M1 - M2) / SD_pooled
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        
        Args:
            group1: Column name or array for group 1
            group2: Column name or array for group 2
            grouping_col: Alternative - column with group labels
            value_col: Alternative - column with values
            pooled: Use pooled SD (True) or group1 SD (False, Glass's delta)
            confidence: Confidence level for CI
            
        Returns:
            Dictionary with d, CI, interpretation
        """
        # Handle different input formats
        if grouping_col and value_col and self.df is not None:
            groups = self.df[grouping_col].unique()
            if len(groups) != 2:
                return {'error': f'Expected 2 groups, found {len(groups)}'}
            g1 = self.df[self.df[grouping_col] == groups[0]][value_col].dropna().values
            g2 = self.df[self.df[grouping_col] == groups[1]][value_col].dropna().values
        elif isinstance(group1, str) and self.df is not None:
            g1 = self.df[group1].dropna().values
            g2 = self.df[group2].dropna().values if group2 else None
        else:
            g1 = np.asarray(group1)
            g2 = np.asarray(group2) if group2 is not None else None
        
        if g2 is None:
            return {'error': 'Two groups required'}
        
        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        
        if pooled:
            # Pooled standard deviation
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            sd = np.sqrt(pooled_var)
        else:
            # Glass's delta - use control group SD
            sd = np.sqrt(var1)
        
        d = (m1 - m2) / sd if sd > 0 else 0.0
        
        # Standard error and confidence interval (Hedges & Olkin, 1985)
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = d - z * se
        ci_upper = d + z * se
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': float(d),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'interpretation': interpretation,
            'n1': n1,
            'n2': n2,
            'mean1': float(m1),
            'mean2': float(m2),
            'pooled_sd': float(sd),
            'method': 'pooled' if pooled else 'glass_delta'
        }
    
    def hedges_g(self, group1: Union[str, np.ndarray], group2: Union[str, np.ndarray] = None,
                 grouping_col: str = None, value_col: str = None,
                 confidence: float = 0.95, confidence_level: float = None) -> Dict[str, Any]:
        """
        Calculate Hedges' g - bias-corrected Cohen's d
        
        Better for small samples (n < 20). Applies correction factor J.
        
        Args:
            group1, group2: Groups to compare
            confidence: Confidence level
            confidence_level: Alias for confidence (for API compatibility)
            
        Returns:
            Dictionary with g, CI, interpretation
        """
        # Support both confidence and confidence_level parameter names
        if confidence_level is not None:
            confidence = confidence_level
        
        # First get Cohen's d
        d_result = self.cohens_d(group1, group2, grouping_col, value_col, confidence=confidence)
        
        if 'error' in d_result:
            return d_result
        
        d = d_result['cohens_d']
        n1, n2 = d_result['n1'], d_result['n2']
        
        # Hedges' correction factor
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))  # Approximate correction
        
        g = d * j
        
        # Adjusted SE
        se = d_result['se'] * j
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = g - z * se
        ci_upper = g + z * se
        
        # Interpretation (same thresholds as Cohen's d)
        abs_g = abs(g)
        if abs_g < 0.2:
            interpretation = "negligible"
        elif abs_g < 0.5:
            interpretation = "small"
        elif abs_g < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'hedges_g': float(g),
            'cohens_d': float(d),
            'correction_factor': float(j),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'interpretation': interpretation,
            'n1': n1,
            'n2': n2
        }
    
    def paired_cohens_d(self, pre: Union[str, np.ndarray], post: Union[str, np.ndarray],
                        use_pre_sd: bool = False, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Cohen's d for paired/repeated measures
        
        Args:
            pre: Pre-treatment values (column name or array)
            post: Post-treatment values
            use_pre_sd: Use pre-treatment SD (True) or SD of differences (False)
            confidence: Confidence level
            
        Returns:
            Dictionary with d, CI
        """
        if isinstance(pre, str) and self.df is not None:
            pre_vals = self.df[pre].dropna().values
            post_vals = self.df[post].dropna().values
        else:
            pre_vals = np.asarray(pre)
            post_vals = np.asarray(post)
        
        n = min(len(pre_vals), len(post_vals))
        pre_vals = pre_vals[:n]
        post_vals = post_vals[:n]
        
        diff = post_vals - pre_vals
        mean_diff = np.mean(diff)
        
        if use_pre_sd:
            sd = np.std(pre_vals, ddof=1)
        else:
            sd = np.std(diff, ddof=1)
        
        d = mean_diff / sd if sd > 0 else 0.0
        
        # SE for paired design
        se = np.sqrt(1/n + d**2 / (2*n))
        z = norm.ppf((1 + confidence) / 2)
        
        return {
            'cohens_d': float(d),
            'mean_difference': float(mean_diff),
            'sd_used': float(sd),
            'se': float(se),
            'ci_lower': float(d - z * se),
            'ci_upper': float(d + z * se),
            'n_pairs': n,
            'method': 'pre_sd' if use_pre_sd else 'diff_sd'
        }
    
    def glass_delta(self, group1: Union[str, np.ndarray], group2: Union[str, np.ndarray] = None,
                    grouping_col: str = None, value_col: str = None,
                    confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate Glass's Δ (delta) effect size
        
        Uses control group (group2) standard deviation as denominator.
        Appropriate when group variances are unequal.
        
        Glass's Δ = (M1 - M2) / SD_control
        
        Args:
            group1: Treatment group (column name or array)
            group2: Control group (column name or array) - its SD is used
            grouping_col: Alternative - column with group labels
            value_col: Alternative - column with values
            confidence: Confidence level for CI
            
        Returns:
            Dictionary with delta, CI, interpretation
        """
        # Handle different input formats
        if grouping_col and value_col and self.df is not None:
            groups = self.df[grouping_col].unique()
            if len(groups) != 2:
                return {'error': f'Expected 2 groups, found {len(groups)}'}
            g1 = self.df[self.df[grouping_col] == groups[0]][value_col].dropna().values
            g2 = self.df[self.df[grouping_col] == groups[1]][value_col].dropna().values
        elif isinstance(group1, str) and self.df is not None:
            g1 = self.df[group1].dropna().values
            g2 = self.df[group2].dropna().values if group2 else None
        else:
            g1 = np.asarray(group1)
            g2 = np.asarray(group2) if group2 is not None else None
        
        if g2 is None:
            return {'error': 'Two groups required (control group is group2)'}
        
        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        
        # Use control group (group2) SD
        sd_control = np.std(g2, ddof=1)
        
        delta = (m1 - m2) / sd_control if sd_control > 0 else 0.0
        
        # Standard error
        se = np.sqrt((n1 + n2) / (n1 * n2) + delta**2 / (2 * n2))
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = delta - z * se
        ci_upper = delta + z * se
        
        # Interpretation (same as Cohen's d)
        abs_d = abs(delta)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'glass_delta': float(delta),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'interpretation': interpretation,
            'n1': n1,
            'n2': n2,
            'mean1': float(m1),
            'mean2': float(m2),
            'control_sd': float(sd_control),
            'method': 'glass_delta'
        }
    
    # =========================================================================
    # ANOVA EFFECT SIZES
    # =========================================================================
    
    def eta_squared(self, groups: List[Union[str, np.ndarray]] = None,
                    grouping_col: str = None, value_col: str = None) -> Dict[str, Any]:
        """
        Calculate eta-squared (η²) for one-way ANOVA
        
        η² = SS_between / SS_total
        
        Interpretation:
        - η² < 0.01: negligible
        - 0.01 ≤ η² < 0.06: small
        - 0.06 ≤ η² < 0.14: medium
        - η² ≥ 0.14: large
        
        Returns:
            Dictionary with η², interpretation
        """
        # Get groups from different input formats
        if grouping_col and value_col and self.df is not None:
            group_labels = self.df[grouping_col].dropna().unique()
            groups = [self.df[self.df[grouping_col] == g][value_col].dropna().values 
                     for g in group_labels]
        elif groups is None:
            return {'error': 'Provide groups or grouping_col/value_col'}
        
        groups = [np.asarray(g) for g in groups]
        
        # Calculate SS
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = ss_total - ss_between
        
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        
        # Interpretation
        if eta_sq < 0.01:
            interpretation = "negligible"
        elif eta_sq < 0.06:
            interpretation = "small"
        elif eta_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'eta_squared': float(eta_sq),
            'ss_between': float(ss_between),
            'ss_within': float(ss_within),
            'ss_total': float(ss_total),
            'interpretation': interpretation,
            'n_groups': len(groups),
            'n_total': len(all_data)
        }
    
    def omega_squared(self, groups: List[Union[str, np.ndarray]] = None,
                      grouping_col: str = None, value_col: str = None) -> Dict[str, Any]:
        """
        Calculate omega-squared (ω²) - less biased than eta-squared
        
        ω² = (SS_between - df_between * MS_within) / (SS_total + MS_within)
        
        Returns:
            Dictionary with ω², interpretation
        """
        # Get groups
        if grouping_col and value_col and self.df is not None:
            group_labels = self.df[grouping_col].dropna().unique()
            groups = [self.df[self.df[grouping_col] == g][value_col].dropna().values 
                     for g in group_labels]
        elif groups is None:
            return {'error': 'Provide groups or grouping_col/value_col'}
        
        groups = [np.asarray(g) for g in groups]
        
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = ss_total - ss_between
        
        k = len(groups)
        n = len(all_data)
        df_between = k - 1
        df_within = n - k
        
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)
        omega_sq = max(0, omega_sq)  # Can be negative, floor at 0
        
        if omega_sq < 0.01:
            interpretation = "negligible"
        elif omega_sq < 0.06:
            interpretation = "small"
        elif omega_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'omega_squared': float(omega_sq),
            'eta_squared': float(ss_between / ss_total) if ss_total > 0 else 0.0,
            'interpretation': interpretation,
            'n_groups': k,
            'df_between': df_between,
            'df_within': df_within
        }
    
    def partial_eta_squared(self, ss_effect: float, ss_error: float) -> Dict[str, Any]:
        """
        Calculate partial eta-squared for factorial ANOVA
        
        ηp² = SS_effect / (SS_effect + SS_error)
        
        Args:
            ss_effect: Sum of squares for the effect
            ss_error: Sum of squares for error
            
        Returns:
            Dictionary with partial η²
        """
        partial_eta = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else 0.0
        
        if partial_eta < 0.01:
            interpretation = "negligible"
        elif partial_eta < 0.06:
            interpretation = "small"
        elif partial_eta < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'partial_eta_squared': float(partial_eta),
            'ss_effect': float(ss_effect),
            'ss_error': float(ss_error),
            'interpretation': interpretation
        }
    
    # =========================================================================
    # CATEGORICAL EFFECT SIZES
    # =========================================================================
    
    def cramers_v(self, var1: str, var2: str = None, 
                  contingency_table: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate Cramér's V for categorical association
        
        V = sqrt(χ² / (n * min(r-1, c-1)))
        
        Interpretation:
        - V < 0.1: negligible
        - 0.1 ≤ V < 0.3: small
        - 0.3 ≤ V < 0.5: medium
        - V ≥ 0.5: large
        
        Returns:
            Dictionary with V, χ², interpretation
        """
        if contingency_table is not None:
            table = np.asarray(contingency_table)
        elif self.df is not None and var1 and var2:
            table = pd.crosstab(self.df[var1], self.df[var2]).values
        else:
            return {'error': 'Provide contingency table or var1/var2'}
        
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        n = table.sum()
        r, c = table.shape
        
        v = np.sqrt(chi2 / (n * min(r - 1, c - 1))) if n > 0 and min(r, c) > 1 else 0.0
        
        if v < 0.1:
            interpretation = "negligible"
        elif v < 0.3:
            interpretation = "small"
        elif v < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cramers_v': float(v),
            'chi_squared': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'n': int(n),
            'interpretation': interpretation
        }
    
    def phi_coefficient(self, var1: str = None, var2: str = None,
                        contingency_table: np.ndarray = None) -> Dict[str, Any]:
        """
        Phi coefficient for 2x2 tables
        
        φ = sqrt(χ² / n)
        
        Returns:
            Dictionary with φ, interpretation
        """
        if contingency_table is not None:
            table = np.asarray(contingency_table)
        elif self.df is not None and var1 and var2:
            table = pd.crosstab(self.df[var1], self.df[var2]).values
        else:
            return {'error': 'Provide contingency table or var1/var2'}
        
        if table.shape != (2, 2):
            return {'error': 'Phi requires 2x2 table. Use Cramér\'s V for larger tables.'}
        
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        n = table.sum()
        
        phi = np.sqrt(chi2 / n) if n > 0 else 0.0
        
        # Determine sign from table structure
        a, b, c, d = table.flatten()
        phi_signed = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d)) if (a+b)*(c+d)*(a+c)*(b+d) > 0 else 0
        
        if abs(phi) < 0.1:
            interpretation = "negligible"
        elif abs(phi) < 0.3:
            interpretation = "small"
        elif abs(phi) < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'phi': float(phi_signed),
            'phi_absolute': float(phi),
            'chi_squared': float(chi2),
            'p_value': float(p_value),
            'n': int(n),
            'interpretation': interpretation
        }
    
    # =========================================================================
    # ODDS RATIOS AND RISK RATIOS
    # =========================================================================
    
    def odds_ratio(self, contingency_table: np.ndarray = None,
                   var1: str = None, var2: str = None,
                   confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate odds ratio from 2x2 contingency table
        
        OR = (a*d) / (b*c)
        
        Table format:
            | Outcome+ | Outcome- |
        Exp+|    a     |    b     |
        Exp-|    c     |    d     |
        
        Returns:
            Dictionary with OR, CI, interpretation
        """
        if contingency_table is not None:
            table = np.asarray(contingency_table)
        elif self.df is not None and var1 and var2:
            table = pd.crosstab(self.df[var1], self.df[var2]).values
        else:
            return {'error': 'Provide contingency table or var1/var2'}
        
        if table.shape != (2, 2):
            return {'error': 'Odds ratio requires 2x2 table'}
        
        a, b, c, d = table.flatten().astype(float)
        
        # Add 0.5 to cells if any are zero (Haldane-Anscombe correction)
        if 0 in [a, b, c, d]:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        
        # Log odds ratio and SE
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = np.exp(log_or - z * se_log_or)
        ci_upper = np.exp(log_or + z * se_log_or)
        
        return {
            'odds_ratio': float(odds_ratio),
            'log_odds_ratio': float(log_or),
            'se_log_or': float(se_log_or),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'table': table.tolist()
        }
    
    def risk_ratio(self, contingency_table: np.ndarray = None,
                   var1: str = None, var2: str = None,
                   confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate relative risk (risk ratio) from 2x2 table
        
        RR = (a/(a+b)) / (c/(c+d))
        
        Returns:
            Dictionary with RR, CI
        """
        if contingency_table is not None:
            table = np.asarray(contingency_table)
        elif self.df is not None and var1 and var2:
            table = pd.crosstab(self.df[var1], self.df[var2]).values
        else:
            return {'error': 'Provide contingency table or var1/var2'}
        
        if table.shape != (2, 2):
            return {'error': 'Risk ratio requires 2x2 table'}
        
        a, b, c, d = table.flatten().astype(float)
        
        risk_exposed = a / (a + b) if (a + b) > 0 else 0
        risk_unexposed = c / (c + d) if (c + d) > 0 else 0
        
        rr = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
        
        # SE of log(RR)
        log_rr = np.log(rr) if rr > 0 and rr != float('inf') else 0
        se_log_rr = np.sqrt(b/(a*(a+b)) + d/(c*(c+d))) if a > 0 and c > 0 else float('inf')
        
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = np.exp(log_rr - z * se_log_rr) if se_log_rr != float('inf') else 0
        ci_upper = np.exp(log_rr + z * se_log_rr) if se_log_rr != float('inf') else float('inf')
        
        return {
            'risk_ratio': float(rr),
            'risk_exposed': float(risk_exposed),
            'risk_unexposed': float(risk_unexposed),
            'log_rr': float(log_rr),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence': confidence,
            'absolute_risk_reduction': float(risk_unexposed - risk_exposed),
            'number_needed_to_treat': float(1 / abs(risk_unexposed - risk_exposed)) if risk_exposed != risk_unexposed else float('inf')
        }
    
    # =========================================================================
    # CORRELATION EFFECT SIZES
    # =========================================================================
    
    def r_to_d(self, r: float) -> Dict[str, float]:
        """
        Convert correlation r to Cohen's d
        
        d = 2r / sqrt(1 - r²)
        """
        if abs(r) >= 1:
            return {'error': 'r must be between -1 and 1'}
        
        d = 2 * r / np.sqrt(1 - r**2)
        
        return {
            'cohens_d': float(d),
            'r': float(r),
            'r_squared': float(r**2)
        }
    
    def d_to_r(self, d: float, n1: int = None, n2: int = None) -> Dict[str, float]:
        """
        Convert Cohen's d to correlation r
        
        r = d / sqrt(d² + a) where a depends on sample sizes
        """
        if n1 and n2:
            a = (n1 + n2)**2 / (n1 * n2)
        else:
            a = 4  # Assumes equal groups
        
        r = d / np.sqrt(d**2 + a)
        
        return {
            'r': float(r),
            'r_squared': float(r**2),
            'cohens_d': float(d)
        }
    
    def r_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence interval for correlation using Fisher's z transformation
        """
        z_r = np.arctanh(r)  # Fisher's z
        se_z = 1 / np.sqrt(n - 3)
        
        z_crit = norm.ppf((1 + confidence) / 2)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        
        return {
            'r': float(r),
            'ci_lower': float(r_lower),
            'ci_upper': float(r_upper),
            'confidence': confidence,
            'fisher_z': float(z_r),
            'se_z': float(se_z)
        }
    
    # =========================================================================
    # SUMMARY METHODS
    # =========================================================================
    
    def comprehensive_effect_sizes(self, group1: str, group2: str = None,
                                   grouping_col: str = None, value_col: str = None) -> Dict[str, Any]:
        """
        Calculate all applicable effect sizes for a comparison
        
        Returns:
            Dictionary with multiple effect size measures
        """
        results = {}
        
        # Cohen's d and Hedges' g
        results['cohens_d'] = self.cohens_d(group1, group2, grouping_col, value_col)
        results['hedges_g'] = self.hedges_g(group1, group2, grouping_col, value_col)
        
        # Glass's delta (using group1 as control)
        results['glass_delta'] = self.cohens_d(group1, group2, grouping_col, value_col, pooled=False)
        
        # Convert d to r
        if 'cohens_d' in results['cohens_d']:
            d = results['cohens_d']['cohens_d']
            results['r_equivalent'] = self.d_to_r(d)
        
        return results
    
    def interpret_effect_size(self, value: float, measure: str) -> str:
        """
        Interpret an effect size value
        
        Args:
            value: The effect size value
            measure: Type ('d', 'r', 'eta', 'omega', 'v', 'phi', 'or')
            
        Returns:
            Interpretation string
        """
        value = abs(value)
        
        if measure in ['d', 'g', 'delta']:
            if value < 0.2:
                return "negligible"
            elif value < 0.5:
                return "small"
            elif value < 0.8:
                return "medium"
            else:
                return "large"
        
        elif measure in ['r', 'phi']:
            if value < 0.1:
                return "negligible"
            elif value < 0.3:
                return "small"
            elif value < 0.5:
                return "medium"
            else:
                return "large"
        
        elif measure in ['eta', 'omega', 'eta_squared', 'omega_squared']:
            if value < 0.01:
                return "negligible"
            elif value < 0.06:
                return "small"
            elif value < 0.14:
                return "medium"
            else:
                return "large"
        
        elif measure in ['v', 'cramers_v']:
            if value < 0.1:
                return "negligible"
            elif value < 0.3:
                return "small"
            elif value < 0.5:
                return "medium"
            else:
                return "large"
        
        elif measure == 'or':
            # Odds ratio interpretation (Chen et al., 2010)
            if value < 1.5:
                return "negligible"
            elif value < 2.5:
                return "small"
            elif value < 4.0:
                return "medium"
            else:
                return "large"
        
        return "unknown measure"
