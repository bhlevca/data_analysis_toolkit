"""
Survival Analysis Module
=========================
Time-to-event analysis methods including:
- Kaplan-Meier survival curves
- Cox Proportional Hazards regression
- Log-rank test for group comparison
- Hazard ratios with confidence intervals
- Survival function estimation
- Cumulative hazard (Nelson-Aalen)

Version: 1.0

Requirements:
    pip install lifelines
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Optional imports
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
    from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import median_survival_times
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


class SurvivalAnalysis:
    """Survival analysis methods for time-to-event data"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.kmf = None
        self.cph = None
        self.naf = None
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame"""
        self.df = df
    
    def _check_lifelines(self):
        """Check if lifelines is available"""
        if not LIFELINES_AVAILABLE:
            return {'error': 'lifelines not installed. Install with: pip install lifelines'}
        return None
    
    # =========================================================================
    # KAPLAN-MEIER ANALYSIS
    # =========================================================================
    
    def kaplan_meier(self, duration_col: str, event_col: str,
                     group_col: str = None, confidence: float = 0.95,
                     timeline: np.ndarray = None) -> Dict[str, Any]:
        """
        Kaplan-Meier survival analysis
        
        Args:
            duration_col: Column with time-to-event or censoring time
            event_col: Column with event indicator (1=event, 0=censored)
            group_col: Optional column for stratification
            confidence: Confidence level for intervals
            timeline: Specific time points to evaluate
            
        Returns:
            Dictionary with survival estimates and statistics
        """
        error = self._check_lifelines()
        if error:
            return error
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        T = self.df[duration_col].dropna()
        E = self.df[event_col].loc[T.index]
        
        alpha = 1 - confidence
        
        if group_col is None:
            # Single group analysis
            self.kmf = KaplanMeierFitter()
            self.kmf.fit(T, E, alpha=alpha, timeline=timeline)
            
            # Handle median CI - attribute name varies by lifelines version
            try:
                if hasattr(self.kmf, 'confidence_interval_median_survival_time_'):
                    median_ci = self.kmf.confidence_interval_median_survival_time_.values.flatten().tolist()
                else:
                    # Fallback for older versions
                    median_ci = [None, None]
            except:
                median_ci = [None, None]
            
            results = {
                'median_survival': float(self.kmf.median_survival_time_) if not np.isinf(self.kmf.median_survival_time_) else None,
                'median_ci': median_ci,
                'survival_function': self.kmf.survival_function_.to_dict(),
                'confidence_interval_lower': self.kmf.confidence_interval_survival_function_.iloc[:, 0].to_dict(),
                'confidence_interval_upper': self.kmf.confidence_interval_survival_function_.iloc[:, 1].to_dict(),
                'timeline': self.kmf.timeline.tolist(),
                'n_observations': int(len(T)),
                'n_events': int(E.sum()),
                'n_censored': int(len(E) - E.sum())
            }
            
            # Survival at specific time points
            survival_at = {}
            for t in [30, 90, 180, 365, 730]:  # Common time points
                if t <= T.max():
                    try:
                        survival_at[t] = float(self.kmf.predict(t))
                    except:
                        pass
            results['survival_at'] = survival_at
            
        else:
            # Grouped analysis
            groups = self.df[group_col].loc[T.index]
            unique_groups = groups.unique()
            
            results = {
                'groups': {},
                'n_groups': len(unique_groups)
            }
            
            for group in unique_groups:
                mask = groups == group
                T_group = T[mask]
                E_group = E[mask]
                
                kmf = KaplanMeierFitter()
                kmf.fit(T_group, E_group, alpha=alpha, label=str(group), timeline=timeline)
                
                results['groups'][str(group)] = {
                    'median_survival': float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None,
                    'survival_function': kmf.survival_function_.iloc[:, 0].to_dict(),
                    'n_observations': int(len(T_group)),
                    'n_events': int(E_group.sum())
                }
            
            # Log-rank test
            if len(unique_groups) == 2:
                logrank = self.log_rank_test(duration_col, event_col, group_col)
                results['log_rank_test'] = logrank
            elif len(unique_groups) > 2:
                logrank = self.multivariate_log_rank_test(duration_col, event_col, group_col)
                results['log_rank_test'] = logrank
        
        return results
    
    def log_rank_test(self, duration_col: str, event_col: str,
                      group_col: str) -> Dict[str, Any]:
        """
        Log-rank test for comparing survival between two groups
        
        Returns:
            Dictionary with test statistics and p-value
        """
        error = self._check_lifelines()
        if error:
            return error
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        T = self.df[duration_col].dropna()
        E = self.df[event_col].loc[T.index]
        groups = self.df[group_col].loc[T.index]
        
        unique_groups = groups.unique()
        if len(unique_groups) != 2:
            return {'error': 'Log-rank test requires exactly 2 groups'}
        
        mask = groups == unique_groups[0]
        
        result = logrank_test(T[mask], T[~mask], E[mask], E[~mask])
        
        return {
            'test_statistic': float(result.test_statistic),
            'p_value': float(result.p_value),
            'is_significant': result.p_value < 0.05,
            'group_1': str(unique_groups[0]),
            'group_2': str(unique_groups[1])
        }
    
    def multivariate_log_rank_test(self, duration_col: str, event_col: str,
                                   group_col: str) -> Dict[str, Any]:
        """
        Log-rank test for comparing survival across multiple groups
        """
        error = self._check_lifelines()
        if error:
            return error
        
        T = self.df[duration_col].dropna()
        E = self.df[event_col].loc[T.index]
        groups = self.df[group_col].loc[T.index]
        
        result = multivariate_logrank_test(T, groups, E)
        
        return {
            'test_statistic': float(result.test_statistic),
            'p_value': float(result.p_value),
            'is_significant': result.p_value < 0.05,
            'n_groups': len(groups.unique())
        }
    
    def plot_kaplan_meier(self, duration_col: str, event_col: str,
                          group_col: str = None, confidence: float = 0.95,
                          at_risk: bool = True, title: str = None) -> plt.Figure:
        """
        Plot Kaplan-Meier survival curves
        
        Args:
            duration_col: Time column
            event_col: Event indicator column
            group_col: Optional grouping column
            confidence: Confidence level
            at_risk: Show at-risk table
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        error = self._check_lifelines()
        if error:
            return None
        
        T = self.df[duration_col].dropna()
        E = self.df[event_col].loc[T.index]
        alpha = 1 - confidence
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if group_col is None:
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, alpha=alpha, label='Survival')
            kmf.plot_survival_function(ax=ax, ci_show=True)
        else:
            groups = self.df[group_col].loc[T.index]
            for group in groups.unique():
                mask = groups == group
                kmf = KaplanMeierFitter()
                kmf.fit(T[mask], E[mask], alpha=alpha, label=str(group))
                kmf.plot_survival_function(ax=ax, ci_show=True)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title(title or 'Kaplan-Meier Survival Curve')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # COX PROPORTIONAL HAZARDS
    # =========================================================================
    
    def cox_regression(self, duration_col: str, event_col: str,
                       covariates: List[str], strata: str = None,
                       penalizer: float = 0.0, l1_ratio: float = 0.0) -> Dict[str, Any]:
        """
        Cox Proportional Hazards regression
        
        Args:
            duration_col: Time-to-event column
            event_col: Event indicator column
            covariates: List of covariate columns
            strata: Optional stratification column
            penalizer: L1/L2 regularization strength
            l1_ratio: Elastic net mixing (0=L2, 1=L1)
            
        Returns:
            Dictionary with regression results and hazard ratios
        """
        error = self._check_lifelines()
        if error:
            return error
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        # Prepare data
        cols = [duration_col, event_col] + covariates
        if strata:
            cols.append(strata)
        
        df_cox = self.df[cols].dropna()
        
        # Fit Cox model
        self.cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.cph.fit(df_cox, duration_col=duration_col, event_col=event_col,
                     strata=strata)
        
        # Extract results
        summary = self.cph.summary
        
        hazard_ratios = {}
        for covariate in covariates:
            if covariate in summary.index:
                hr = summary.loc[covariate, 'exp(coef)']
                hr_lower = summary.loc[covariate, 'exp(coef) lower 95%']
                hr_upper = summary.loc[covariate, 'exp(coef) upper 95%']
                p_value = summary.loc[covariate, 'p']
                
                hazard_ratios[covariate] = {
                    'hazard_ratio': float(hr),
                    'ci_lower': float(hr_lower),
                    'ci_upper': float(hr_upper),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # Model statistics
        results = {
            'hazard_ratios': hazard_ratios,
            'coefficients': self.cph.params_.to_dict(),
            'standard_errors': self.cph.standard_errors_.to_dict(),
            'log_likelihood': float(self.cph.log_likelihood_),
            'concordance_index': float(self.cph.concordance_index_),
            'aic_partial': float(self.cph.AIC_partial_),
            'n_observations': int(len(df_cox)),
            'n_events': int(df_cox[event_col].sum()),
            'summary_table': summary.to_dict()
        }
        
        # Proportional hazards test
        try:
            ph_test = self.cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=0.05)
            results['proportional_hazards_test'] = 'passed' if ph_test is None else 'check required'
        except:
            results['proportional_hazards_test'] = 'not computed'
        
        return results
    
    def hazard_ratio_forest_plot(self, cox_results: Dict[str, Any]) -> plt.Figure:
        """
        Create forest plot of hazard ratios
        
        Args:
            cox_results: Results from cox_regression()
            
        Returns:
            Matplotlib figure
        """
        hr_data = cox_results['hazard_ratios']
        
        fig, ax = plt.subplots(figsize=(10, len(hr_data) * 0.6 + 2))
        
        y_positions = range(len(hr_data))
        covariates = list(hr_data.keys())
        
        for i, cov in enumerate(covariates):
            data = hr_data[cov]
            hr = data['hazard_ratio']
            ci_lower = data['ci_lower']
            ci_upper = data['ci_upper']
            
            color = 'red' if data['significant'] else 'blue'
            
            # Plot point and CI
            ax.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=2)
            ax.scatter([hr], [i], color=color, s=100, zorder=5)
            
            # Add text
            ax.text(ci_upper + 0.1, i, f'HR={hr:.2f} [{ci_lower:.2f}-{ci_upper:.2f}]',
                   va='center', fontsize=9)
        
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(covariates)
        ax.set_xlabel('Hazard Ratio (95% CI)')
        ax.set_title('Forest Plot of Hazard Ratios')
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_cox_coefficients(self, cox_results: Dict[str, Any] = None) -> plt.Figure:
        """Plot Cox regression coefficients"""
        if self.cph is None and cox_results is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.cph is not None:
            self.cph.plot(ax=ax)
        else:
            coefs = cox_results['coefficients']
            se = cox_results['standard_errors']
            
            y_pos = range(len(coefs))
            ax.barh(y_pos, list(coefs.values()), xerr=[1.96 * s for s in se.values()])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(coefs.keys()))
            ax.axvline(x=0, color='black', linestyle='--')
            ax.set_xlabel('Coefficient (log hazard ratio)')
        
        ax.set_title('Cox Regression Coefficients')
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # NELSON-AALEN CUMULATIVE HAZARD
    # =========================================================================
    
    def cumulative_hazard(self, duration_col: str, event_col: str,
                          group_col: str = None) -> Dict[str, Any]:
        """
        Nelson-Aalen cumulative hazard estimation
        
        Returns:
            Dictionary with cumulative hazard estimates
        """
        error = self._check_lifelines()
        if error:
            return error
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        T = self.df[duration_col].dropna()
        E = self.df[event_col].loc[T.index]
        
        if group_col is None:
            self.naf = NelsonAalenFitter()
            self.naf.fit(T, E)
            
            return {
                'cumulative_hazard': self.naf.cumulative_hazard_.iloc[:, 0].to_dict(),
                'confidence_interval_lower': self.naf.confidence_interval_.iloc[:, 0].to_dict(),
                'confidence_interval_upper': self.naf.confidence_interval_.iloc[:, 1].to_dict(),
                'timeline': self.naf.timeline.tolist()
            }
        else:
            groups = self.df[group_col].loc[T.index]
            results = {'groups': {}}
            
            for group in groups.unique():
                mask = groups == group
                naf = NelsonAalenFitter()
                naf.fit(T[mask], E[mask], label=str(group))
                
                results['groups'][str(group)] = {
                    'cumulative_hazard': naf.cumulative_hazard_.iloc[:, 0].to_dict(),
                    'timeline': naf.timeline.tolist()
                }
            
            return results
    
    # =========================================================================
    # PARAMETRIC MODELS
    # =========================================================================
    
    def parametric_survival(self, duration_col: str, event_col: str,
                            covariates: List[str], distribution: str = 'weibull') -> Dict[str, Any]:
        """
        Parametric survival models (AFT models)
        
        Args:
            duration_col: Time column
            event_col: Event column
            covariates: Covariate columns
            distribution: 'weibull', 'lognormal', or 'loglogistic'
            
        Returns:
            Dictionary with parametric model results
        """
        error = self._check_lifelines()
        if error:
            return error
        
        if self.df is None:
            return {'error': 'No data loaded'}
        
        cols = [duration_col, event_col] + covariates
        df_model = self.df[cols].dropna()
        
        # Select model
        if distribution == 'weibull':
            model = WeibullAFTFitter()
        elif distribution == 'lognormal':
            model = LogNormalAFTFitter()
        elif distribution == 'loglogistic':
            model = LogLogisticAFTFitter()
        else:
            return {'error': f'Unknown distribution: {distribution}'}
        
        model.fit(df_model, duration_col=duration_col, event_col=event_col)
        
        summary = model.summary
        
        return {
            'distribution': distribution,
            'coefficients': model.params_.to_dict(),
            'summary': summary.to_dict(),
            'aic': float(model.AIC_),
            'log_likelihood': float(model.log_likelihood_),
            'concordance_index': float(model.concordance_index_)
        }
    
    # =========================================================================
    # SURVIVAL PREDICTIONS
    # =========================================================================
    
    def predict_survival(self, duration_col: str, event_col: str,
                         covariates: List[str], new_data: pd.DataFrame,
                         times: List[float] = None) -> Dict[str, Any]:
        """
        Predict survival probabilities for new observations
        
        Args:
            duration_col, event_col: Time and event columns
            covariates: Covariate columns
            new_data: DataFrame with covariate values
            times: Time points for prediction
            
        Returns:
            Dictionary with survival predictions
        """
        if self.cph is None:
            # Fit model first
            cox_result = self.cox_regression(duration_col, event_col, covariates)
            if 'error' in cox_result:
                return cox_result
        
        if times is None:
            times = [30, 90, 180, 365, 730]
        
        survival_probs = self.cph.predict_survival_function(new_data)
        
        predictions = {}
        for i, row in new_data.iterrows():
            predictions[i] = {}
            for t in times:
                if t in survival_probs.index:
                    predictions[i][t] = float(survival_probs.loc[t, i])
                else:
                    # Interpolate
                    closest_idx = (survival_probs.index - t).abs().argmin()
                    predictions[i][t] = float(survival_probs.iloc[closest_idx, i])
        
        return {
            'survival_predictions': predictions,
            'times': times,
            'full_survival_function': survival_probs.to_dict()
        }
