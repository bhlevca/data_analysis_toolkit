"""
Multivariate Hypothesis Testing Module for the Data Analysis Toolkit
=====================================================================

Provides PERMANOVA, ANOSIM, MANOVA, discriminant analysis (LDA/CVA),
SIMPER, and Hotelling's T² test.

Uses numpy/scipy/scikit-learn only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy import stats as scipy_stats


# ============================================================================
# PERMANOVA (Permutational Multivariate Analysis of Variance)
# ============================================================================

def permanova(distance_matrix: np.ndarray, groups: np.ndarray,
              permutations: int = 999) -> Dict[str, Any]:
    """
    PERMANOVA (Anderson, 2001) — non-parametric MANOVA using distances.

    Tests whether centroids of groups differ in multivariate space.

    Args:
        distance_matrix: Square symmetric distance matrix (n × n)
        groups: Array of group labels (length n)
        permutations: Number of permutations

    Returns:
        Dictionary with 'pseudo_F', 'p_value', 'R2', 'df_between',
        'df_within', 'SS_between', 'SS_within', 'SS_total'
    """
    D = np.asarray(distance_matrix, dtype=float)
    groups = np.asarray(groups)
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if k < 2:
        raise ValueError("Need at least 2 groups for PERMANOVA")

    # Total sum of squares (from distances)
    SS_total = np.sum(D ** 2) / (2 * n)

    # Within-group sum of squares
    SS_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 1:
            D_g = D[np.ix_(mask, mask)]
            SS_within += np.sum(D_g ** 2) / (2 * n_g)

    SS_between = SS_total - SS_within

    # Pseudo-F statistic
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or SS_within == 0:
        return {
            'pseudo_F': float('inf'),
            'p_value': 0.0,
            'R2': 1.0,
            'df_between': df_between,
            'df_within': df_within,
            'SS_between': SS_between,
            'SS_within': SS_within,
            'SS_total': SS_total,
            'n_permutations': permutations,
        }

    F_obs = (SS_between / df_between) / (SS_within / df_within)
    R2 = SS_between / SS_total

    # Permutation test
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(permutations):
        perm_groups = rng.permutation(groups)
        SS_w_perm = 0.0
        for g in unique_groups:
            mask = perm_groups == g
            n_g = mask.sum()
            if n_g > 1:
                D_g = D[np.ix_(mask, mask)]
                SS_w_perm += np.sum(D_g ** 2) / (2 * n_g)
        SS_b_perm = SS_total - SS_w_perm
        if SS_w_perm > 0:
            F_perm = (SS_b_perm / df_between) / (SS_w_perm / df_within)
        else:
            F_perm = float('inf')
        if F_perm >= F_obs:
            count += 1

    p_value = (count + 1) / (permutations + 1)

    return {
        'pseudo_F': F_obs,
        'p_value': p_value,
        'R2': R2,
        'df_between': df_between,
        'df_within': df_within,
        'SS_between': SS_between,
        'SS_within': SS_within,
        'SS_total': SS_total,
        'n_permutations': permutations,
    }


# ============================================================================
# ANOSIM (Analysis of Similarities)
# ============================================================================

def anosim(distance_matrix: np.ndarray, groups: np.ndarray,
           permutations: int = 999) -> Dict[str, Any]:
    """
    ANOSIM (Clarke, 1993) — tests whether between-group distances
    are greater than within-group distances using rank-based statistic.

    Args:
        distance_matrix: Square symmetric distance matrix
        groups: Array of group labels
        permutations: Number of permutations

    Returns:
        Dictionary with 'R_statistic', 'p_value', 'n_permutations'
    """
    D = np.asarray(distance_matrix, dtype=float)
    groups = np.asarray(groups)
    n = len(groups)

    # Rank all pairwise distances
    idx_upper = np.triu_indices(n, k=1)
    dists = D[idx_upper]
    ranks = scipy_stats.rankdata(dists)

    # Rebuild ranked distance matrix
    R = np.zeros((n, n))
    R[idx_upper] = ranks
    R = R + R.T

    def _compute_R_stat(g):
        """Compute ANOSIM R statistic for given grouping."""
        within_ranks = []
        between_ranks = []
        for i in range(n):
            for j in range(i + 1, n):
                if g[i] == g[j]:
                    within_ranks.append(R[i, j])
                else:
                    between_ranks.append(R[i, j])

        if not within_ranks or not between_ranks:
            return 0.0

        r_w = np.mean(within_ranks)
        r_b = np.mean(between_ranks)
        M = n * (n - 1) / 4  # expected mean rank
        return (r_b - r_w) / M if M > 0 else 0.0

    R_obs = _compute_R_stat(groups)

    # Permutation test
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(permutations):
        perm_groups = rng.permutation(groups)
        R_perm = _compute_R_stat(perm_groups)
        if R_perm >= R_obs:
            count += 1

    p_value = (count + 1) / (permutations + 1)

    return {
        'R_statistic': R_obs,
        'p_value': p_value,
        'n_permutations': permutations,
    }


# ============================================================================
# SIMPER (Similarity Percentages)
# ============================================================================

def simper(data: np.ndarray, groups: np.ndarray,
           species_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    SIMPER analysis (Clarke, 1993) — identifies which species contribute
    most to Bray-Curtis dissimilarity between groups.

    Args:
        data: Species matrix (samples × species)
        groups: Array of group labels
        species_names: Optional names for species columns

    Returns:
        Dictionary mapping group-pair strings to DataFrames of species contributions
    """
    data = np.asarray(data, dtype=float)
    groups = np.asarray(groups)
    n_species = data.shape[1]

    if species_names is None:
        species_names = [f"Species_{i}" for i in range(n_species)]

    unique_groups = np.unique(groups)
    results = {}

    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i + 1:]:
            mask1 = groups == g1
            mask2 = groups == g2
            data1 = data[mask1]
            data2 = data[mask2]

            # For each species, compute average contribution to Bray-Curtis
            contributions = np.zeros(n_species)
            n_pairs = 0

            for s1 in range(data1.shape[0]):
                for s2 in range(data2.shape[0]):
                    total = data1[s1].sum() + data2[s2].sum()
                    if total > 0:
                        contributions += np.abs(data1[s1] - data2[s2]) / total
                        n_pairs += 1

            if n_pairs > 0:
                contributions /= n_pairs

            total_dissim = contributions.sum()
            cumulative = np.cumsum(np.sort(contributions)[::-1])

            # Sort by contribution
            order = np.argsort(contributions)[::-1]
            sorted_names = [species_names[j] for j in order]
            sorted_contrib = contributions[order]
            sorted_pct = sorted_contrib / total_dissim * 100 if total_dissim > 0 else sorted_contrib
            cumul_pct = np.cumsum(sorted_pct)

            # Mean abundance in each group
            mean1 = data1.mean(axis=0)[order]
            mean2 = data2.mean(axis=0)[order]

            df = pd.DataFrame({
                'Species': sorted_names,
                f'Mean_{g1}': mean1,
                f'Mean_{g2}': mean2,
                'Avg_Dissimilarity': sorted_contrib,
                'Contribution_%': sorted_pct,
                'Cumulative_%': cumul_pct,
            })

            results[f"{g1}_vs_{g2}"] = df

    return results


# ============================================================================
# MANOVA (Multivariate Analysis of Variance)
# ============================================================================

def manova(data: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """
    One-way MANOVA using Wilks' Lambda with F-approximation.

    Args:
        data: Multivariate data matrix (n × p)
        groups: Group labels (length n)

    Returns:
        Dictionary with 'wilks_lambda', 'F_statistic', 'df1', 'df2',
        'p_value', 'pillai_trace', 'hotelling_lawley'
    """
    data = np.asarray(data, dtype=float)
    groups = np.asarray(groups)
    n, p = data.shape
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if k < 2:
        raise ValueError("Need at least 2 groups")

    # Grand mean
    grand_mean = data.mean(axis=0)

    # Total SS matrix
    T = np.zeros((p, p))
    for i in range(n):
        diff = data[i] - grand_mean
        T += np.outer(diff, diff)

    # Between-groups SS matrix (H)
    H = np.zeros((p, p))
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        group_mean = data[mask].mean(axis=0)
        diff = group_mean - grand_mean
        H += n_g * np.outer(diff, diff)

    # Within-groups SS matrix (E = T - H)
    E = T - H

    # Wilks' Lambda = |E| / |E + H|
    det_E = np.linalg.det(E)
    det_EH = np.linalg.det(E + H)
    wilks = det_E / det_EH if det_EH != 0 else 0

    # F-approximation for Wilks' Lambda (Rao's F)
    df_h = k - 1
    df_e = n - k
    s = min(p, df_h)

    if s == 1:
        F_stat = (1 - wilks) / wilks * df_e / p
        df1 = p
        df2 = df_e
    elif s == 2:
        wilks_sqrt = np.sqrt(wilks) if wilks > 0 else 0
        r = df_e - (p - df_h + 1) / 2
        u = (p * df_h - 2) / 4
        if p ** 2 + df_h ** 2 <= 5:
            t = 1
        else:
            t = np.sqrt((p ** 2 * df_h ** 2 - 4) / (p ** 2 + df_h ** 2 - 5))
        df1 = p * df_h
        df2 = r * t - u if t > 0 else 1
        df2 = max(df2, 1)
        if wilks > 0:
            lam_1t = wilks ** (1 / t) if t > 0 else wilks
            F_stat = ((1 - lam_1t) / lam_1t) * (df2 / df1) if lam_1t > 0 else float('inf')
        else:
            F_stat = float('inf')
    else:
        # General Rao's F approximation
        r = df_e - (p - df_h + 1) / 2
        u = (p * df_h - 2) / 4
        t_sq = (p ** 2 * df_h ** 2 - 4) / (p ** 2 + df_h ** 2 - 5)
        t = np.sqrt(max(t_sq, 0))
        df1 = p * df_h
        df2 = max(r * t - u, 1)
        if wilks > 0 and t > 0:
            lam_1t = wilks ** (1 / t)
            F_stat = ((1 - lam_1t) / lam_1t) * (df2 / df1) if lam_1t > 0 else float('inf')
        else:
            F_stat = float('inf')

    p_value = 1 - scipy_stats.f.cdf(F_stat, df1, df2) if np.isfinite(F_stat) else 0.0

    # Pillai's trace
    try:
        E_inv = np.linalg.inv(E)
        eigenvalues = np.real(np.linalg.eigvals(H @ E_inv))
        pillai = np.sum(eigenvalues / (1 + eigenvalues))
        hotelling_lawley = eigenvalues.sum()
    except np.linalg.LinAlgError:
        pillai = np.nan
        hotelling_lawley = np.nan

    return {
        'wilks_lambda': wilks,
        'F_statistic': F_stat,
        'df1': int(df1),
        'df2': int(df2),
        'p_value': p_value,
        'pillai_trace': pillai,
        'hotelling_lawley': hotelling_lawley,
    }


# ============================================================================
# HOTELLING'S T² TEST
# ============================================================================

def hotelling_t2(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """
    Two-sample Hotelling's T² test.

    Multivariate generalization of the two-sample t-test.

    Args:
        group1: Data for group 1 (n1 × p)
        group2: Data for group 2 (n2 × p)

    Returns:
        Dictionary with 'T2', 'F_statistic', 'df1', 'df2', 'p_value'
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, p = g1.shape
    n2 = g2.shape[0]

    mean_diff = g1.mean(axis=0) - g2.mean(axis=0)

    # Pooled covariance
    S1 = np.cov(g1.T, ddof=1)
    S2 = np.cov(g2.T, ddof=1)
    S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # T² = n1*n2/(n1+n2) * diff' * S_pooled^-1 * diff
    try:
        S_inv = np.linalg.inv(S_pooled)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_pooled)

    T2 = (n1 * n2) / (n1 + n2) * mean_diff @ S_inv @ mean_diff

    # Convert to F-distribution
    F_stat = T2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    df1 = p
    df2 = n1 + n2 - p - 1

    p_value = 1 - scipy_stats.f.cdf(F_stat, df1, df2) if df2 > 0 else np.nan

    return {
        'T2': T2,
        'F_statistic': F_stat,
        'df1': df1,
        'df2': df2,
        'p_value': p_value,
    }


# ============================================================================
# LINEAR DISCRIMINANT ANALYSIS (LDA / CVA)
# ============================================================================

def discriminant_analysis(data: np.ndarray, groups: np.ndarray,
                          n_components: Optional[int] = None) -> Dict[str, Any]:
    """
    Linear Discriminant Analysis (LDA) / Canonical Variate Analysis (CVA).

    Finds linear combinations that maximize between-group to within-group
    variance ratio.

    Args:
        data: Data matrix (n × p)
        groups: Group labels (length n)
        n_components: Number of discriminant axes (default: k-1)

    Returns:
        Dictionary with:
          - 'scores': Discriminant scores (n × n_components)
          - 'coefficients': Discriminant function coefficients
          - 'eigenvalues': Eigenvalues
          - 'explained_variance': Proportion per axis
          - 'group_centroids': Group means in discriminant space
          - 'classification': Predicted group labels
          - 'accuracy': Classification accuracy
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    data = np.asarray(data, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if n_components is None:
        n_components = min(k - 1, data.shape[1])

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    scores = lda.fit_transform(data, groups)

    # Group centroids in discriminant space
    centroids = {}
    for g in unique_groups:
        mask = groups == g
        centroids[str(g)] = scores[mask].mean(axis=0).tolist()

    # Classification
    predicted = lda.predict(data)
    accuracy = np.mean(predicted == groups)

    explained_ratio = lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else np.ones(n_components) / n_components

    return {
        'scores': scores,
        'coefficients': lda.scalings_[:, :n_components] if hasattr(lda, 'scalings_') else lda.coef_.T,
        'eigenvalues': np.array(explained_ratio) * n_components,
        'explained_variance': explained_ratio,
        'group_centroids': centroids,
        'classification': predicted,
        'accuracy': accuracy,
        'classes': unique_groups.tolist(),
    }


# ============================================================================
# UTILITY: RUN FROM DATAFRAME
# ============================================================================

def multivariate_test_dataframe(
    df: pd.DataFrame,
    response_cols: List[str],
    group_col: str,
    test: str = 'permanova',
    distance_metric: str = 'bray_curtis',
    permutations: int = 999
) -> Dict[str, Any]:
    """
    Convenience function to run multivariate tests on a DataFrame.

    Args:
        df: Input DataFrame
        response_cols: Response variable columns
        group_col: Grouping variable column
        test: 'permanova', 'anosim', 'manova', 'hotelling', 'lda', 'simper'
        distance_metric: For PERMANOVA/ANOSIM
        permutations: For permutation tests

    Returns:
        Test results dictionary
    """
    data = df[response_cols].values.astype(float)
    groups = df[group_col].values

    if test in ('permanova', 'anosim'):
        # Compute distance matrix
        if distance_metric == 'euclidean':
            dm = squareform(pdist(data, metric='euclidean'))
        elif distance_metric in ('bray_curtis', 'jaccard', 'sorensen'):
            from ecology import distance_matrix as eco_dm
            dm = eco_dm(data, metric=distance_metric)
        else:
            dm = squareform(pdist(data, metric=distance_metric))

        if test == 'permanova':
            return permanova(dm, groups, permutations=permutations)
        else:
            return anosim(dm, groups, permutations=permutations)

    elif test == 'manova':
        return manova(data, groups)

    elif test == 'hotelling':
        unique = np.unique(groups)
        if len(unique) != 2:
            raise ValueError("Hotelling's T² requires exactly 2 groups")
        g1 = data[groups == unique[0]]
        g2 = data[groups == unique[1]]
        return hotelling_t2(g1, g2)

    elif test == 'lda':
        return discriminant_analysis(data, groups)

    elif test == 'simper':
        return simper(data, groups, species_names=response_cols)

    else:
        raise ValueError(f"Unknown test: {test}")
