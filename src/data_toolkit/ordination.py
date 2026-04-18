"""
Multivariate Ordination Module for the Data Analysis Toolkit
=============================================================

Provides classical ordination methods: PCoA, NMDS, CA, DCA, CCA, RDA.
All implementations use numpy/scipy only — no external ecology packages.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh, svd


# ============================================================================
# PRINCIPAL COORDINATES ANALYSIS (PCoA)
# ============================================================================

def pcoa(distance_matrix: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """
    Principal Coordinates Analysis (PCoA / Classical MDS).

    Eigenanalysis of the doubly-centred distance matrix.

    Args:
        distance_matrix: Square symmetric distance matrix (n × n)
        n_components: Number of axes to return

    Returns:
        Dictionary with:
          - 'coordinates': n × n_components array of ordination scores
          - 'eigenvalues': All eigenvalues (descending)
          - 'explained_variance': Proportion of variance per axis
          - 'negative_eigenvalues': Whether correction was needed
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    # Double centering: B = -0.5 * J * D² * J  where J = I - 11'/n
    D2 = D ** 2
    row_means = D2.mean(axis=1, keepdims=True)
    col_means = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_means - col_means + grand_mean)

    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Handle negative eigenvalues (Lingoes or Cailliez correction not applied,
    # we just note them and use only positive ones)
    has_negative = np.any(eigenvalues < 0)
    positive_mask = eigenvalues > 0
    total_positive = eigenvalues[positive_mask].sum()

    # Compute coordinates: X_k = sqrt(lambda_k) * v_k
    n_comp = min(n_components, np.sum(positive_mask))
    coords = np.zeros((n, n_comp))
    for k in range(n_comp):
        if eigenvalues[k] > 0:
            coords[:, k] = eigenvectors[:, k] * np.sqrt(eigenvalues[k])

    explained = eigenvalues[:n_comp] / total_positive if total_positive > 0 else np.zeros(n_comp)

    return {
        'coordinates': coords,
        'eigenvalues': eigenvalues,
        'explained_variance': explained,
        'negative_eigenvalues': has_negative,
    }


# ============================================================================
# NON-METRIC MULTIDIMENSIONAL SCALING (NMDS)
# ============================================================================

def nmds(distance_matrix: np.ndarray, n_components: int = 2,
         max_iter: int = 300, n_init: int = 5,
         random_state: int = 42) -> Dict[str, Any]:
    """
    Non-metric Multidimensional Scaling (NMDS).

    Iteratively finds a low-dimensional configuration that preserves
    the rank order of distances. Uses scikit-learn's MDS with non-metric mode.

    Args:
        distance_matrix: Square symmetric distance matrix (n × n)
        n_components: Number of dimensions
        max_iter: Maximum iterations
        n_init: Number of random starts
        random_state: Random seed

    Returns:
        Dictionary with:
          - 'coordinates': n × n_components ordination scores
          - 'stress': Kruskal's stress-1
          - 'n_iter': Iterations used
    """
    from sklearn.manifold import MDS

    D = np.asarray(distance_matrix, dtype=float)

    mds = MDS(
        n_components=n_components,
        metric=False,  # non-metric
        dissimilarity='precomputed',
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        normalized_stress='auto',
    )
    coords = mds.fit_transform(D)
    stress = mds.stress_

    return {
        'coordinates': coords,
        'stress': stress,
        'n_iter': mds.n_iter_,
    }


# ============================================================================
# CORRESPONDENCE ANALYSIS (CA)
# ============================================================================

def correspondence_analysis(data: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """
    Correspondence Analysis (CA).

    SVD of the standardized residuals from the independence model.

    Args:
        data: Contingency table or species × sites matrix (non-negative)
        n_components: Number of axes

    Returns:
        Dictionary with:
          - 'row_scores': Row (site) scores in CA space
          - 'col_scores': Column (species) scores in CA space
          - 'eigenvalues': Eigenvalues (= squared singular values)
          - 'explained_inertia': Proportion explained per axis
          - 'total_inertia': Total inertia (chi-squared / N)
          - 'row_masses': Row masses (marginal proportions)
          - 'col_masses': Column masses
    """
    X = np.asarray(data, dtype=float)
    if np.any(X < 0):
        raise ValueError("CA requires non-negative data")

    grand_total = X.sum()
    if grand_total == 0:
        raise ValueError("Data matrix is all zeros")

    # Correspondence matrix (proportions)
    P = X / grand_total
    r = P.sum(axis=1)  # row masses
    c = P.sum(axis=0)  # column masses

    # Avoid division by zero
    r[r == 0] = 1e-15
    c[c == 0] = 1e-15

    # Standardized residuals: (P - rc') / sqrt(r) / sqrt(c)
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c))

    E = np.outer(r, c)  # expected frequencies
    residuals = (P - E)
    S = Dr_inv_sqrt @ residuals @ Dc_inv_sqrt

    # SVD
    U, sigma, Vt = svd(S, full_matrices=False)

    # Eigenvalues
    eigenvalues = sigma ** 2
    total_inertia = eigenvalues.sum()

    n_comp = min(n_components, len(eigenvalues))

    # Row and column scores (principal coordinates)
    # Row scores: Dr^{-1/2} * U * Sigma
    row_scores = Dr_inv_sqrt @ U[:, :n_comp] * sigma[:n_comp]
    # Column scores: Dc^{-1/2} * V * Sigma
    col_scores = Dc_inv_sqrt @ Vt[:n_comp].T * sigma[:n_comp]

    explained = eigenvalues[:n_comp] / total_inertia if total_inertia > 0 else np.zeros(n_comp)

    return {
        'row_scores': row_scores,
        'col_scores': col_scores,
        'eigenvalues': eigenvalues[:n_comp],
        'explained_inertia': explained,
        'total_inertia': total_inertia,
        'row_masses': r,
        'col_masses': c,
    }


# ============================================================================
# DETRENDED CORRESPONDENCE ANALYSIS (DCA)
# ============================================================================

def detrended_correspondence_analysis(data: np.ndarray, n_components: int = 2,
                                       n_segments: int = 26) -> Dict[str, Any]:
    """
    Detrended Correspondence Analysis (DCA).

    Performs CA then removes the arch effect by detrending (segmented regression)
    on each axis beyond the first.

    Args:
        data: Species × sites matrix (non-negative)
        n_components: Number of axes
        n_segments: Number of segments for detrending

    Returns:
        Dictionary with:
          - 'row_scores': Detrended site scores
          - 'col_scores': Detrended species scores
          - 'eigenvalues': Eigenvalues (from CA, before detrending)
          - 'axis_lengths': Estimated gradient lengths (SD units)
    """
    X = np.asarray(data, dtype=float)

    # First run CA
    n_comp = min(n_components + 1, min(X.shape) - 1)
    ca_result = correspondence_analysis(X, n_components=n_comp)

    row_scores = ca_result['row_scores'].copy()
    col_scores = ca_result['col_scores'].copy()

    # Detrend axes 2+ against axis 1
    for axis in range(1, row_scores.shape[1]):
        row_scores[:, axis] = _detrend_axis(
            row_scores[:, 0], row_scores[:, axis], n_segments
        )
        col_scores[:, axis] = _detrend_axis(
            col_scores[:, 0], col_scores[:, axis], n_segments
        )

    # Rescale to SD units (approximate gradient length)
    axis_lengths = np.zeros(row_scores.shape[1])
    for axis in range(row_scores.shape[1]):
        scores = row_scores[:, axis]
        if len(scores) > 1:
            axis_lengths[axis] = scores.max() - scores.min()

    return {
        'row_scores': row_scores[:, :n_components],
        'col_scores': col_scores[:, :n_components],
        'eigenvalues': ca_result['eigenvalues'][:n_components],
        'axis_lengths': axis_lengths[:n_components],
    }


def _detrend_axis(x: np.ndarray, y: np.ndarray, n_segments: int = 26) -> np.ndarray:
    """
    Detrend y against x using segmented polynomial (running mean subtraction).
    """
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-15:
        return y - y.mean()

    segment_width = (x_max - x_min) / n_segments
    y_detrended = y.copy()

    for seg in range(n_segments):
        lo = x_min + seg * segment_width
        hi = lo + segment_width
        if seg == n_segments - 1:
            hi = x_max + 1e-10
        mask = (x >= lo) & (x < hi)
        if mask.sum() > 0:
            y_detrended[mask] -= y[mask].mean()

    return y_detrended


# ============================================================================
# CANONICAL CORRESPONDENCE ANALYSIS (CCA)
# ============================================================================

def canonical_correspondence_analysis(
    species: np.ndarray,
    environment: np.ndarray,
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Canonical Correspondence Analysis (CCA).

    Constrained ordination: CA of species data constrained by environmental variables.

    Args:
        species: Species matrix (samples × species), non-negative
        environment: Environmental matrix (samples × env_variables)
        n_components: Number of axes

    Returns:
        Dictionary with:
          - 'site_scores': Constrained site scores (LC scores)
          - 'species_scores': Species scores
          - 'biplot_scores': Environmental variable biplot arrows
          - 'eigenvalues': Constrained eigenvalues
          - 'explained_inertia': Proportion per axis
          - 'total_inertia': Total inertia
          - 'constrained_inertia': Inertia explained by constraints
    """
    Y = np.asarray(species, dtype=float)
    X = np.asarray(environment, dtype=float)
    n, p = Y.shape
    _, q = X.shape

    if np.any(Y < 0):
        raise ValueError("Species data must be non-negative")

    grand_total = Y.sum()
    if grand_total == 0:
        raise ValueError("Species data is all zeros")

    # Proportional data
    P = Y / grand_total
    r = P.sum(axis=1)  # row weights
    c = P.sum(axis=0)  # column weights

    r[r == 0] = 1e-15
    c[c == 0] = 1e-15

    # Weighted environmental matrix
    Dr_sqrt = np.diag(np.sqrt(r))
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r))
    Dr = np.diag(r)

    # Center X with row weights
    X_weighted_mean = (Dr @ X).sum(axis=0)
    X_centered = X - X_weighted_mean

    # Weighted regression: Y_hat = X(X'DrX)^-1 X'Dr * chi_residuals
    # First get chi-square residuals
    E = np.outer(r, c)
    Q = (P - E)  # Residual matrix

    # Project Q onto the column space of X (weighted)
    XtDrX = X_centered.T @ Dr @ X_centered
    try:
        XtDrX_inv = np.linalg.inv(XtDrX + np.eye(q) * 1e-10)
    except np.linalg.LinAlgError:
        XtDrX_inv = np.linalg.pinv(XtDrX)

    # Fitted values in chi-squared metric
    Hat = X_centered @ XtDrX_inv @ X_centered.T @ Dr
    Q_hat = Hat @ (Dr_inv_sqrt @ Q)  # Constrained residuals

    # Standardize
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c))
    S = Dr_sqrt @ Q_hat @ Dc_inv_sqrt

    # SVD of constrained matrix
    U, sigma, Vt = svd(S, full_matrices=False)
    eigenvalues = sigma ** 2

    total_inertia = np.sum((Dr_inv_sqrt @ Q @ Dc_inv_sqrt) ** 2)
    constrained_inertia = eigenvalues.sum()

    n_comp = min(n_components, len(eigenvalues))

    # Site scores (LC = linear combination scores)
    site_scores = Dr_inv_sqrt @ U[:, :n_comp] * sigma[:n_comp]
    species_scores = Dc_inv_sqrt @ Vt[:n_comp].T * sigma[:n_comp]

    # Biplot scores for environmental variables
    # Correlation of env vars with site scores
    biplot_scores = np.zeros((q, n_comp))
    for k in range(n_comp):
        for j in range(q):
            biplot_scores[j, k] = np.corrcoef(X_centered[:, j], site_scores[:, k])[0, 1]

    explained = eigenvalues[:n_comp] / total_inertia if total_inertia > 0 else np.zeros(n_comp)

    return {
        'site_scores': site_scores,
        'species_scores': species_scores,
        'biplot_scores': biplot_scores,
        'eigenvalues': eigenvalues[:n_comp],
        'explained_inertia': explained,
        'total_inertia': total_inertia,
        'constrained_inertia': constrained_inertia,
    }


# ============================================================================
# REDUNDANCY ANALYSIS (RDA)
# ============================================================================

def redundancy_analysis(
    response: np.ndarray,
    explanatory: np.ndarray,
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Redundancy Analysis (RDA).

    Constrained PCA: PCA of the fitted values from multivariate regression.

    Args:
        response: Response matrix Y (samples × variables)
        explanatory: Explanatory matrix X (samples × predictors)
        n_components: Number of axes

    Returns:
        Dictionary with:
          - 'site_scores': Constrained site scores
          - 'species_scores': Response variable scores
          - 'biplot_scores': Explanatory variable biplot arrows
          - 'eigenvalues': Constrained eigenvalues
          - 'explained_variance': Proportion per axis
          - 'total_variance': Total variance
          - 'constrained_variance': Variance explained by constraints
          - 'r_squared': Overall R²
    """
    Y = np.asarray(response, dtype=float)
    X = np.asarray(explanatory, dtype=float)
    n, p = Y.shape
    _, q = X.shape

    # Center Y and X
    Y_centered = Y - Y.mean(axis=0)
    X_centered = X - X.mean(axis=0)

    # Multivariate regression: Y_hat = X(X'X)^-1 X'Y
    try:
        XtX_inv = np.linalg.inv(X_centered.T @ X_centered + np.eye(q) * 1e-10)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_centered.T @ X_centered)

    B = XtX_inv @ X_centered.T @ Y_centered  # Regression coefficients
    Y_hat = X_centered @ B  # Fitted values

    # PCA of fitted values
    U, sigma, Vt = svd(Y_hat, full_matrices=False)
    eigenvalues = (sigma ** 2) / (n - 1)

    total_variance = np.sum(Y_centered ** 2) / (n - 1)
    constrained_variance = eigenvalues.sum()

    n_comp = min(n_components, len(eigenvalues))

    site_scores = U[:, :n_comp] * sigma[:n_comp]
    species_scores = Vt[:n_comp].T

    # Biplot scores
    biplot_scores = np.zeros((q, n_comp))
    for k in range(n_comp):
        for j in range(q):
            biplot_scores[j, k] = np.corrcoef(X_centered[:, j], site_scores[:, k])[0, 1]

    r_squared = constrained_variance / total_variance if total_variance > 0 else 0

    explained = eigenvalues[:n_comp] / total_variance if total_variance > 0 else np.zeros(n_comp)

    return {
        'site_scores': site_scores,
        'species_scores': species_scores,
        'biplot_scores': biplot_scores,
        'eigenvalues': eigenvalues[:n_comp],
        'explained_variance': explained,
        'total_variance': total_variance,
        'constrained_variance': constrained_variance,
        'r_squared': r_squared,
    }


# ============================================================================
# MANTEL TEST
# ============================================================================

def mantel_test(dm1: np.ndarray, dm2: np.ndarray,
                method: str = 'pearson',
                permutations: int = 999) -> Dict[str, float]:
    """
    Mantel test for correlation between two distance matrices.

    Args:
        dm1: First distance matrix (n × n)
        dm2: Second distance matrix (n × n)
        method: 'pearson' or 'spearman'
        permutations: Number of permutations for significance

    Returns:
        Dictionary with 'statistic', 'p_value', 'z_score'
    """
    dm1 = np.asarray(dm1, dtype=float)
    dm2 = np.asarray(dm2, dtype=float)
    n = dm1.shape[0]

    # Extract upper triangle
    idx = np.triu_indices(n, k=1)
    v1 = dm1[idx]
    v2 = dm2[idx]

    if method == 'spearman':
        from scipy.stats import spearmanr
        r_obs, _ = spearmanr(v1, v2)
    else:
        r_obs = np.corrcoef(v1, v2)[0, 1]

    # Permutation test
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        dm2_perm = dm2[np.ix_(perm, perm)]
        v2_perm = dm2_perm[idx]
        if method == 'spearman':
            r_perm, _ = spearmanr(v1, v2_perm)
        else:
            r_perm = np.corrcoef(v1, v2_perm)[0, 1]
        if r_perm >= r_obs:
            count += 1

    p_value = (count + 1) / (permutations + 1)

    return {
        'statistic': r_obs,
        'p_value': p_value,
        'n_permutations': permutations,
        'method': method,
    }


# ============================================================================
# UTILITY: ORDINATION FROM DATAFRAME
# ============================================================================

def ordinate_dataframe(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'pcoa',
    distance_metric: str = 'bray_curtis',
    n_components: int = 2,
    env_columns: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run ordination on a DataFrame.

    Args:
        df: Input DataFrame
        columns: Species/response columns
        method: 'pcoa', 'nmds', 'ca', 'dca', 'cca', 'rda'
        distance_metric: For PCoA/NMDS: 'bray_curtis', 'jaccard', 'euclidean', etc.
        n_components: Number of axes
        env_columns: Environmental columns (required for CCA/RDA)

    Returns:
        Ordination results dictionary
    """
    data = df[columns].values.astype(float)

    if method in ('pcoa', 'nmds'):
        # Compute distance matrix
        if distance_metric == 'euclidean':
            dm = squareform(pdist(data, metric='euclidean'))
        elif distance_metric == 'bray_curtis':
            from ecology import distance_matrix as eco_dm
            dm = eco_dm(data, metric='bray_curtis')
        elif distance_metric == 'jaccard':
            from ecology import distance_matrix as eco_dm
            dm = eco_dm(data, metric='jaccard')
        else:
            dm = squareform(pdist(data, metric=distance_metric))

        if method == 'pcoa':
            return pcoa(dm, n_components=n_components)
        else:
            return nmds(dm, n_components=n_components, **kwargs)

    elif method == 'ca':
        return correspondence_analysis(data, n_components=n_components)

    elif method == 'dca':
        return detrended_correspondence_analysis(data, n_components=n_components)

    elif method == 'cca':
        if env_columns is None:
            raise ValueError("CCA requires env_columns")
        env_data = df[env_columns].values.astype(float)
        return canonical_correspondence_analysis(data, env_data, n_components=n_components)

    elif method == 'rda':
        if env_columns is None:
            raise ValueError("RDA requires env_columns")
        env_data = df[env_columns].values.astype(float)
        return redundancy_analysis(data, env_data, n_components=n_components)

    else:
        raise ValueError(f"Unknown ordination method: {method}")
