"""
Community Ecology Module for the Data Analysis Toolkit
======================================================

Provides diversity indices, beta diversity, rarefaction, and species
accumulation curves for community ecology data (species × sites matrices).

All methods use only numpy/scipy — no external ecology packages required.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats
from scipy.special import comb


# ============================================================================
# ALPHA DIVERSITY INDICES
# ============================================================================

def shannon_index(counts: np.ndarray) -> float:
    """
    Shannon diversity index H' = -Σ(pi * ln(pi))

    Args:
        counts: Array of species abundances (integers or floats > 0)

    Returns:
        Shannon index value
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    N = counts.sum()
    p = counts / N
    return -np.sum(p * np.log(p))


def simpson_index(counts: np.ndarray) -> float:
    """
    Simpson's diversity index (1 - D) where D = Σ(pi²).

    Args:
        counts: Array of species abundances

    Returns:
        Simpson's diversity (1 - D), range [0, 1]
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    N = counts.sum()
    p = counts / N
    return 1.0 - np.sum(p ** 2)


def simpson_dominance(counts: np.ndarray) -> float:
    """
    Simpson's dominance D = Σ(ni*(ni-1)) / (N*(N-1))

    Uses the finite-sample (unbiased) formula.
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = counts.sum()
    if N <= 1:
        return 0.0
    return np.sum(counts * (counts - 1)) / (N * (N - 1))


def inverse_simpson(counts: np.ndarray) -> float:
    """Inverse Simpson index 1/D."""
    d = simpson_dominance(counts)
    return 1.0 / d if d > 0 else float('inf')


def fisher_alpha(counts: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> float:
    """
    Fisher's alpha diversity index.

    Solves S = a * ln(1 + N/a) iteratively.

    Args:
        counts: Array of species abundances
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Fisher's alpha
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    S = len(counts)
    N = counts.sum()

    if S == 0 or N == 0:
        return 0.0

    # Newton-Raphson: solve S = a * ln(1 + N/a)
    # f(a) = a * ln(1 + N/a) - S
    # f'(a) = ln(1 + N/a) - N/(a + N)
    alpha = S / np.log(N) if N > 1 else S  # initial guess

    for _ in range(max_iter):
        ratio = N / alpha
        f_a = alpha * np.log(1 + ratio) - S
        f_prime = np.log(1 + ratio) - N / (alpha + N)
        if abs(f_prime) < 1e-15:
            break
        alpha_new = alpha - f_a / f_prime
        if alpha_new <= 0:
            alpha_new = alpha / 2
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


def margalef_index(counts: np.ndarray) -> float:
    """Margalef's richness index: (S - 1) / ln(N)"""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    S = len(counts)
    N = counts.sum()
    if N <= 1:
        return 0.0
    return (S - 1) / np.log(N)


def menhinick_index(counts: np.ndarray) -> float:
    """Menhinick's richness index: S / sqrt(N)"""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    S = len(counts)
    N = counts.sum()
    if N == 0:
        return 0.0
    return S / np.sqrt(N)


def berger_parker_index(counts: np.ndarray) -> float:
    """Berger-Parker dominance index: Nmax / N"""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    return counts.max() / counts.sum()


def evenness_pielou(counts: np.ndarray) -> float:
    """Pielou's evenness: J = H' / ln(S)"""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    S = len(counts)
    if S <= 1:
        return 1.0
    H = shannon_index(counts)
    return H / np.log(S)


def chao1(counts: np.ndarray) -> float:
    """
    Chao1 richness estimator.

    Chao1 = S_obs + f1*(f1-1) / (2*(f2+1))

    where f1 = number of singletons, f2 = number of doubletons.
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    S_obs = len(counts)
    f1 = np.sum(counts == 1)  # singletons
    f2 = np.sum(counts == 2)  # doubletons

    if f2 == 0:
        # Bias-corrected form
        return S_obs + f1 * (f1 - 1) / 2 if f1 > 0 else float(S_obs)
    return S_obs + (f1 ** 2) / (2 * f2)


def ace_estimator(counts: np.ndarray, threshold: int = 10) -> float:
    """
    ACE (Abundance-based Coverage Estimator).

    Args:
        counts: Array of species abundances
        threshold: Abundance threshold for 'rare' species (default: 10)

    Returns:
        ACE richness estimate
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]

    S_rare = np.sum(counts <= threshold)
    S_abund = np.sum(counts > threshold)
    N_rare = np.sum(counts[counts <= threshold])

    if N_rare == 0 or S_rare == 0:
        return float(len(counts))

    # Sample coverage
    f1 = np.sum(counts == 1)
    C_ace = 1 - f1 / N_rare

    if C_ace <= 0:
        return float(len(counts))

    # Coefficient of variation
    fi = np.array([np.sum(counts == i) for i in range(1, threshold + 1)])
    i_vals = np.arange(1, threshold + 1)
    numerator = np.sum(i_vals * (i_vals - 1) * fi)
    gamma_sq = max(0, (S_rare / C_ace) * numerator / (N_rare * (N_rare - 1)) - 1)

    return S_abund + S_rare / C_ace + f1 * gamma_sq / C_ace


def all_alpha_diversity(counts: np.ndarray) -> Dict[str, float]:
    """
    Calculate all alpha diversity indices for a single sample.

    Args:
        counts: Array of species abundances

    Returns:
        Dictionary of index_name: value
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]

    return {
        'Species Richness (S)': len(counts),
        'Total Abundance (N)': counts.sum(),
        'Shannon (H\')': shannon_index(counts),
        'Simpson (1-D)': simpson_index(counts),
        'Inverse Simpson (1/D)': inverse_simpson(counts),
        'Fisher\'s Alpha': fisher_alpha(counts),
        'Margalef': margalef_index(counts),
        'Menhinick': menhinick_index(counts),
        'Berger-Parker': berger_parker_index(counts),
        'Pielou Evenness (J)': evenness_pielou(counts),
        'Chao1': chao1(counts),
        'ACE': ace_estimator(counts),
    }


# ============================================================================
# BETA DIVERSITY (DISSIMILARITY MATRICES)
# ============================================================================

def bray_curtis(x: np.ndarray, y: np.ndarray) -> float:
    """Bray-Curtis dissimilarity between two samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    total = np.abs(x).sum() + np.abs(y).sum()
    if total == 0:
        return 0.0
    return np.sum(np.abs(x - y)) / total


def jaccard_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Jaccard distance (1 - Jaccard similarity) based on presence/absence."""
    x_pa = (np.asarray(x) > 0)
    y_pa = (np.asarray(y) > 0)
    shared = np.sum(x_pa & y_pa)
    total = np.sum(x_pa | y_pa)
    if total == 0:
        return 0.0
    return 1.0 - shared / total


def sorensen_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Sørensen distance (presence/absence based)."""
    x_pa = (np.asarray(x) > 0)
    y_pa = (np.asarray(y) > 0)
    shared = np.sum(x_pa & y_pa)
    total_species = np.sum(x_pa) + np.sum(y_pa)
    if total_species == 0:
        return 0.0
    return 1.0 - (2 * shared) / total_species


def morisita_horn(x: np.ndarray, y: np.ndarray) -> float:
    """Morisita-Horn dissimilarity index."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    N_x, N_y = x.sum(), y.sum()
    if N_x == 0 or N_y == 0:
        return 1.0
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    denom = (sum_x2 / (N_x ** 2) + sum_y2 / (N_y ** 2)) * N_x * N_y
    if denom == 0:
        return 1.0
    return 1.0 - 2 * sum_xy / denom


_DISTANCE_FUNCS = {
    'bray_curtis': bray_curtis,
    'jaccard': jaccard_distance,
    'sorensen': sorensen_distance,
    'morisita_horn': morisita_horn,
}


def distance_matrix(data: np.ndarray, metric: str = 'bray_curtis') -> np.ndarray:
    """
    Compute pairwise dissimilarity matrix.

    Args:
        data: 2D array (samples × species)
        metric: 'bray_curtis', 'jaccard', 'sorensen', 'morisita_horn'

    Returns:
        n×n distance matrix
    """
    func = _DISTANCE_FUNCS.get(metric, bray_curtis)
    n = data.shape[0]
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = func(data[i], data[j])
            dm[i, j] = d
            dm[j, i] = d
    return dm


def whittaker_beta(data: np.ndarray) -> float:
    """
    Whittaker's beta diversity: β_W = (gamma / mean_alpha) - 1

    Args:
        data: 2D array (samples × species)

    Returns:
        Whittaker beta diversity
    """
    data = np.asarray(data, dtype=float)
    # Alpha: species richness per sample
    alphas = np.sum(data > 0, axis=1)
    mean_alpha = alphas.mean()
    # Gamma: total species richness
    gamma = np.sum(np.any(data > 0, axis=0))
    if mean_alpha == 0:
        return 0.0
    return gamma / mean_alpha - 1


# ============================================================================
# RAREFACTION
# ============================================================================

def rarefaction_curve(counts: np.ndarray, steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Individual-based rarefaction curve using the exact formula.

    E[S_n] = S - Σ C(N-Ni, n) / C(N, n)

    Args:
        counts: Array of species abundances
        steps: Number of points along the curve

    Returns:
        Tuple of (sample_sizes, expected_species)
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = int(counts.sum())
    S = len(counts)

    if N == 0 or S == 0:
        return np.array([0]), np.array([0])

    sample_sizes = np.unique(np.linspace(1, N, steps).astype(int))
    expected = np.zeros(len(sample_sizes))

    for idx, n in enumerate(sample_sizes):
        if n >= N:
            expected[idx] = S
        else:
            # E[S_n] = S - Σ C(N-Ni, n) / C(N, n)
            # Use log-combinations for numerical stability
            log_denom = _log_comb(N, n)
            sum_term = 0.0
            for ni in counts:
                ni = int(ni)
                if N - ni >= n:
                    sum_term += np.exp(_log_comb(N - ni, n) - log_denom)
                # else: C(N-ni, n) = 0 when N-ni < n
            expected[idx] = S - sum_term

    return sample_sizes, expected


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient using gammaln for numerical stability."""
    from scipy.special import gammaln
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rarefied_richness(counts: np.ndarray, n: int) -> float:
    """
    Expected species richness for a rarefied sample of size n.

    Args:
        counts: Array of species abundances
        n: Sample size to rarefy to

    Returns:
        Expected number of species
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = int(counts.sum())
    S = len(counts)

    if n >= N:
        return float(S)
    if n <= 0:
        return 0.0

    log_denom = _log_comb(N, n)
    sum_term = 0.0
    for ni in counts:
        ni = int(ni)
        if N - ni >= n:
            sum_term += np.exp(_log_comb(N - ni, n) - log_denom)
    return S - sum_term


# ============================================================================
# SPECIES ACCUMULATION CURVE
# ============================================================================

def species_accumulation(data: np.ndarray, permutations: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample-based species accumulation curve with confidence intervals.

    Randomly permutes the sample order and tracks cumulative richness.

    Args:
        data: 2D array (samples × species)
        permutations: Number of random orderings

    Returns:
        Tuple of (n_samples, mean_richness, std_richness)
    """
    data = np.asarray(data, dtype=float)
    n_samples, n_species = data.shape
    presence = (data > 0)

    all_curves = np.zeros((permutations, n_samples))

    rng = np.random.default_rng(42)
    for p in range(permutations):
        order = rng.permutation(n_samples)
        cumulative = np.zeros(n_species, dtype=bool)
        for i, idx in enumerate(order):
            cumulative = cumulative | presence[idx]
            all_curves[p, i] = cumulative.sum()

    x = np.arange(1, n_samples + 1)
    mean_richness = all_curves.mean(axis=0)
    std_richness = all_curves.std(axis=0)

    return x, mean_richness, std_richness


# ============================================================================
# SHE ANALYSIS
# ============================================================================

def she_analysis(counts: np.ndarray, steps: int = 50) -> Dict[str, np.ndarray]:
    """
    SHE analysis: plots ln(S), H', and ln(E) against ln(N) on a rarefaction basis.

    Used to distinguish community assembly patterns.

    Args:
        counts: Array of species abundances
        steps: Number of rarefaction steps

    Returns:
        Dictionary with 'ln_N', 'ln_S', 'H', 'ln_E' arrays
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = int(counts.sum())

    if N == 0:
        return {'ln_N': np.array([]), 'ln_S': np.array([]),
                'H': np.array([]), 'ln_E': np.array([])}

    sample_sizes = np.unique(np.linspace(10, N, steps).astype(int))

    ln_N_arr = []
    ln_S_arr = []
    H_arr = []
    ln_E_arr = []

    for n in sample_sizes:
        S_n = rarefied_richness(counts, n)
        if S_n <= 0:
            continue

        # Approximate H at this sample size via subsampling
        # Use proportional abundance approximation
        p = counts / N
        H_n = -np.sum(p[p > 0] * np.log(p[p > 0]))

        E_n = np.exp(H_n) / S_n if S_n > 0 else 1.0

        ln_N_arr.append(np.log(n))
        ln_S_arr.append(np.log(S_n))
        H_arr.append(H_n)
        ln_E_arr.append(np.log(E_n) if E_n > 0 else 0)

    return {
        'ln_N': np.array(ln_N_arr),
        'ln_S': np.array(ln_S_arr),
        'H': np.array(H_arr),
        'ln_E': np.array(ln_E_arr),
    }


# ============================================================================
# UTILITY: ANALYZE COMMUNITY DATA FRAME
# ============================================================================

def analyze_community_dataframe(
    df: pd.DataFrame,
    species_cols: List[str],
    sample_col: Optional[str] = None,
    metric: str = 'bray_curtis'
) -> Dict[str, Any]:
    """
    Full community analysis on a DataFrame.

    Args:
        df: DataFrame with species abundance columns
        species_cols: Column names for species abundances
        sample_col: Optional column identifying samples (rows used if None)
        metric: Beta diversity metric

    Returns:
        Dictionary with:
          - 'alpha': DataFrame of alpha diversity per sample
          - 'beta_matrix': Distance matrix as DataFrame
          - 'whittaker_beta': Whittaker beta value
          - 'rarefaction': Rarefaction curves per sample
          - 'accumulation': Species accumulation curve data
    """
    data = df[species_cols].values.astype(float)

    if sample_col and sample_col in df.columns:
        labels = df[sample_col].astype(str).tolist()
    else:
        labels = [f"Sample_{i}" for i in range(data.shape[0])]

    # Alpha diversity for each sample
    alpha_rows = []
    for i in range(data.shape[0]):
        row_div = all_alpha_diversity(data[i])
        row_div['Sample'] = labels[i]
        alpha_rows.append(row_div)
    alpha_df = pd.DataFrame(alpha_rows)
    cols = ['Sample'] + [c for c in alpha_df.columns if c != 'Sample']
    alpha_df = alpha_df[cols]

    # Beta diversity matrix
    dm = distance_matrix(data, metric=metric)
    beta_df = pd.DataFrame(dm, index=labels, columns=labels)

    # Whittaker beta
    wb = whittaker_beta(data)

    # Rarefaction for each sample
    rarefaction = {}
    for i in range(data.shape[0]):
        sizes, expected = rarefaction_curve(data[i])
        rarefaction[labels[i]] = (sizes, expected)

    # Species accumulation
    acc_x, acc_mean, acc_std = species_accumulation(data)

    return {
        'alpha': alpha_df,
        'beta_matrix': beta_df,
        'whittaker_beta': wb,
        'rarefaction': rarefaction,
        'accumulation': {
            'n_samples': acc_x,
            'mean_richness': acc_mean,
            'std_richness': acc_std,
        },
    }
