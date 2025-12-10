//! High-performance Rust extensions for the Data Analysis Toolkit
//! 
//! This module provides optimized implementations of computationally
//! intensive statistical and machine learning operations.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;

// ============================================================================
// DISTANCE CORRELATION
// ============================================================================

/// Compute distance matrix for a 1D array
fn distance_matrix_1d(x: &[f64]) -> Array2<f64> {
    let n = x.len();
    let mut dist = Array2::zeros((n, n));
    
    for i in 0..n {
        for j in i..n {
            let d = (x[i] - x[j]).abs();
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    dist
}

/// Double-center a distance matrix
fn double_center(dist: &Array2<f64>) -> Array2<f64> {
    let n = dist.nrows();
    let row_means = dist.mean_axis(Axis(1)).unwrap();
    let col_means = dist.mean_axis(Axis(0)).unwrap();
    let grand_mean = dist.mean().unwrap();
    
    let mut centered = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            centered[[i, j]] = dist[[i, j]] - row_means[i] - col_means[j] + grand_mean;
        }
    }
    centered
}

/// Calculate distance correlation between two arrays
/// 
/// Distance correlation can detect non-linear relationships, unlike Pearson correlation.
/// Returns a value between 0 and 1, where 0 indicates independence.
#[pyfunction]
fn distance_correlation(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    
    if x.len() != y.len() {
        return Err(PyValueError::new_err("Arrays must have the same length"));
    }
    
    let n = x.len();
    if n < 2 {
        return Ok(0.0);
    }
    
    // Compute distance matrices
    let a = distance_matrix_1d(x);
    let b = distance_matrix_1d(y);
    
    // Double-center the matrices
    let a_centered = double_center(&a);
    let b_centered = double_center(&b);
    
    // Compute distance covariance and variances
    let n_sq = (n * n) as f64;
    let dcov_xy: f64 = (&a_centered * &b_centered).sum() / n_sq;
    let dcov_xx: f64 = (&a_centered * &a_centered).sum() / n_sq;
    let dcov_yy: f64 = (&b_centered * &b_centered).sum() / n_sq;
    
    // Distance correlation
    let denom = (dcov_xx.sqrt() * dcov_yy.sqrt()).sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }
    
    Ok(dcov_xy.max(0.0).sqrt() / denom)
}

/// Calculate distance correlation for multiple features against a target
#[pyfunction]
fn distance_correlation_matrix(
    py: Python<'_>,
    features: PyReadonlyArray2<f64>,
    target: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let features = features.as_array();
    let target = target.as_slice()?;
    
    let n_features = features.ncols();
    
    // Parallel computation across features
    let correlations: Vec<f64> = (0..n_features)
        .into_par_iter()
        .map(|i| {
            let feature: Vec<f64> = features.column(i).to_vec();
            let a = distance_matrix_1d(&feature);
            let b = distance_matrix_1d(target);
            let a_centered = double_center(&a);
            let b_centered = double_center(&b);
            
            let n = feature.len();
            let n_sq = (n * n) as f64;
            let dcov_xy: f64 = (&a_centered * &b_centered).sum() / n_sq;
            let dcov_xx: f64 = (&a_centered * &a_centered).sum() / n_sq;
            let dcov_yy: f64 = (&b_centered * &b_centered).sum() / n_sq;
            
            let denom = (dcov_xx.sqrt() * dcov_yy.sqrt()).sqrt();
            if denom == 0.0 { 0.0 } else { dcov_xy.max(0.0).sqrt() / denom }
        })
        .collect();
    
    Ok(Array1::from_vec(correlations).into_pyarray(py).to_owned())
}

// ============================================================================
// BOOTSTRAP CONFIDENCE INTERVALS
// ============================================================================

/// Perform bootstrap resampling for linear regression coefficients
/// 
/// Returns (mean_coefficients, ci_lower, ci_upper)
#[pyfunction]
fn bootstrap_linear_regression(
    py: Python<'_>,
    x_data: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    n_bootstrap: usize,
    confidence: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let x_arr = x_data.as_array();
    let y = y.as_array();
    
    let n_samples = x_arr.nrows();
    let n_features = x_arr.ncols();
    
    // Parallel bootstrap iterations
    let bootstrap_coefs: Vec<Vec<f64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            
            // Resample with replacement
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();
            
            // Build resampled matrices
            let mut x_boot = Array2::zeros((n_samples, n_features));
            let mut y_boot = Array1::zeros(n_samples);
            
            for (i, &idx) in indices.iter().enumerate() {
                for j in 0..n_features {
                    x_boot[[i, j]] = x_arr[[idx, j]];
                }
                y_boot[i] = y[idx];
            }
            
            // Simple OLS: coefficients = (X'X)^(-1) X'y
            // Using normal equations (simplified for demonstration)
            solve_ols(&x_boot, &y_boot)
        })
        .collect();
    
    // Compute statistics
    let alpha = (1.0 - confidence) / 2.0;
    let lower_percentile = (alpha * n_bootstrap as f64) as usize;
    let upper_percentile = ((1.0 - alpha) * n_bootstrap as f64) as usize;
    
    let mut mean_coefs = vec![0.0; n_features];
    let mut ci_lower = vec![0.0; n_features];
    let mut ci_upper = vec![0.0; n_features];
    
    for j in 0..n_features {
        let mut coef_values: Vec<f64> = bootstrap_coefs.iter().map(|c| c[j]).collect();
        coef_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        mean_coefs[j] = coef_values.iter().sum::<f64>() / n_bootstrap as f64;
        ci_lower[j] = coef_values[lower_percentile.min(n_bootstrap - 1)];
        ci_upper[j] = coef_values[upper_percentile.min(n_bootstrap - 1)];
    }
    
    Ok((
        Array1::from_vec(mean_coefs).into_pyarray(py).to_owned(),
        Array1::from_vec(ci_lower).into_pyarray(py).to_owned(),
        Array1::from_vec(ci_upper).into_pyarray(py).to_owned(),
    ))
}

/// Simple OLS solver using normal equations
fn solve_ols(x_matrix: &Array2<f64>, y: &Array1<f64>) -> Vec<f64> {
    let n_features = x_matrix.ncols();
    
    // X'X
    let mut xt_x = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            xt_x[[i, j]] = x_matrix.column(i).dot(&x_matrix.column(j));
        }
    }
    
    // X'y
    let mut xt_y = Array1::zeros(n_features);
    for i in 0..n_features {
        xt_y[i] = x_matrix.column(i).dot(y);
    }
    
    // Solve using simple Gaussian elimination (for small problems)
    // In production, use a proper linear algebra library
    solve_linear_system(&xt_x, &xt_y)
}

fn solve_linear_system(coef_matrix: &Array2<f64>, rhs: &Array1<f64>) -> Vec<f64> {
    let n = coef_matrix.nrows();
    let mut aug = Array2::zeros((n, n + 1));
    
    // Build augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = coef_matrix[[i, j]];
        }
        aug[[i, n]] = rhs[i];
    }
    
    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                max_idx = i;
            }
        }
        
        // Swap rows
        if max_idx != k {
            for j in 0..=n {
                let tmp = aug[[k, j]];
                aug[[k, j]] = aug[[max_idx, j]];
                aug[[max_idx, j]] = tmp;
            }
        }
        
        // Eliminate
        if aug[[k, k]].abs() < 1e-10 {
            continue;
        }
        
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[[i, i]].abs() < 1e-10 {
            x[i] = 0.0;
            continue;
        }
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }
    
    x
}

// ============================================================================
// MONTE CARLO SIMULATION
// ============================================================================

/// Monte Carlo simulation for prediction uncertainty
#[pyfunction]
fn monte_carlo_predictions(
    py: Python<'_>,
    x_data: PyReadonlyArray2<f64>,
    coefficients: PyReadonlyArray1<f64>,
    intercept: f64,
    residual_std: f64,
    n_simulations: usize,
    confidence: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let x_arr = x_data.as_array();
    let coef = coefficients.as_array();
    
    let n_samples = x_arr.nrows();
    let n_features = x_arr.ncols();
    
    // Run parallel simulations
    let predictions: Vec<Vec<f64>> = (0..n_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            
            // Add noise to coefficients
            let noisy_coef: Vec<f64> = coef.iter()
                .map(|&c| c + rng.sample::<f64, _>(StandardNormal) * residual_std / 10.0)
                .collect();
            let noisy_intercept = intercept + rng.sample::<f64, _>(StandardNormal) * residual_std;
            
            // Predict with noise
            let mut preds = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut pred = noisy_intercept;
                for j in 0..n_features {
                    pred += x_arr[[i, j]] * noisy_coef[j];
                }
                pred += rng.sample::<f64, _>(StandardNormal) * residual_std;
                preds.push(pred);
            }
            preds
        })
        .collect();
    
    // Compute statistics
    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * n_simulations as f64) as usize;
    let upper_idx = ((1.0 - alpha) * n_simulations as f64) as usize;
    
    let mut mean_pred = vec![0.0; n_samples];
    let mut ci_lower = vec![0.0; n_samples];
    let mut ci_upper = vec![0.0; n_samples];
    
    for i in 0..n_samples {
        let mut sample_preds: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
        sample_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        mean_pred[i] = sample_preds.iter().sum::<f64>() / n_simulations as f64;
        ci_lower[i] = sample_preds[lower_idx.min(n_simulations - 1)];
        ci_upper[i] = sample_preds[upper_idx.min(n_simulations - 1)];
    }
    
    Ok((
        Array1::from_vec(mean_pred).into_pyarray(py).to_owned(),
        Array1::from_vec(ci_lower).into_pyarray(py).to_owned(),
        Array1::from_vec(ci_upper).into_pyarray(py).to_owned(),
    ))
}

// ============================================================================
// TRANSFER ENTROPY
// ============================================================================

/// Calculate transfer entropy between two time series
/// 
/// Transfer entropy measures directed information flow.
#[pyfunction]
fn transfer_entropy(
    _py: Python<'_>,
    source: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<f64>,
    n_bins: usize,
    lag: usize,
) -> PyResult<f64> {
    let source = source.as_slice()?;
    let target = target.as_slice()?;
    
    if source.len() != target.len() {
        return Err(PyValueError::new_err("Arrays must have the same length"));
    }
    
    let n = source.len();
    if n <= lag {
        return Ok(0.0);
    }
    
    // Bin the data
    let source_binned = bin_data(source, n_bins);
    let target_binned = bin_data(target, n_bins);
    
    // Count joint and marginal probabilities
    let mut p_y_ypast_xpast: HashMap<(usize, usize, usize), f64> = HashMap::new();
    let mut p_y_ypast: HashMap<(usize, usize), f64> = HashMap::new();
    let mut p_ypast_xpast: HashMap<(usize, usize), f64> = HashMap::new();
    let mut p_ypast: HashMap<usize, f64> = HashMap::new();
    
    let count = (n - lag) as f64;
    
    for i in lag..n {
        let y = target_binned[i];
        let y_past = target_binned[i - lag];
        let x_past = source_binned[i - lag];
        
        *p_y_ypast_xpast.entry((y, y_past, x_past)).or_insert(0.0) += 1.0 / count;
        *p_y_ypast.entry((y, y_past)).or_insert(0.0) += 1.0 / count;
        *p_ypast_xpast.entry((y_past, x_past)).or_insert(0.0) += 1.0 / count;
        *p_ypast.entry(y_past).or_insert(0.0) += 1.0 / count;
    }
    
    // Calculate transfer entropy
    let mut te = 0.0;
    for ((y, y_past, x_past), p_joint) in &p_y_ypast_xpast {
        let p_yy = p_y_ypast.get(&(*y, *y_past)).unwrap_or(&1e-10);
        let p_yx = p_ypast_xpast.get(&(*y_past, *x_past)).unwrap_or(&1e-10);
        let p_y_only = p_ypast.get(y_past).unwrap_or(&1e-10);
        
        if *p_joint > 0.0 && *p_yy > 0.0 && *p_yx > 0.0 && *p_y_only > 0.0 {
            let ratio = (*p_joint * *p_y_only) / (*p_yy * *p_yx);
            if ratio > 0.0 {
                te += p_joint * ratio.ln();
            }
        }
    }
    
    Ok(te.max(0.0))
}

/// Bin continuous data into discrete bins
fn bin_data(data: &[f64], n_bins: usize) -> Vec<usize> {
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    
    if range == 0.0 {
        return vec![0; data.len()];
    }
    
    let bin_width = range / n_bins as f64;
    
    data.iter()
        .map(|&x| {
            let bin = ((x - min_val) / bin_width) as usize;
            bin.min(n_bins - 1)
        })
        .collect()
}

// ============================================================================
// LEAD-LAG CORRELATION
// ============================================================================

/// Calculate correlations at multiple lags in parallel
#[pyfunction]
fn lead_lag_correlations(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    max_lag: i32,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f64>>)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    
    let n = x.len().min(y.len());
    
    let lags: Vec<i32> = (-max_lag..=max_lag).collect();
    
    let correlations: Vec<f64> = lags
        .par_iter()
        .map(|&lag| {
            if lag == 0 {
                pearson_correlation(x, y)
            } else if lag > 0 {
                let lag_usize = lag as usize;
                if lag_usize >= n {
                    return 0.0;
                }
                let x_slice = &x[..n - lag_usize];
                let y_slice = &y[lag_usize..n];
                pearson_correlation(x_slice, y_slice)
            } else {
                let lag_usize = (-lag) as usize;
                if lag_usize >= n {
                    return 0.0;
                }
                let x_slice = &x[lag_usize..n];
                let y_slice = &y[..n - lag_usize];
                pearson_correlation(x_slice, y_slice)
            }
        })
        .collect();
    
    Ok((
        Array1::from_vec(lags).into_pyarray(py).to_owned(),
        Array1::from_vec(correlations).into_pyarray(py).to_owned(),
    ))
}

/// Calculate Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    
    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;
    
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

// ============================================================================
// OUTLIER DETECTION (IQR METHOD)
// ============================================================================

/// Detect outliers using IQR method (parallelized for multiple columns)
#[pyfunction]
fn detect_outliers_iqr(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    multiplier: f64,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f64>>)> {
    let data = data.as_array();
    let n_cols = data.ncols();
    
    let results: Vec<(usize, f64)> = (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let col: Vec<f64> = data.column(j).iter()
                .filter(|x| !x.is_nan())
                .cloned()
                .collect();
            
            if col.len() < 4 {
                return (0, 0.0);
            }
            
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let q1_idx = sorted.len() / 4;
            let q3_idx = (3 * sorted.len()) / 4;
            let q1 = sorted[q1_idx];
            let q3 = sorted[q3_idx];
            let iqr = q3 - q1;
            
            let lower = q1 - multiplier * iqr;
            let upper = q3 + multiplier * iqr;
            
            let n_outliers = col.iter()
                .filter(|&&x| x < lower || x > upper)
                .count();
            
            let pct = (n_outliers as f64 / col.len() as f64) * 100.0;
            
            (n_outliers, pct)
        })
        .collect();
    
    let counts: Vec<usize> = results.iter().map(|&(c, _)| c).collect();
    let percentages: Vec<f64> = results.iter().map(|&(_, p)| p).collect();
    
    Ok((
        Array1::from_vec(counts).into_pyarray(py).to_owned(),
        Array1::from_vec(percentages).into_pyarray(py).to_owned(),
    ))
}

// ============================================================================
// MUTUAL INFORMATION (Binned Estimation)
// ============================================================================

/// Estimate mutual information between two arrays
#[pyfunction]
fn mutual_information(
    _py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<f64> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    
    let n = x.len().min(y.len());
    if n < 2 {
        return Ok(0.0);
    }
    
    let x_binned = bin_data(&x[..n], n_bins);
    let y_binned = bin_data(&y[..n], n_bins);
    
    // Joint and marginal counts
    let mut p_xy: HashMap<(usize, usize), f64> = HashMap::new();
    let mut p_x: HashMap<usize, f64> = HashMap::new();
    let mut p_y: HashMap<usize, f64> = HashMap::new();
    
    let count = n as f64;
    
    for i in 0..n {
        *p_xy.entry((x_binned[i], y_binned[i])).or_insert(0.0) += 1.0 / count;
        *p_x.entry(x_binned[i]).or_insert(0.0) += 1.0 / count;
        *p_y.entry(y_binned[i]).or_insert(0.0) += 1.0 / count;
    }
    
    // Calculate MI
    let mut mi = 0.0;
    for ((xi, yi), p_joint) in &p_xy {
        let px = p_x.get(xi).unwrap_or(&1e-10);
        let py = p_y.get(yi).unwrap_or(&1e-10);
        
        if *p_joint > 0.0 && *px > 0.0 && *py > 0.0 {
            mi += p_joint * (p_joint / (px * py)).ln();
        }
    }
    
    Ok(mi.max(0.0))
}

// ============================================================================
// ROLLING STATISTICS (Optimized)
// ============================================================================

/// Calculate rolling mean and std efficiently
#[pyfunction]
fn rolling_statistics(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let n = data.len();
    
    if window > n || window == 0 {
        return Err(PyValueError::new_err("Invalid window size"));
    }
    
    let mut means = vec![f64::NAN; n];
    let mut stds = vec![f64::NAN; n];
    
    // Initial window
    let mut sum: f64 = data[..window].iter().sum();
    let mut sum_sq: f64 = data[..window].iter().map(|x| x * x).sum();
    
    means[window - 1] = sum / window as f64;
    let variance = (sum_sq / window as f64) - (means[window - 1] * means[window - 1]);
    stds[window - 1] = variance.max(0.0).sqrt();
    
    // Slide the window
    for i in window..n {
        sum += data[i] - data[i - window];
        sum_sq += data[i] * data[i] - data[i - window] * data[i - window];
        
        means[i] = sum / window as f64;
        let variance = (sum_sq / window as f64) - (means[i] * means[i]);
        stds[i] = variance.max(0.0).sqrt();
    }
    
    Ok((
        Array1::from_vec(means).into_pyarray(py).to_owned(),
        Array1::from_vec(stds).into_pyarray(py).to_owned(),
    ))
}

// ============================================================================
// PYTHON MODULE
// ============================================================================

/// A Python module implemented in Rust for high-performance data analysis.
#[pymodule]
fn data_toolkit_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(distance_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_predictions, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(lead_lag_correlations, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers_iqr, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_statistics, m)?)?;
    
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}
