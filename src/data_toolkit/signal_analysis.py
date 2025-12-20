"""Signal analysis helpers: Fourier and Wavelet routines.

These functions are designed to be called from `TimeSeriesAnalysis` as thin
wrappers so the public API remains stable while grouping signal-specific code
in a single module for easier maintenance and future additions.
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import chi2

warnings.filterwarnings('ignore')

# Try to import pywt for CWT and DWT
try:
    import pywt
    PYWT_AVAILABLE = True
except Exception:
    PYWT_AVAILABLE = False


def fourier_transform(df, column: str, sampling_rate: float = 1.0) -> Dict[str, Any]:
    data = df[column].dropna().values
    n = len(data)

    fft_values = fft(data)
    frequencies = fftfreq(n, 1 / sampling_rate)

    magnitude = np.abs(fft_values[:n // 2])
    positive_freq = frequencies[:n // 2]
    power = magnitude ** 2

    # Find dominant frequencies excluding DC component (0 Hz) which is just the mean
    # Start from index 1 to skip DC component
    power_no_dc = power[1:]
    top_indices_no_dc = np.argsort(power_no_dc)[-5:][::-1]
    top_indices = top_indices_no_dc + 1  # Adjust indices back to full array

    results = {
        'fft': fft_values,
        'frequencies': frequencies,
        'magnitude': magnitude,
        'power': power,
        'positive_frequencies': positive_freq,
        'dominant_frequencies': [float(positive_freq[i]) for i in top_indices],
        'dominant_powers': [float(power[i]) for i in top_indices],
        'dominant_frequency': float(positive_freq[top_indices[0]]) if len(top_indices) > 0 else 0.0,
        'peak_power': float(power[top_indices[0]]) if len(top_indices) > 0 else 0.0,
        'sampling_rate': sampling_rate,
        'dc_component': float(magnitude[0])  # Store DC for reference
    }

    return results


def power_spectral_density(df, column: str, sampling_rate: float = 1.0,
                           window: str = 'hamming', nperseg: int = None) -> Dict[str, Any]:
    data = df[column].dropna().values
    if nperseg is None:
        nperseg = len(data) // 4

    frequencies, psd = welch(data, fs=sampling_rate, window=window, nperseg=nperseg)
    top_indices = np.argsort(psd)[-5:][::-1]

    results = {
        'frequencies': frequencies,
        'power_spectral_density': psd,
        'dominant_frequencies': [float(frequencies[i]) for i in top_indices],
        'dominant_powers': [float(psd[i]) for i in top_indices],
        'dominant_frequency': float(frequencies[top_indices[0]]) if len(top_indices) > 0 else 0.0,
        'total_power': float(np.sum(psd)),
        'sampling_rate': sampling_rate,
        'window_type': window
    }

    return results


def plot_fft(results: Dict[str, Any], column: str, max_freq: float = None) -> plt.Figure:
    positive_freq = results.get('positive_frequencies')
    magnitude = results.get('magnitude')

    if positive_freq is None:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(positive_freq, magnitude, linewidth=0.5)
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'FFT Magnitude Spectrum - {column}')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(positive_freq, magnitude, linewidth=0.5)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude (log scale)')
    ax2.set_title(f'FFT Magnitude Spectrum (Log Scale) - {column}')
    ax2.grid(True, alpha=0.3)

    if max_freq is not None:
        ax1.set_xlim([0, max_freq])
        ax2.set_xlim([0, max_freq])

    plt.tight_layout()
    return fig


def plot_power_spectral_density(results: Dict[str, Any], column: str) -> plt.Figure:
    frequencies = results.get('frequencies')
    psd = results.get('power_spectral_density')

    if frequencies is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(frequencies, psd, linewidth=1)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Spectral Density (Welch) - {column}')
    ax.grid(True, alpha=0.3, which='both')

    dominant_freqs = results.get('dominant_frequencies', [])
    for freq in dominant_freqs[:3]:
        ax.axvline(freq, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def continuous_wavelet_transform(df, column: str, scales: np.ndarray = None,
                                 wavelet: str = 'morl', sampling_rate: float = 1.0) -> Dict[str, Any]:
    if not PYWT_AVAILABLE:
        return {'error': 'PyWavelets not installed. Install with: pip install PyWavelets'}

    data = df[column].dropna().values
    n = len(data)
    if sampling_rate <= 0:
        return {'error': 'sampling_rate must be > 0'}
    dt = 1.0 / float(sampling_rate)

    if scales is None:
        scales = np.arange(1, min(128, len(data) // 2))

    try:
        coefficients, frequencies = pywt.cwt(data, scales, wavelet)
        omega0 = 6.0
        fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))
        periods = scales * fourier_factor * dt

        half = n // 2
        if n % 2 == 0:
            left = np.arange(0, half)
            right = np.arange(half, 0, -1)
        else:
            left = np.arange(0, half + 1)
            right = np.arange(half, 0, -1)

        coi_time = dt * np.concatenate((left, right))
        if coi_time.shape[0] != n:
            coi_time = dt * np.linspace(0, (n - 1) * 0.5, n)

        coi_period = coi_time * fourier_factor

        results = {
            'coefficients': coefficients,
            'scales': scales,
            'frequencies': frequencies,
            'periods': periods,
            'time': np.arange(len(data)) * dt,
            'wavelet': wavelet,
            'power': np.abs(coefficients) ** 2,
            'phase': np.angle(coefficients),
            'coi': coi_period
        }
    except Exception as e:
        return {'error': f'CWT computation failed: {str(e)}'}

    return results


def discrete_wavelet_transform(df, column: str, wavelet: str = 'db4', level: int = 3) -> Dict[str, Any]:
    data = df[column].dropna().values
    if not PYWT_AVAILABLE:
        coeffs = []
        approx = data.copy()
        for i in range(level):
            window = 2 ** (i + 1)
            if window > len(approx):
                break
            cA = np.convolve(approx, np.ones(window) / window, mode='valid')
            cD = approx[:len(cA)] - cA
            coeffs.append({
                'level': i + 1,
                'approximation': cA,
                'detail': cD,
                'approximation_length': len(cA),
                'detail_length': len(cD)
            })
            approx = cA

        results = {
            'coefficients': coeffs,
            'final_approximation': approx,
            'wavelet': f'{wavelet} (approximate - pywt not available)',
            'decomposition_level': level,
            'note': 'Using simple filtering - install pywt for proper wavelets'
        }
        return results

    coeffs = []
    approx = data
    for i in range(level):
        cA, cD = pywt.dwt(approx, wavelet)
        coeffs.append({
            'level': i + 1,
            'approximation': cA,
            'detail': cD,
            'approximation_length': len(cA),
            'detail_length': len(cD)
        })
        approx = cA

    results = {
        'coefficients': coeffs,
        'final_approximation': approx,
        'wavelet': wavelet,
        'decomposition_level': level
    }

    return results


def plot_wavelet_power(cwt_results: Dict[str, Any], column: str) -> plt.Figure:
    power = cwt_results.get('power')
    scales = cwt_results.get('scales')
    time = cwt_results.get('time')

    if power is None:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    periods = cwt_results.get('periods')
    if periods is not None:
        Y = periods
        y_label = 'Period'
    else:
        Y = scales
        y_label = 'Scale'

    log_power = np.log(power + 1e-8)
    im = ax.contourf(time, Y, log_power, levels=50, cmap='jet')
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time')
    ax.set_title(f'Continuous Wavelet Transform - {column}')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log(Power)')

    coi = cwt_results.get('coi')
    if coi is not None and len(coi) == len(time):
        try:
            ax.plot(time, coi, 'w--', linewidth=2)
        except Exception:
            pass

    plt.tight_layout()
    return fig


def plot_wavelet_torrence(cwt_results: Dict[str, Any], column: str, y_scale: str = 'log', significance_level: float = 0.95, show_coi: bool = True, wavelet: str = None, ax=None) -> plt.Figure:
    """
    Plot CWT power spectrum following Torrence & Compo (1998) style.

    If ax=None: Creates full layout with:
    - Main power spectrum heatmap with COI shading
    - Global wavelet power spectrum on the right
    - Scale-averaged power (time-averaged variance) on the bottom

    If ax is provided: Plots only to that axes (for embedding in custom layouts)
    """
    power = cwt_results.get('power')
    time = cwt_results.get('time')
    periods = cwt_results.get('periods')
    coi = cwt_results.get('coi')

    if power is None or periods is None or time is None:
        return None

    # Normalize power for better visualization
    coefficients = cwt_results.get('coefficients', power)
    if np.iscomplexobj(coefficients):
        variance = np.var(coefficients.real)
    else:
        variance = np.var(coefficients)
    if variance == 0:
        variance = 1.0
    normalized_power = power / variance
    log_power = np.log2(normalized_power + 1e-10)

    # Helper function to plot main heatmap on an axes
    def plot_main_heatmap(ax_main, add_colorbar=True, show_xlabel=True):
        # Use pcolormesh for full heatmap without truncation
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        time_edges = np.concatenate([time - dt/2, [time[-1] + dt/2]])

        # For periods, use log spacing for edges
        period_edges = np.zeros(len(periods) + 1)
        period_edges[0] = periods[0] * 0.9
        period_edges[-1] = periods[-1] * 1.1
        for i in range(1, len(periods)):
            period_edges[i] = np.sqrt(periods[i-1] * periods[i])

        # Plot the full power spectrum
        im = ax_main.pcolormesh(time_edges, period_edges, log_power,
                                 cmap='viridis', shading='flat')

        if y_scale == 'log':
            ax_main.set_yscale('log')
        # Small periods (high freq) at TOP, large periods at BOTTOM
        # Invert y-axis: set ylim with large value first, small value second
        ax_main.set_ylim([periods[-1] * 1.1, periods[0] * 0.9])

        # Draw COI as filled region (hatched area = unreliable)
        if show_coi and coi is not None and len(coi) == len(time):
            ax_main.fill_between(time, coi, periods[-1] * 2,
                                  facecolor='white', alpha=0.3,
                                  hatch='///', edgecolor='gray')
            ax_main.plot(time, coi, 'k--', linewidth=2, label='COI')

        # Add significance contours
        try:
            mean_power_scale = np.mean(power, axis=1)
            chi2_crit = chi2.ppf(significance_level, df=2)
            signif = mean_power_scale * chi2_crit / 2.0
            mask = power > signif[:, None]
            ax_main.contour(time, periods, mask.astype(int), levels=[0.5],
                            colors='black', linewidths=1.5)
        except Exception:
            pass

        ax_main.set_ylabel('Period', fontsize=12)
        if show_xlabel:
            ax_main.set_xlabel('Time', fontsize=12)
        title = f'Wavelet Power Spectrum (Torrence & Compo) - {column}'
        if wavelet:
            title += f' [{wavelet}]'
        ax_main.set_title(title, fontsize=14)

        return im

    # ═══════════════════════════════════════════════════════════════
    # MODE 1: If ax is provided, just plot to that axes (simple mode)
    # ═══════════════════════════════════════════════════════════════
    if ax is not None:
        fig = ax.figure
        im = plot_main_heatmap(ax, add_colorbar=True, show_xlabel=True)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Log₂(Power/Variance)')
        return fig

    # ═══════════════════════════════════════════════════════════════
    # MODE 2: Full Torrence & Compo layout with side panels
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[4, 1], height_ratios=[3, 1, 0.15],
                          hspace=0.15, wspace=0.08)

    # Main power spectrum axes
    ax_main = fig.add_subplot(gs[0, 0])
    # Global power spectrum (right panel)
    ax_global = fig.add_subplot(gs[0, 1], sharey=ax_main)
    # Scale-averaged power (bottom panel)
    ax_scale_avg = fig.add_subplot(gs[1, 0], sharex=ax_main)
    # Colorbar axes
    ax_cbar = fig.add_subplot(gs[2, 0])

    # Plot main heatmap
    im = plot_main_heatmap(ax_main, add_colorbar=False, show_xlabel=False)
    ax_main.tick_params(labelbottom=False)  # Hide x labels on main plot

    # ═══════════════════════════════════════════════════════════════
    # GLOBAL WAVELET POWER SPECTRUM (right panel)
    # ═══════════════════════════════════════════════════════════════
    global_power = np.mean(normalized_power, axis=1)
    ax_global.plot(global_power, periods, 'b-', linewidth=1.5)

    # Add significance level for global power
    try:
        mean_power_scale = np.mean(power, axis=1)
        chi2_crit = chi2.ppf(significance_level, df=2)
        signif = mean_power_scale * chi2_crit / 2.0
        dof = len(time) - len(periods)
        if dof < 2:
            dof = 2
        chi2_crit_global = chi2.ppf(significance_level, df=dof)
        global_signif = np.mean(signif) * chi2_crit_global / dof
        ax_global.axvline(global_signif, color='r', linestyle='--',
                          linewidth=1, label=f'{int(significance_level*100)}% signif')
    except Exception:
        pass

    ax_global.set_xlabel('Power', fontsize=10)
    ax_global.set_title('Global\nPower', fontsize=10)
    ax_global.tick_params(labelleft=False)
    ax_global.set_xlim([0, np.max(global_power) * 1.1])
    if y_scale == 'log':
        ax_global.set_yscale('log')
    # Match main plot: small periods at top (inverted)
    ax_global.set_ylim([periods[-1] * 1.1, periods[0] * 0.9])    # ═══════════════════════════════════════════════════════════════
    # SCALE-AVERAGED POWER (bottom panel) - Time series of variance
    # ═══════════════════════════════════════════════════════════════
    scale_avg_power = np.mean(normalized_power, axis=0)
    ax_scale_avg.plot(time, scale_avg_power, 'b-', linewidth=1)
    ax_scale_avg.fill_between(time, 0, scale_avg_power, alpha=0.3)
    ax_scale_avg.set_xlabel('Time', fontsize=12)
    ax_scale_avg.set_ylabel('Avg Power', fontsize=10)
    ax_scale_avg.set_title('Scale-Averaged Power (Variance)', fontsize=10)
    ax_scale_avg.set_xlim([time[0], time[-1]])
    ax_scale_avg.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════
    # COLORBAR
    # ═══════════════════════════════════════════════════════════════
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Log₂(Power/Variance)', fontsize=10)

    plt.tight_layout()
    return fig


def plot_discrete_wavelet(dwt_results: Dict[str, Any], column: str) -> plt.Figure:
    """
    Plot DWT decomposition showing:
    - Detail coefficients for each level (cD1, cD2, cD3, ...)
    - Final approximation coefficients at the bottom

    Following standard wavelet decomposition visualization.
    """
    coefficients = dwt_results.get('coefficients')
    final_approx = dwt_results.get('final_approximation')
    wavelet_name = dwt_results.get('wavelet', 'Unknown')

    if coefficients is None:
        return None

    n_levels = len(coefficients)

    # Create figure with n_levels + 1 subplots (details + final approximation)
    fig, axes = plt.subplots(n_levels + 1, 1, figsize=(14, 3 * (n_levels + 1)))

    if n_levels == 0:
        return fig

    # Ensure axes is always iterable
    if n_levels == 0:
        axes = [axes]

    # Color scheme for different levels
    detail_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_levels))

    # Plot detail coefficients for each level
    for i, coeff in enumerate(coefficients):
        ax = axes[i]
        detail = coeff['detail']
        level = coeff['level']

        # Create x-axis that represents time position
        x = np.linspace(0, 1, len(detail))

        # Plot detail coefficients as stem plot for better visualization
        ax.fill_between(x, 0, detail, alpha=0.5, color=detail_colors[i])
        ax.plot(x, detail, color=detail_colors[i], linewidth=0.8)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        ax.set_ylabel(f'cD{level}', fontsize=11, fontweight='bold')
        ax.set_title(f'Level {level} Detail Coefficients (High Frequency)', fontsize=10)
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelbottom=False)  # Hide x labels except last

        # Add statistics
        rms = np.sqrt(np.mean(detail**2))
        ax.text(0.98, 0.95, f'RMS: {rms:.4f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot final approximation at the bottom
    ax_approx = axes[-1]
    x_approx = np.linspace(0, 1, len(final_approx))
    ax_approx.fill_between(x_approx, 0, final_approx, alpha=0.5, color='green')
    ax_approx.plot(x_approx, final_approx, color='darkgreen', linewidth=1.5)
    ax_approx.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax_approx.set_ylabel(f'cA{n_levels}', fontsize=11, fontweight='bold')
    ax_approx.set_title(f'Final Approximation Coefficients (Level {n_levels} - Low Frequency)', fontsize=10)
    ax_approx.set_xlabel('Normalized Time', fontsize=11)
    ax_approx.set_xlim([0, 1])
    ax_approx.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle(f'Discrete Wavelet Transform Decomposition - {column}\n'
                 f'Wavelet: {wavelet_name}, Levels: {n_levels}',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# =============================================================================
# EXTENDED SPECTRAL ANALYSIS: Coherence, Cross-Wavelet, Harmonic Analysis
# =============================================================================

def coherence_analysis(df, column1: str, column2: str, sampling_rate: float = 1.0,
                       nperseg: int = None, noverlap: int = None) -> Dict[str, Any]:
    """
    Compute magnitude-squared coherence between two signals.
    
    Coherence measures the linear correlation between two signals as a 
    function of frequency. Values range from 0 (no correlation) to 1 
    (perfect linear correlation at that frequency).
    
    Args:
        df: DataFrame containing both columns
        column1: First signal column name
        column2: Second signal column name
        sampling_rate: Sampling frequency (samples per unit time)
        nperseg: Length of each segment for Welch method
        noverlap: Number of overlapping samples
        
    Returns:
        Dictionary with coherence results including frequencies, coherence,
        phase, and cross-spectral density
    """
    from scipy.signal import coherence as scipy_coherence, csd
    
    data1 = df[column1].dropna().values
    data2 = df[column2].dropna().values
    
    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    if nperseg is None:
        nperseg = min(256, min_len // 4)
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute coherence
    frequencies, coh = scipy_coherence(data1, data2, fs=sampling_rate,
                                        nperseg=nperseg, noverlap=noverlap)
    
    # Compute cross-spectral density for phase information
    freq_csd, Pxy = csd(data1, data2, fs=sampling_rate,
                        nperseg=nperseg, noverlap=noverlap)
    
    # Phase angle (in radians and degrees)
    phase_rad = np.angle(Pxy)
    phase_deg = np.degrees(phase_rad)
    
    # Find frequencies with significant coherence (> 0.5)
    significant_mask = coh > 0.5
    significant_freqs = frequencies[significant_mask]
    significant_coh = coh[significant_mask]
    
    # Find peak coherence
    peak_idx = np.argmax(coh)
    
    results = {
        'frequencies': frequencies,
        'coherence': coh,
        'phase_radians': phase_rad,
        'phase_degrees': phase_deg,
        'cross_spectral_density': Pxy,
        'peak_frequency': float(frequencies[peak_idx]),
        'peak_coherence': float(coh[peak_idx]),
        'peak_phase_deg': float(phase_deg[peak_idx]),
        'significant_frequencies': significant_freqs.tolist(),
        'significant_coherence': significant_coh.tolist(),
        'mean_coherence': float(np.mean(coh)),
        'sampling_rate': sampling_rate,
        'nperseg': nperseg,
        'column1': column1,
        'column2': column2
    }
    
    return results


def plot_coherence(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot coherence analysis results with coherence and phase spectra.
    """
    frequencies = results.get('frequencies')
    coherence = results.get('coherence')
    phase_deg = results.get('phase_degrees')
    col1 = results.get('column1', 'Signal 1')
    col2 = results.get('column2', 'Signal 2')
    
    if frequencies is None or coherence is None:
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Coherence plot
    axes[0].plot(frequencies, coherence, 'b-', linewidth=1.5)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Significance threshold (0.5)')
    axes[0].fill_between(frequencies, 0, coherence, alpha=0.3)
    axes[0].set_ylabel('Coherence', fontsize=11)
    axes[0].set_title(f'Magnitude-Squared Coherence: {col1} vs {col2}', fontsize=12)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Phase plot
    axes[1].plot(frequencies, phase_deg, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Frequency', fontsize=11)
    axes[1].set_ylabel('Phase (degrees)', fontsize=11)
    axes[1].set_title('Phase Spectrum', fontsize=12)
    axes[1].set_ylim([-180, 180])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def cross_wavelet_transform(df, column1: str, column2: str, scales: np.ndarray = None,
                            wavelet: str = 'cmor1.5-1.0', sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Compute Cross-Wavelet Transform (XWT) between two time series.
    
    XWT reveals common power and relative phase in time-frequency space,
    useful for detecting relationships between signals that vary over time.
    
    Args:
        df: DataFrame containing both columns
        column1: First signal column name
        column2: Second signal column name
        scales: Wavelet scales to analyze
        wavelet: Wavelet type (default: 'cmor1.5-1.0' for Complex Morlet).
                 Must be a complex wavelet to compute phase relationships.
                 Options: 'cmor1.5-1.0', 'cmor2.0-1.0', 'cgau1', etc.
        sampling_rate: Sampling frequency
        
    Returns:
        Dictionary with XWT results including cross-power, phase, and significance
    """
    if not PYWT_AVAILABLE:
        return {'error': 'PyWavelets not installed. Install with: pip install PyWavelets'}
    
    data1 = df[column1].dropna().values
    data2 = df[column2].dropna().values
    
    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    n = min_len
    
    if sampling_rate <= 0:
        return {'error': 'sampling_rate must be > 0'}
    dt = 1.0 / float(sampling_rate)
    
    if scales is None:
        scales = np.arange(1, min(128, n // 2))
    
    try:
        # Compute CWT for both signals using complex wavelet
        coeffs1, freqs1 = pywt.cwt(data1, scales, wavelet)
        coeffs2, freqs2 = pywt.cwt(data2, scales, wavelet)
        
        # Cross-wavelet spectrum: W1 * conj(W2)
        xwt = coeffs1 * np.conj(coeffs2)
        
        # Cross-wavelet power
        xwt_power = np.abs(xwt)
        
        # Phase difference - requires complex coefficients for full -pi to pi range
        phase_diff = np.angle(xwt)
        
        # Convert scales to periods using Fourier factor (consistent with CWT)
        # Use standard Morlet formula for consistency
        omega0 = 6.0
        fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))
        periods = scales * fourier_factor * dt
        
        # Cone of influence (consistent with CWT)
        half = n // 2
        if n % 2 == 0:
            left = np.arange(0, half)
            right = np.arange(half, 0, -1)
        else:
            left = np.arange(0, half + 1)
            right = np.arange(half, 0, -1)
        coi_time = dt * np.concatenate((left, right))
        if coi_time.shape[0] != n:
            coi_time = dt * np.linspace(0, (n - 1) * 0.5, n)
        # COI uses same Fourier factor as period calculation
        coi_period = coi_time * fourier_factor
        
        # Compute individual powers for normalization
        power1 = np.abs(coeffs1) ** 2
        power2 = np.abs(coeffs2) ** 2
        
        results = {
            'xwt': xwt,
            'xwt_power': xwt_power,
            'phase_difference': phase_diff,
            'scales': scales,
            'periods': periods,
            'time': np.arange(n) * dt,
            'wavelet': wavelet,
            'coi': coi_period,
            'power1': power1,
            'power2': power2,
            'coefficients1': coeffs1,
            'coefficients2': coeffs2,
            'column1': column1,
            'column2': column2
        }
        
    except Exception as e:
        return {'error': f'XWT computation failed: {str(e)}'}
    
    return results


def wavelet_coherence(df, column1: str, column2: str, scales: np.ndarray = None,
                      wavelet: str = 'cmor1.5-1.0', sampling_rate: float = 1.0,
                      smooth_factor: int = 5) -> Dict[str, Any]:
    """
    Compute Wavelet Coherence (WTC) between two time series.
    
    WTC is a measure of the intensity of the covariance of the two series
    in time-frequency space, normalized by their individual power spectra.
    Values range from 0 to 1 like regular coherence.
    
    Args:
        df: DataFrame containing both columns
        column1: First signal column name
        column2: Second signal column name
        scales: Wavelet scales to analyze
        wavelet: Wavelet type (default: 'cmor1.5-1.0' for Complex Morlet).
                 Must be a complex wavelet to compute phase relationships.
        sampling_rate: Sampling frequency
        smooth_factor: Smoothing window size for coherence estimation
        
    Returns:
        Dictionary with WTC results including coherence, phase, and arrows for plotting
    """
    if not PYWT_AVAILABLE:
        return {'error': 'PyWavelets not installed. Install with: pip install PyWavelets'}
    
    # First compute XWT
    xwt_results = cross_wavelet_transform(df, column1, column2, scales, wavelet, sampling_rate)
    
    if 'error' in xwt_results:
        return xwt_results
    
    xwt = xwt_results['xwt']
    power1 = xwt_results['power1']
    power2 = xwt_results['power2']
    
    # Smoothing function (simple moving average in time)
    def smooth(data, window):
        if window < 2:
            return data
        kernel = np.ones(window) / window
        smoothed = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed[i] = np.convolve(data[i], kernel, mode='same')
        return smoothed
    
    # Smooth the cross-spectrum and individual power spectra
    smooth_xwt = smooth(xwt, smooth_factor)
    smooth_power1 = smooth(power1, smooth_factor)
    smooth_power2 = smooth(power2, smooth_factor)
    
    # Wavelet coherence: |S(W1 * conj(W2))|^2 / (S(|W1|^2) * S(|W2|^2))
    numerator = np.abs(smooth_xwt) ** 2
    denominator = smooth_power1 * smooth_power2
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)
    
    wtc = numerator / denominator
    wtc = np.clip(wtc, 0, 1)  # Ensure values are in [0, 1]
    
    # Phase difference from smoothed XWT
    phase_diff = np.angle(smooth_xwt)
    
    results = {
        'coherence': wtc,
        'phase_difference': phase_diff,
        'scales': xwt_results['scales'],
        'periods': xwt_results['periods'],
        'time': xwt_results['time'],
        'wavelet': wavelet,
        'coi': xwt_results['coi'],
        'xwt_power': xwt_results['xwt_power'],
        'column1': column1,
        'column2': column2,
        'smooth_factor': smooth_factor
    }
    
    return results


def plot_cross_wavelet(results: Dict[str, Any], show_phase_arrows: bool = True,
                       arrow_density: tuple = (3, 3)) -> plt.Figure:
    """
    Plot Cross-Wavelet Transform with phase arrows.
    
    Phase arrow convention follows Torrence and Webster (1999) / Grinsted et al. (2004):
    - ↑ (up/N): In-phase (0°) - signals move together
    - ↓ (down/S): Anti-phase (180°) - signals move opposite
    - → (right/E): X leads Y by 90°
    - ← (left/W): Y leads X by 90° (X lags)
    
    Args:
        results: Dictionary from cross_wavelet_transform()
        show_phase_arrows: Whether to display phase arrows
        arrow_density: Tuple (y_skip, x_skip) controlling arrow density.
                       Higher values = fewer arrows. Default (3, 3).
    """
    xwt_power = results.get('xwt_power')
    phase_diff = results.get('phase_difference')
    time = results.get('time')
    periods = results.get('periods')
    coi = results.get('coi')
    col1 = results.get('column1', 'Signal 1')
    col2 = results.get('column2', 'Signal 2')
    
    if xwt_power is None or time is None or periods is None:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot log of cross-wavelet power using log2 transform on both power and y-axis
    log_power = np.log2(xwt_power + 1e-10)
    log_periods = np.log2(periods)
    
    im = ax.contourf(time, log_periods, log_power, 32, cmap='jet', extend='both')
    
    # Set axis limits: small periods (high freq) at top, large periods at bottom
    # periods[0] is smallest, periods[-1] is largest
    # ylim[0] = bottom, ylim[1] = top, so we want large at bottom, small at top
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([log_periods[-1], log_periods[0]])  # Large periods at bottom, small at top
    
    # COI - shade area where edge effects are significant
    if coi is not None and len(coi) == len(time):
        coi_clipped = np.clip(coi, periods[0], periods[-1])
        log_coi = np.log2(coi_clipped)
        # Fill from COI down to largest period (bottom of plot)
        ax.fill_between(time, log_coi, log_periods[-1],
                        facecolor='white', alpha=0.4,
                        hatch='///', edgecolor='gray')
        ax.plot(time, log_coi, 'k--', linewidth=1.5, label='COI')
    
    # Phase arrows using quiver - following pycwt convention exactly
    # angle = 0.5*pi - phase transforms so arrows rotate with 'north' origin:
    # - In-phase (0°) -> UP, Anti-phase (180°) -> DOWN
    # - X leads 90° -> RIGHT, Y leads 90° -> LEFT
    if show_phase_arrows and phase_diff is not None:
        angle = 0.5 * np.pi - phase_diff
        u_full = np.cos(angle)
        v_full = np.sin(angle)
        
        # Adaptive sampling: ~12 rows, ~20 columns total for uniform visual density
        n_rows_target = 12
        n_cols_target = 20
        
        row_indices = np.linspace(0, len(periods) - 1, n_rows_target).astype(int)
        col_step = max(1, len(time) // n_cols_target)
        col_indices = np.arange(0, len(time), col_step)
        
        # Extract sampled data
        t_sample = time[col_indices]
        p_sample = log_periods[row_indices]
        u_sample = u_full[np.ix_(row_indices, col_indices)]
        v_sample = v_full[np.ix_(row_indices, col_indices)]
        
        # Create meshgrid for quiver positions
        T, P = np.meshgrid(t_sample, p_sample)
        
        # Use width parameter (fraction of plot width) for thin arrows
        ax.quiver(T, P, u_sample, v_sample,
                  units='width', angles='uv', pivot='mid',
                  width=0.002, color='black',
                  headwidth=4, headlength=4, headaxislength=3,
                  scale=35)
    
    # Set y-axis ticks to show actual period values
    y_ticks = np.log2(periods[::max(1, len(periods)//8)])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{2**y:.2f}' for y in y_ticks])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    ax.set_title(f'Cross-Wavelet Transform: {col1} vs {col2}\n'
                 f'(↑ in-phase, ↓ anti-phase, → {col1} leads 90°, ← {col2} leads 90°)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log₂(Power)', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_wavelet_coherence(results: Dict[str, Any], show_phase_arrows: bool = True,
                           arrow_density: tuple = (3, 3)) -> plt.Figure:
    """
    Plot Wavelet Coherence with phase arrows.
    
    Similar to Torrence & Compo style but for coherence between two signals.
    Uses jet colormap for coherence (0-1).
    
    Phase arrow convention follows Torrence and Webster (1999) / Grinsted et al. (2004):
    - ↑ (up/N): In-phase (0°) - signals move together
    - ↓ (down/S): Anti-phase (180°) - signals move opposite
    - → (right/E): X leads Y by 90°
    - ← (left/W): Y leads X by 90° (X lags)
    
    Args:
        results: Dictionary from wavelet_coherence()
        show_phase_arrows: Whether to display phase arrows
        arrow_density: Tuple (y_skip, x_skip) controlling arrow density.
                       Higher values = fewer arrows. Default (3, 3).
    """
    wtc = results.get('coherence')
    phase_diff = results.get('phase_difference')
    time = results.get('time')
    periods = results.get('periods')
    coi = results.get('coi')
    col1 = results.get('column1', 'Signal 1')
    col2 = results.get('column2', 'Signal 2')
    
    if wtc is None or time is None or periods is None:
        return None
    
    # Clip extreme values for cleaner visualization
    wtc_clipped = np.clip(wtc, 0, 1)
    log_periods = np.log2(periods)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot coherence with jet colormap (standard for coherence)
    im = ax.contourf(time, log_periods, wtc_clipped, 32, cmap='jet', 
                     extend='both', vmin=0, vmax=1)
    
    # Add significance contour at 0.5 (single clean line)
    try:
        cs = ax.contour(time, log_periods, wtc_clipped, levels=[0.5], colors='black', linewidths=1.5)
    except Exception:
        pass  # Skip if contour fails
    
    # Set axis limits: small periods (high freq) at top, large periods at bottom
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([log_periods[-1], log_periods[0]])  # Large periods at bottom, small at top
    
    # COI - shade area where edge effects are significant
    if coi is not None and len(coi) == len(time):
        coi_clipped = np.clip(coi, periods[0], periods[-1])
        log_coi = np.log2(coi_clipped)
        # Fill from COI down to largest period (bottom of plot)
        ax.fill_between(time, log_coi, log_periods[-1],
                        facecolor='white', alpha=0.4,
                        hatch='///', edgecolor='gray')
        ax.plot(time, log_coi, 'k--', linewidth=1.5, label='COI')
    
    # Phase arrows using quiver - following pycwt convention exactly
    # angle = 0.5*pi - phase transforms so arrows rotate with 'north' origin:
    # - In-phase (0°) -> UP, Anti-phase (180°) -> DOWN
    # - X leads 90° -> RIGHT, Y leads 90° -> LEFT
    if show_phase_arrows and phase_diff is not None:
        angle = 0.5 * np.pi - phase_diff
        u_full = np.cos(angle)
        v_full = np.sin(angle)
        
        # Adaptive sampling: ~12 rows, ~20 columns total for uniform visual density
        n_rows_target = 12
        n_cols_target = 20
        
        row_indices = np.linspace(0, len(periods) - 1, n_rows_target).astype(int)
        col_step = max(1, len(time) // n_cols_target)
        col_indices = np.arange(0, len(time), col_step)
        
        # Extract sampled data
        t_sample = time[col_indices]
        p_sample = log_periods[row_indices]
        u_sample = u_full[np.ix_(row_indices, col_indices)]
        v_sample = v_full[np.ix_(row_indices, col_indices)]
        
        # Create meshgrid for quiver positions
        T, P = np.meshgrid(t_sample, p_sample)
        
        # Use width parameter (fraction of plot width) for thin arrows
        ax.quiver(T, P, u_sample, v_sample,
                  units='width', angles='uv', pivot='mid',
                  width=0.002, color='black',
                  headwidth=4, headlength=4, headaxislength=3,
                  scale=35)
    
    # Set y-axis ticks to show actual period values
    y_ticks = np.log2(periods[::max(1, len(periods)//8)])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{2**y:.2f}' for y in y_ticks])
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    ax.set_title(f'Wavelet Coherence: {col1} vs {col2}\n'
                 f'(↑ in-phase, ↓ anti-phase, → {col1} leads 90°, ← {col2} leads 90°)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coherence', fontsize=10)
    
    plt.tight_layout()
    return fig


def harmonic_analysis(df, column: str, n_harmonics: int = 5, 
                      sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Perform harmonic analysis using least-squares fitting of sinusoids.
    
    Fits a sum of sinusoids to the data to extract dominant periodic components.
    This is particularly useful for tidal analysis, seasonal patterns, and
    any data with known periodic structure.
    
    Args:
        df: DataFrame containing the data
        column: Column name to analyze
        n_harmonics: Number of harmonics to fit
        sampling_rate: Sampling frequency
        
    Returns:
        Dictionary with fitted harmonics, amplitudes, phases, and residuals
    """
    from scipy.optimize import curve_fit
    
    data = df[column].dropna().values
    n = len(data)
    t = np.arange(n) / sampling_rate
    
    # First, use FFT to identify dominant frequencies
    fft_vals = fft(data)
    freqs = fftfreq(n, 1/sampling_rate)
    power = np.abs(fft_vals[:n//2]) ** 2
    positive_freqs = freqs[:n//2]
    
    # Find top n_harmonics frequencies (excluding DC)
    power_no_dc = power[1:]
    top_indices = np.argsort(power_no_dc)[-n_harmonics:][::-1] + 1
    dominant_freqs = positive_freqs[top_indices]
    
    # Build design matrix for least-squares fit
    # Model: y = a0 + sum_i(A_i * cos(2*pi*f_i*t) + B_i * sin(2*pi*f_i*t))
    X = np.ones((n, 1 + 2 * n_harmonics))
    X[:, 0] = 1  # Constant term
    
    for i, freq in enumerate(dominant_freqs):
        X[:, 1 + 2*i] = np.cos(2 * np.pi * freq * t)
        X[:, 2 + 2*i] = np.sin(2 * np.pi * freq * t)
    
    # Least-squares fit
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, data, rcond=None)
    except Exception as e:
        return {'error': f'Least-squares fitting failed: {str(e)}'}
    
    # Extract amplitudes and phases
    harmonics = []
    mean_value = coeffs[0]
    fitted = np.ones(n) * mean_value
    
    for i, freq in enumerate(dominant_freqs):
        A = coeffs[1 + 2*i]  # cos coefficient
        B = coeffs[2 + 2*i]  # sin coefficient
        
        amplitude = np.sqrt(A**2 + B**2)
        phase = np.arctan2(B, A)  # Phase in radians
        period = 1.0 / freq if freq > 0 else np.inf
        
        # Add to fitted curve
        fitted += A * np.cos(2 * np.pi * freq * t) + B * np.sin(2 * np.pi * freq * t)
        
        harmonics.append({
            'frequency': float(freq),
            'period': float(period),
            'amplitude': float(amplitude),
            'phase_radians': float(phase),
            'phase_degrees': float(np.degrees(phase)),
            'cos_coefficient': float(A),
            'sin_coefficient': float(B),
            'variance_explained': float((amplitude**2 / 2) / np.var(data) * 100)
        })
    
    # Sort by amplitude
    harmonics = sorted(harmonics, key=lambda x: x['amplitude'], reverse=True)
    
    # Calculate residuals and goodness of fit
    residual = data - fitted
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    results = {
        'harmonics': harmonics,
        'mean_value': float(mean_value),
        'fitted_values': fitted,
        'residuals': residual,
        'original_data': data,
        'time': t,
        'r_squared': float(r_squared),
        'rmse': float(np.sqrt(np.mean(residual**2))),
        'total_variance_explained': float(r_squared * 100),
        'n_harmonics': n_harmonics,
        'sampling_rate': sampling_rate,
        'column': column
    }
    
    return results


def plot_harmonic_analysis(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot harmonic analysis results showing original data, fitted curve,
    residuals, and individual harmonic components.
    """
    original = results.get('original_data')
    fitted = results.get('fitted_values')
    residuals = results.get('residuals')
    time = results.get('time')
    harmonics = results.get('harmonics', [])
    column = results.get('column', 'Data')
    r_squared = results.get('r_squared', 0)
    
    if original is None or fitted is None:
        return None
    
    n_plots = 3 + min(len(harmonics), 3)  # Original+fit, residuals, spectrum, top 3 harmonics
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    
    # Plot 1: Original data and fitted curve
    axes[0].plot(time, original, 'b-', alpha=0.7, linewidth=1, label='Original')
    axes[0].plot(time, fitted, 'r-', linewidth=2, label=f'Fitted (R² = {r_squared:.4f})')
    axes[0].set_ylabel('Value', fontsize=10)
    axes[0].set_title(f'Harmonic Analysis: {column}', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    axes[1].plot(time, residuals, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].fill_between(time, 0, residuals, alpha=0.3, color='green')
    axes[1].set_ylabel('Residual', fontsize=10)
    axes[1].set_title('Residuals', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Harmonic amplitudes bar chart
    if harmonics:
        freqs = [h['frequency'] for h in harmonics]
        amps = [h['amplitude'] for h in harmonics]
        periods = [h['period'] for h in harmonics]
        
        bars = axes[2].bar(range(len(freqs)), amps, color='steelblue', alpha=0.7)
        axes[2].set_xticks(range(len(freqs)))
        axes[2].set_xticklabels([f'{f:.4f}\n(T={p:.2f})' for f, p in zip(freqs, periods)], fontsize=8)
        axes[2].set_ylabel('Amplitude', fontsize=10)
        axes[2].set_xlabel('Frequency (Period)', fontsize=10)
        axes[2].set_title('Harmonic Amplitudes', fontsize=12)
        axes[2].grid(True, alpha=0.3, axis='y')
    
    # Plot individual harmonics
    mean_val = results.get('mean_value', 0)
    for i, h in enumerate(harmonics[:3]):
        ax_idx = 3 + i
        if ax_idx < len(axes):
            freq = h['frequency']
            A = h['cos_coefficient']
            B = h['sin_coefficient']
            component = A * np.cos(2 * np.pi * freq * time) + B * np.sin(2 * np.pi * freq * time)
            
            axes[ax_idx].plot(time, component, linewidth=1.5)
            axes[ax_idx].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[ax_idx].set_ylabel('Value', fontsize=10)
            axes[ax_idx].set_title(f'Harmonic {i+1}: f={freq:.4f}, T={h["period"]:.2f}, A={h["amplitude"]:.4f}', fontsize=10)
            axes[ax_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time', fontsize=11)
    
    plt.tight_layout()
    return fig


# =============================================================================
# PLOTLY VERSIONS OF WAVELET PLOTS (for Streamlit interactive charts)
# =============================================================================

def _downsample_wavelet_data(z, x, y, max_size=150):
    """
    Downsample wavelet data for faster Plotly rendering.
    Returns downsampled z, x, y arrays.
    """
    from scipy.ndimage import zoom
    
    n_scales, n_times = z.shape
    
    # Calculate downsampling factors
    scale_factor = min(1.0, max_size / n_scales)
    time_factor = min(1.0, max_size / n_times)
    
    if scale_factor < 1.0 or time_factor < 1.0:
        # Downsample z using zoom (faster than interpolation)
        z_down = zoom(z, (scale_factor, time_factor), order=1)
        
        # Downsample x and y to match
        new_n_times = z_down.shape[1]
        new_n_scales = z_down.shape[0]
        
        x_indices = np.linspace(0, len(x) - 1, new_n_times).astype(int)
        y_indices = np.linspace(0, len(y) - 1, new_n_scales).astype(int)
        
        x_down = x[x_indices]
        y_down = y[y_indices]
        
        return z_down, x_down, y_down
    
    return z, x, y


def plot_wavelet_torrence_plotly(cwt_results: Dict[str, Any], column: str,
                                  y_scale: str = 'log', significance_level: float = 0.95,
                                  show_coi: bool = True, wavelet: str = None):
    """
    Plotly version of Torrence & Compo style wavelet power plot.
    
    Creates a 3-panel layout:
    - Main: Power spectrum heatmap with COI
    - Right: Global wavelet power spectrum
    - Bottom: Scale-averaged power (variance over time)
    
    Returns a Plotly Figure object for use with st.plotly_chart().
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    power = cwt_results.get('power')
    time = cwt_results.get('time')
    periods = cwt_results.get('periods')
    coi = cwt_results.get('coi')
    
    if power is None or periods is None or time is None:
        return None
    
    # Normalize power for better visualization
    coefficients = cwt_results.get('coefficients', power)
    if np.iscomplexobj(coefficients):
        variance = np.var(coefficients.real)
    else:
        variance = np.var(coefficients)
    if variance == 0:
        variance = 1.0
    normalized_power = power / variance
    log_power = np.log2(normalized_power + 1e-10)
    
    # Downsample for performance
    log_power_ds, time_ds, periods_ds = _downsample_wavelet_data(log_power, time, periods)
    
    # Compute global power (mean across time) - use original for accuracy
    global_power = np.mean(normalized_power, axis=1)
    
    # Compute scale-averaged power (mean across scales)
    scale_avg_power = np.mean(normalized_power, axis=0)
    # Downsample scale_avg_power to match time_ds
    if len(scale_avg_power) > 150:
        indices = np.linspace(0, len(scale_avg_power) - 1, 150).astype(int)
        scale_avg_power_ds = scale_avg_power[indices]
        time_avg_ds = time[indices]
    else:
        scale_avg_power_ds = scale_avg_power
        time_avg_ds = time
    
    # Create subplots: main heatmap (large), global power (right), scale-avg (bottom)
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.85, 0.15],
        row_heights=[0.75, 0.25],
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, None]],
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN HEATMAP - use go.Heatmap (faster than Contour)
    # ═══════════════════════════════════════════════════════════════
    fig.add_trace(
        go.Heatmap(
            z=log_power_ds,
            x=time_ds,
            y=periods_ds,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Log₂(Power/Var)', side='right'),
                len=0.65,
                y=0.7,
                yanchor='middle',
                thickness=15
            ),
            hovertemplate='Time: %{x:.2f}<br>Period: %{y:.2f}<br>Log₂Power: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add COI line
    if show_coi and coi is not None and len(coi) == len(time):
        # Clip COI to period range
        coi_clipped = np.clip(coi, periods[0], periods[-1])
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=coi_clipped,
                mode='lines',
                line=dict(color='white', width=2, dash='dash'),
                name='COI',
                showlegend=True,
                hovertemplate='COI at time %{x:.2f}: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add shaded region below COI (unreliable zone) - for inverted y-axis
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([coi_clipped, np.full(len(time), periods[-1])]),
                fill='toself',
                fillcolor='rgba(200, 200, 200, 0.4)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Unreliable region',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # ═══════════════════════════════════════════════════════════════
    # GLOBAL POWER (row 1, col 2)
    # ═══════════════════════════════════════════════════════════════
    fig.add_trace(
        go.Scatter(
            x=global_power,
            y=periods,
            mode='lines',
            line=dict(color='blue', width=1.5),
            name='Global Power',
            showlegend=False,
            hovertemplate='Power: %{x:.2f}<br>Period: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add significance line for global power
    try:
        mean_power_scale = np.mean(power, axis=1)
        chi2_crit = chi2.ppf(significance_level, df=2)
        signif = mean_power_scale * chi2_crit / 2.0
        dof = max(2, len(time) - len(periods))
        chi2_crit_global = chi2.ppf(significance_level, df=dof)
        global_signif = np.mean(signif) * chi2_crit_global / dof
        fig.add_trace(
            go.Scatter(
                x=[global_signif, global_signif],
                y=[periods[0], periods[-1]],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name=f'{int(significance_level*100)}% signif',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
    except Exception:
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # SCALE-AVERAGED POWER (row 2, col 1)
    # ═══════════════════════════════════════════════════════════════
    fig.add_trace(
        go.Scatter(
            x=time_avg_ds,
            y=scale_avg_power_ds,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 100, 200, 0.3)',
            line=dict(color='blue', width=1),
            name='Scale-Avg Power',
            showlegend=False,
            hovertemplate='Time: %{x:.2f}<br>Avg Power: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # ═══════════════════════════════════════════════════════════════
    # LAYOUT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    title = f'Wavelet Power Spectrum (Torrence & Compo) - {column}'
    if wavelet:
        title += f' [{wavelet}]'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=700,
        showlegend=True,
        legend=dict(x=1.02, y=0.5, xanchor='left'),
        template='plotly_white'
    )
    
    # Main heatmap axes - log scale with reversed range (small periods at top)
    fig.update_xaxes(title_text='', row=1, col=1, showticklabels=False)
    fig.update_yaxes(
        title_text='Period',
        type='log' if y_scale == 'log' else 'linear',
        autorange='reversed',
        row=1, col=1
    )
    
    # Global power axes - match main plot
    fig.update_xaxes(title_text='Power', row=1, col=2)
    fig.update_yaxes(
        type='log' if y_scale == 'log' else 'linear',
        autorange='reversed',
        showticklabels=False,
        row=1, col=2
    )
    
    # Scale-averaged power axes
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Avg Power', row=2, col=1)
    
    return fig


def plot_cross_wavelet_plotly(results: Dict[str, Any], show_phase_arrows: bool = True):
    """
    Plotly version of Cross-Wavelet Transform plot.
    
    Shows cross-wavelet power with optional phase arrows indicating
    the phase relationship between the two signals.
    
    Returns a Plotly Figure object.
    """
    import plotly.graph_objects as go
    
    xwt_power = results.get('xwt_power')
    phase_diff = results.get('phase_difference')
    time = results.get('time')
    periods = results.get('periods')
    coi = results.get('coi')
    col1 = results.get('column1', 'Signal 1')
    col2 = results.get('column2', 'Signal 2')
    
    if xwt_power is None or time is None or periods is None:
        return None
    
    log_power = np.log2(xwt_power + 1e-10)
    
    # Downsample for performance
    log_power_ds, time_ds, periods_ds = _downsample_wavelet_data(log_power, time, periods)
    
    fig = go.Figure()
    
    # Main heatmap (faster than Contour)
    fig.add_trace(
        go.Heatmap(
            z=log_power_ds,
            x=time_ds,
            y=periods_ds,
            colorscale='Turbo',
            colorbar=dict(
                title=dict(text='Log₂(XWT Power)', side='right'),
                y=0.5,
                yanchor='middle',
                thickness=15
            ),
            hovertemplate='Time: %{x:.2f}<br>Period: %{y:.2f}<br>Log₂Power: %{z:.2f}<extra></extra>'
        )
    )
    
    # COI line and shaded region - use original time for accuracy
    if coi is not None and len(coi) == len(time):
        # Downsample COI to match
        coi_indices = np.linspace(0, len(coi) - 1, min(150, len(coi))).astype(int)
        coi_ds = coi[coi_indices]
        time_coi_ds = time[coi_indices]
        coi_clipped = np.clip(coi_ds, periods[0], periods[-1])
        
        fig.add_trace(
            go.Scatter(
                x=time_coi_ds,
                y=coi_clipped,
                mode='lines',
                line=dict(color='white', width=2, dash='dash'),
                name='COI',
                hovertemplate='COI: %{y:.2f}<extra></extra>'
            )
        )
    
    # Phase arrows (fewer for performance)
    if show_phase_arrows and phase_diff is not None:
        step_t = max(1, len(time) // 8)
        step_s = max(1, len(periods) // 6)
        
        annotations = []
        threshold = np.percentile(xwt_power, 70)
        
        for i in range(step_s, len(periods) - step_s, step_s):
            for j in range(step_t, len(time) - step_t, step_t):
                if coi is not None and periods[i] > coi[j]:
                    continue
                    
                if xwt_power[i, j] > threshold:
                    # Transform angle following pycwt convention:
                    # 90° - phase_diff rotates arrows so north = X leads Y
                    angle = 0.5 * np.pi - phase_diff[i, j]
                    t_center = time[j]
                    p_center = periods[i]
                    
                    arrow_len_t = (time[-1] - time[0]) * 0.015
                    log_dp = 0.04
                    
                    dx = np.cos(angle) * arrow_len_t
                    dy_log = np.sin(angle) * log_dp
                    new_p = 10 ** (np.log10(periods[i]) + dy_log)
                    
                    new_p = np.clip(new_p, periods[0], periods[-1])
                    new_t = np.clip(t_center + dx, time[0], time[-1])
                    
                    annotations.append(dict(
                        x=new_t, y=new_p,
                        ax=t_center, ay=p_center,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2,
                        arrowsize=1, arrowwidth=1.5,
                        arrowcolor='black'
                    ))
        
        if len(annotations) > 40:
            annotations = annotations[::len(annotations) // 40]
        
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title=f'Cross-Wavelet Transform: {col1} vs {col2}<br><sup>Arrows: → in-phase, ← anti-phase</sup>',
        xaxis_title='Time',
        yaxis_title='Period',
        yaxis=dict(type='log', autorange='reversed'),
        height=550,
        template='plotly_white',
        margin=dict(r=80)
    )
    
    return fig


def plot_wavelet_coherence_plotly(results: Dict[str, Any], show_phase_arrows: bool = True):
    """
    Plotly version of Wavelet Coherence plot.
    
    Shows wavelet coherence (0-1) with significance contour at 0.5
    and optional phase arrows where coherence is significant.
    
    Returns a Plotly Figure object.
    """
    import plotly.graph_objects as go
    
    wtc = results.get('coherence')
    phase_diff = results.get('phase_difference')
    time = results.get('time')
    periods = results.get('periods')
    coi = results.get('coi')
    col1 = results.get('column1', 'Signal 1')
    col2 = results.get('column2', 'Signal 2')
    
    if wtc is None or time is None or periods is None:
        return None
    
    # Downsample for performance
    wtc_ds, time_ds, periods_ds = _downsample_wavelet_data(wtc, time, periods)
    
    fig = go.Figure()
    
    # Main heatmap - Green to Red colorscale (RdYlGn reversed)
    # Custom green-to-red: low coherence = green, high = red
    green_to_red = [
        [0.0, 'rgb(0, 128, 0)'],      # Green
        [0.25, 'rgb(144, 238, 144)'], # Light green
        [0.5, 'rgb(255, 255, 0)'],    # Yellow
        [0.75, 'rgb(255, 165, 0)'],   # Orange
        [1.0, 'rgb(255, 0, 0)']       # Red
    ]
    
    fig.add_trace(
        go.Heatmap(
            z=wtc_ds,
            x=time_ds,
            y=periods_ds,
            colorscale=green_to_red,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text='Coherence', side='right'),
                y=0.5,
                yanchor='middle',
                thickness=15
            ),
            hovertemplate='Time: %{x:.2f}<br>Period: %{y:.2f}<br>Coherence: %{z:.3f}<extra></extra>'
        )
    )
    
    # COI line - use downsampled version
    if coi is not None and len(coi) == len(time):
        coi_indices = np.linspace(0, len(coi) - 1, min(150, len(coi))).astype(int)
        coi_ds = coi[coi_indices]
        time_coi_ds = time[coi_indices]
        coi_clipped = np.clip(coi_ds, periods[0], periods[-1])
        
        fig.add_trace(
            go.Scatter(
                x=time_coi_ds,
                y=coi_clipped,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='COI',
                hovertemplate='COI: %{y:.2f}<extra></extra>'
            )
        )
    
    # Phase arrows where coherence is significant (> 0.5) - fewer for performance
    if show_phase_arrows and phase_diff is not None:
        step_t = max(1, len(time) // 8)
        step_s = max(1, len(periods) // 6)
        
        annotations = []
        
        for i in range(step_s, len(periods) - step_s, step_s):
            for j in range(step_t, len(time) - step_t, step_t):
                if coi is not None and periods[i] > coi[j]:
                    continue
                    
                if wtc[i, j] > 0.5:
                    # Transform angle following pycwt convention:
                    # 90° - phase_diff rotates arrows so north = X leads Y
                    angle = 0.5 * np.pi - phase_diff[i, j]
                    t_center = time[j]
                    p_center = periods[i]
                    
                    arrow_len_t = (time[-1] - time[0]) * 0.015
                    log_dp = 0.04
                    
                    dx = np.cos(angle) * arrow_len_t
                    dy_log = np.sin(angle) * log_dp
                    new_p = 10 ** (np.log10(periods[i]) + dy_log)
                    
                    new_p = np.clip(new_p, periods[0], periods[-1])
                    new_t = np.clip(t_center + dx, time[0], time[-1])
                    
                    annotations.append(dict(
                        x=new_t, y=new_p,
                        ax=t_center, ay=p_center,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2,
                        arrowsize=1, arrowwidth=1.5,
                        arrowcolor='black'
                    ))
        
        if len(annotations) > 35:
            annotations = annotations[::len(annotations) // 35]
        
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title=f'Wavelet Coherence: {col1} vs {col2}<br><sup>Arrows: → in-phase, ← anti-phase (where coherence > 0.5)</sup>',
        xaxis_title='Time',
        yaxis_title='Period',
        yaxis=dict(type='log', autorange='reversed'),
        height=550,
        template='plotly_white',
        margin=dict(r=80)
    )
    
    return fig
