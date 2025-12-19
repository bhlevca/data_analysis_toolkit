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
                            wavelet: str = 'morl', sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Compute Cross-Wavelet Transform (XWT) between two time series.
    
    XWT reveals common power and relative phase in time-frequency space,
    useful for detecting relationships between signals that vary over time.
    
    Args:
        df: DataFrame containing both columns
        column1: First signal column name
        column2: Second signal column name
        scales: Wavelet scales to analyze
        wavelet: Wavelet type (default: 'morl' for Morlet)
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
        # Compute CWT for both signals
        coeffs1, freqs1 = pywt.cwt(data1, scales, wavelet)
        coeffs2, freqs2 = pywt.cwt(data2, scales, wavelet)
        
        # Cross-wavelet spectrum: W1 * conj(W2)
        xwt = coeffs1 * np.conj(coeffs2)
        
        # Cross-wavelet power
        xwt_power = np.abs(xwt)
        
        # Phase difference
        phase_diff = np.angle(xwt)
        
        # Convert scales to periods (for Morlet wavelet)
        omega0 = 6.0
        fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))
        periods = scales * fourier_factor * dt
        
        # Cone of influence
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
                      wavelet: str = 'morl', sampling_rate: float = 1.0,
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
        wavelet: Wavelet type
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


def plot_cross_wavelet(results: Dict[str, Any], show_phase_arrows: bool = True) -> plt.Figure:
    """
    Plot Cross-Wavelet Transform with phase arrows.
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
    
    # Plot log of cross-wavelet power
    log_power = np.log2(xwt_power + 1e-10)
    im = ax.pcolormesh(time, periods, log_power, cmap='jet', shading='auto')
    
    ax.set_yscale('log')
    ax.set_ylim([periods[-1] * 1.1, periods[0] * 0.9])
    
    # COI
    if coi is not None and len(coi) == len(time):
        ax.fill_between(time, coi, periods[-1] * 2,
                        facecolor='white', alpha=0.3, hatch='///', edgecolor='gray')
        ax.plot(time, coi, 'k--', linewidth=2)
    
    # Phase arrows (subsampled for clarity)
    if show_phase_arrows and phase_diff is not None:
        step_t = max(1, len(time) // 20)
        step_s = max(1, len(periods) // 15)
        
        for i in range(0, len(periods), step_s):
            for j in range(0, len(time), step_t):
                # Arrow direction based on phase
                angle = phase_diff[i, j]
                dx = np.cos(angle) * 0.3
                dy = np.sin(angle) * 0.3
                
                # Only show arrows where coherence is significant
                if xwt_power[i, j] > np.percentile(xwt_power, 50):
                    ax.annotate('', xy=(time[j] + dx, periods[i]),
                               xytext=(time[j], periods[i]),
                               arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    ax.set_title(f'Cross-Wavelet Transform: {col1} vs {col2}', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log2(Power)', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_wavelet_coherence(results: Dict[str, Any], show_phase_arrows: bool = True) -> plt.Figure:
    """
    Plot Wavelet Coherence with phase arrows.
    
    Similar to Torrence & Compo style but for coherence between two signals.
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
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot coherence (0 to 1 scale)
    im = ax.pcolormesh(time, periods, wtc, cmap='hot_r', shading='auto', vmin=0, vmax=1)
    
    # Add significance contour (e.g., 0.5 threshold)
    ax.contour(time, periods, wtc, levels=[0.5], colors='black', linewidths=1.5)
    
    ax.set_yscale('log')
    ax.set_ylim([periods[-1] * 1.1, periods[0] * 0.9])
    
    # COI
    if coi is not None and len(coi) == len(time):
        ax.fill_between(time, coi, periods[-1] * 2,
                        facecolor='white', alpha=0.3, hatch='///', edgecolor='gray')
        ax.plot(time, coi, 'k--', linewidth=2)
    
    # Phase arrows
    if show_phase_arrows and phase_diff is not None:
        step_t = max(1, len(time) // 20)
        step_s = max(1, len(periods) // 15)
        
        for i in range(0, len(periods), step_s):
            for j in range(0, len(time), step_t):
                # Only show arrows where coherence is significant
                if wtc[i, j] > 0.5:
                    angle = phase_diff[i, j]
                    dx = np.cos(angle) * 0.4
                    dy = np.sin(angle) * 0.4
                    ax.annotate('', xy=(time[j] + dx, periods[i]),
                               xytext=(time[j], periods[i]),
                               arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    ax.set_title(f'Wavelet Coherence: {col1} vs {col2}\n(Arrows show phase relationship)', fontsize=14)
    
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
