"""Signal analysis helpers: Fourier and Wavelet routines.

These functions are designed to be called from `TimeSeriesAnalysis` as thin
wrappers so the public API remains stable while grouping signal-specific code
in a single module for easier maintenance and future additions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import chi2
from typing import Dict, Any
import warnings
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
