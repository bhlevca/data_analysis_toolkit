"""
Visualization Methods Module
Contains methods for various data visualizations including scatter plots,
heatmaps, FFT analysis, 3D plots, and more
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class VisualizationMethods:
    """Data visualization methods"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        # Set a nicer style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to visualize"""
        self.df = df
    
    def scatter_matrix(self, columns: List[str], figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """
        Create scatter matrix plot
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        data = self.df[columns].dropna()
        
        # Close any existing figures to prevent empty window
        plt.close('all')
        
        # scatter_matrix creates its own figure, so we don't create one beforehand
        axes = pd.plotting.scatter_matrix(data, figsize=figsize, diagonal='kde', alpha=0.6)
        
        # Get the figure from the axes
        fig = plt.gcf()
        fig.suptitle('Scatter Matrix', y=1.02)
        plt.tight_layout()
        return fig
    
    def correlation_heatmap(self, columns: List[str], method: str = 'pearson',
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create correlation heatmap
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        corr = self.df[columns].corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'{method.capitalize()} Correlation Heatmap')
        plt.tight_layout()
        
        return fig
    
    def box_plots(self, columns: List[str], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create box plots for multiple columns
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        self.df[columns].boxplot(ax=ax)
        ax.set_title('Box Plots')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def fft_analysis(self, column: str, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Perform and plot FFT spectrum analysis
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        data = self.df[column].dropna().values
        
        N = len(data)
        freqs = rfftfreq(N, 1)
        fft_vals = rfft(data)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(freqs, np.abs(fft_vals))
        ax.set_title(f'FFT Spectrum: {column}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def noise_filter(self, column: str, window_length: int = 51, 
                    polyorder: int = 3, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Apply and plot Savitzky-Golay noise filter
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        signal = self.df[column].dropna().values
        
        # Ensure window_length is odd and less than signal length
        if window_length >= len(signal):
            window_length = len(signal) - 1 if len(signal) % 2 == 0 else len(signal) - 2
        if window_length % 2 == 0:
            window_length += 1
        
        filtered = savgol_filter(signal, window_length, polyorder)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(signal, label='Original', alpha=0.7)
        ax.plot(filtered, label='Filtered (Savitzky-Golay)', linewidth=2)
        ax.legend()
        ax.set_title(f'Noise Filtering: {column}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def scatter_3d(self, columns: List[str], figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create 3D scatter plot (requires 3 columns)
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or len(columns) < 3:
            return None
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        data = self.df[columns].dropna()
        
        ax.scatter(data[columns[0]], data[columns[1]], data[columns[2]], alpha=0.6)
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])
        ax.set_title('3D Scatter Plot')
        
        return fig
    
    def line_plot(self, columns: List[str], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create line plot for time series data
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for col in columns:
            ax.plot(self.df[col], label=col, alpha=0.8)
        
        ax.set_title('Line Plot')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def scatter_plot(self, x_col: str, y_col: str, 
                    figsize: Tuple[int, int] = (10, 8),
                    add_regression: bool = True) -> plt.Figure:
        """
        Create scatter plot with optional regression line
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        data = self.df[[x_col, y_col]].dropna()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(data[x_col], data[y_col], alpha=0.6)
        
        if add_regression and len(data) > 1:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def histogram(self, column: str, bins: int = 30,
                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create histogram with KDE
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        data = self.df[column].dropna()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor='black')
        
        # Add KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend()
        except:
            pass
        
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {column}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def violin_plot(self, columns: List[str], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create violin plot
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        data = [self.df[col].dropna().values for col in columns]
        
        parts = ax.violinplot(data, positions=range(len(columns)), showmeans=True, showmedians=True)
        
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Violin Plot')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def pairwise_scatter(self, x_col: str, y_col: str, hue_col: str = None,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create scatter plot with optional coloring by third variable
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if hue_col and hue_col in self.df.columns:
            scatter = ax.scatter(self.df[x_col], self.df[y_col], 
                               c=self.df[hue_col], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label=hue_col)
        else:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}')
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def density_plot(self, columns: List[str], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create density plot (KDE) for multiple columns
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None or not columns:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for col in columns:
            data = self.df[col].dropna()
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_range, kde(x_range), label=col, linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.2)
            except:
                ax.hist(data, bins=30, density=True, alpha=0.5, label=col)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Density Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def quick_xy_plot(self, x_col: str, y_col: str) -> plt.Figure:
        """
        Create a quick X-Y scatter plot for data preview
        
        Returns:
            matplotlib Figure object
        """
        if self.df is None:
            return None
        
        data = self.df[[x_col, y_col]].dropna()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        ax.scatter(data[x_col], data[y_col], alpha=0.6, edgecolors='white', linewidth=0.5)
        
        # Add trend line
        if len(data) > 1:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
            
            # Calculate R²
            from scipy.stats import pearsonr
            r, _ = pearsonr(data[x_col], data[y_col])
            ax.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(f'{x_col} vs {y_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
