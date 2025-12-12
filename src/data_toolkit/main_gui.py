"""
Advanced Data Analysis Toolkit - Modern GUI
A comprehensive data analysis application with modular architecture

Author: Modernized version
Version: 7.0
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from .data_loading_methods import DataLoader
from .statistical_analysis import StatisticalAnalysis
from .ml_models import MLModels
from .bayesian_analysis import BayesianAnalysis
from .uncertainty_analysis import UncertaintyAnalysis
from .nonlinear_analysis import NonLinearAnalysis
from .timeseries_analysis import TimeSeriesAnalysis
from .causality_analysis import CausalityAnalysis
from .visualization_methods import VisualizationMethods
from .rust_accelerated import AccelerationSettings, is_rust_available, get_backend_name
from .plugin_system import PluginManager, get_plugin_template, get_example_plugins


# ============================================================================
# MODERN THEME CONFIGURATION
# ============================================================================
class ModernTheme:
    """Modern color theme and styling - Clean White Theme"""
    
    # Color palette - Clean white/light theme
    BG_DARK = "#ffffff"        # Main background - white
    BG_MEDIUM = "#f5f5f5"      # Cards/panels - light gray
    BG_LIGHT = "#e8e8e8"       # Buttons/selected - slightly darker gray
    ACCENT = "#2563eb"         # Blue accent (modern blue)
    ACCENT_HOVER = "#1d4ed8"   # Darker blue on hover
    TEXT_PRIMARY = "#1f2937"   # Dark gray text
    TEXT_SECONDARY = "#6b7280" # Medium gray text
    SUCCESS = "#10b981"        # Green
    WARNING = "#f59e0b"        # Orange/amber
    BORDER = "#d1d5db"         # Light border
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_LARGE = 14
    FONT_SIZE_MEDIUM = 11
    FONT_SIZE_SMALL = 10
    
    @classmethod
    def configure_style(cls, root):
        """Configure ttk styles for modern look"""
        style = ttk.Style(root)
        
        # Try to use clam theme as base (most customizable)
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Configure colors
        style.configure(".", 
                       background=cls.BG_DARK,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        
        # Frame styles
        style.configure("TFrame", background=cls.BG_DARK)
        style.configure("Card.TFrame", background=cls.BG_MEDIUM, relief="flat")
        
        # Label styles
        style.configure("TLabel", 
                       background=cls.BG_DARK, 
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        style.configure("Header.TLabel", 
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_LARGE, "bold"),
                       foreground=cls.ACCENT)
        style.configure("Info.TLabel",
                       foreground=cls.TEXT_SECONDARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL))
        
        # Button styles - white theme with blue accent
        style.configure("TButton",
                       background=cls.BG_LIGHT,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM),
                       padding=(15, 8),
                       borderwidth=1)
        style.map("TButton",
                 background=[("active", cls.ACCENT), ("pressed", cls.ACCENT_HOVER)],
                 foreground=[("active", "#ffffff")])
        
        style.configure("Accent.TButton",
                       background=cls.ACCENT,
                       foreground="#ffffff",
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, "bold"),
                       padding=(20, 10))
        style.map("Accent.TButton",
                 background=[("active", cls.ACCENT_HOVER)])
        
        # Notebook (tabs)
        style.configure("TNotebook", 
                       background=cls.BG_DARK,
                       borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=cls.BG_MEDIUM,
                       foreground=cls.TEXT_SECONDARY,
                       padding=(20, 10),
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        style.map("TNotebook.Tab",
                 background=[("selected", cls.ACCENT)],
                 foreground=[("selected", "#ffffff")])
        
        # LabelFrame
        style.configure("TLabelframe",
                       background=cls.BG_DARK,
                       foreground=cls.TEXT_PRIMARY,
                       bordercolor=cls.BORDER,
                       relief="solid",
                       borderwidth=1)
        style.configure("TLabelframe.Label",
                       background=cls.BG_DARK,
                       foreground=cls.ACCENT,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, "bold"))
        
        # Entry
        style.configure("TEntry",
                       fieldbackground="#ffffff",
                       foreground=cls.TEXT_PRIMARY,
                       bordercolor=cls.BORDER,
                       insertcolor=cls.TEXT_PRIMARY)
        
        # Combobox
        style.configure("TCombobox",
                       fieldbackground="#ffffff",
                       background=cls.BG_LIGHT,
                       foreground=cls.TEXT_PRIMARY,
                       arrowcolor=cls.ACCENT)
        
        # Treeview (for data table)
        style.configure("Treeview",
                       background="#ffffff",
                       foreground=cls.TEXT_PRIMARY,
                       fieldbackground="#ffffff",
                       borderwidth=1,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL))
        style.configure("Treeview.Heading",
                       background=cls.BG_LIGHT,
                       foreground=cls.ACCENT,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL, "bold"))
        style.map("Treeview",
                 background=[("selected", cls.ACCENT)],
                 foreground=[("selected", "#ffffff")])
        
        # Scrollbar
        style.configure("TScrollbar",
                       background=cls.BG_LIGHT,
                       troughcolor=cls.BG_MEDIUM,
                       bordercolor=cls.BORDER,
                       arrowcolor=cls.TEXT_SECONDARY)
        
        # Checkbutton
        style.configure("TCheckbutton",
                       background=cls.BG_DARK,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        style.map("TCheckbutton",
                 background=[("active", cls.BG_DARK)],
                 foreground=[("disabled", cls.TEXT_SECONDARY)])
        
        return style


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================
class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    
    def __init__(self, parent, text, command=None, width=120, height=36, 
                 bg=ModernTheme.BG_LIGHT, hover_bg=ModernTheme.ACCENT,
                 fg=ModernTheme.TEXT_PRIMARY, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=ModernTheme.BG_DARK, highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.bg = bg
        self.hover_bg = hover_bg
        self.fg = fg
        self.is_hovered = False
        
        self._draw_button()
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
    
    def _draw_button(self):
        self.delete("all")
        color = self.hover_bg if self.is_hovered else self.bg
        
        # Draw rounded rectangle
        radius = 8
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, radius, fill=color, outline="")
        
        # Draw text
        self.create_text(self.width//2, self.height//2, text=self.text, 
                        fill=self.fg, font=(ModernTheme.FONT_FAMILY, 10))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_enter(self, event):
        self.is_hovered = True
        self._draw_button()
        self.config(cursor="hand2")
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw_button()
    
    def _on_click(self, event):
        if self.command:
            self.command()


class DataPreviewTable(ttk.Frame):
    """Modern data preview table with scrolling"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create treeview with scrollbars
        self.tree = ttk.Treeview(self, show="headings", selectmode="browse")
        
        # Scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
    
    def load_data(self, df: pd.DataFrame, max_rows: int = 100):
        """Load DataFrame into the table"""
        # Clear existing data
        self.tree.delete(*self.tree.get_children())
        
        if df is None or df.empty:
            return
        
        # Configure columns
        columns = list(df.columns)
        self.tree["columns"] = columns
        
        for col in columns:
            self.tree.heading(col, text=col)
            # Calculate column width based on content
            max_width = max(len(str(col)), 
                          df[col].astype(str).str.len().max() if len(df) > 0 else 10)
            width = min(max(max_width * 8, 80), 200)
            self.tree.column(col, width=width, minwidth=50)
        
        # Insert rows
        for idx, row in df.head(max_rows).iterrows():
            values = [str(v) if pd.notna(v) else "" for v in row]
            self.tree.insert("", "end", values=values)


class FileInfoPanel(ttk.Frame):
    """Panel showing loaded file information"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        
        self.file_name_var = tk.StringVar(value="No file loaded")
        self.rows_var = tk.StringVar(value="Rows: -")
        self.cols_var = tk.StringVar(value="Columns: -")
        self.memory_var = tk.StringVar(value="Memory: -")
        
        # File icon and name
        self.name_label = ttk.Label(self, textvariable=self.file_name_var,
                                   style="Header.TLabel")
        self.name_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Info row
        info_frame = ttk.Frame(self, style="Card.TFrame")
        info_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ttk.Label(info_frame, textvariable=self.rows_var, 
                 style="Info.TLabel").pack(side="left", padx=(0, 20))
        ttk.Label(info_frame, textvariable=self.cols_var,
                 style="Info.TLabel").pack(side="left", padx=(0, 20))
        ttk.Label(info_frame, textvariable=self.memory_var,
                 style="Info.TLabel").pack(side="left")
    
    def update_info(self, file_name: str, rows: int, cols: int, memory_bytes: int):
        """Update the displayed file information"""
        self.file_name_var.set(f"📄 {file_name}")
        self.rows_var.set(f"Rows: {rows:,}")
        self.cols_var.set(f"Columns: {cols}")
        
        # Format memory
        if memory_bytes < 1024:
            mem_str = f"{memory_bytes} B"
        elif memory_bytes < 1024**2:
            mem_str = f"{memory_bytes/1024:.1f} KB"
        else:
            mem_str = f"{memory_bytes/1024**2:.1f} MB"
        self.memory_var.set(f"Memory: {mem_str}")
    
    def clear_info(self):
        """Clear the displayed information"""
        self.file_name_var.set("No file loaded")
        self.rows_var.set("Rows: -")
        self.cols_var.set("Columns: -")
        self.memory_var.set("Memory: -")


class PlotPanel(ttk.Frame):
    """Panel for displaying matplotlib plots"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor=ModernTheme.BG_DARK)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
    
    def _style_axes(self):
        """Apply theme styling to axes"""
        self.ax.set_facecolor(ModernTheme.BG_MEDIUM)
        self.ax.tick_params(colors=ModernTheme.TEXT_SECONDARY)
        self.ax.spines['bottom'].set_color(ModernTheme.BORDER)
        self.ax.spines['top'].set_color(ModernTheme.BORDER)
        self.ax.spines['left'].set_color(ModernTheme.BORDER)
        self.ax.spines['right'].set_color(ModernTheme.BORDER)
        self.ax.xaxis.label.set_color(ModernTheme.TEXT_PRIMARY)
        self.ax.yaxis.label.set_color(ModernTheme.TEXT_PRIMARY)
        self.ax.title.set_color(ModernTheme.TEXT_PRIMARY)
    
    def plot_xy(self, x_data, y_data, x_label: str, y_label: str, title: str = ""):
        """Create X-Y scatter plot with regression line"""
        self.ax.clear()
        self._style_axes()
        
        # Scatter plot
        self.ax.scatter(x_data, y_data, alpha=0.6, c=ModernTheme.ACCENT, 
                       edgecolors='white', linewidth=0.5, s=50)
        
        # Regression line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            self.ax.plot(x_line, p(x_line), '--', color=ModernTheme.SUCCESS, 
                        linewidth=2, alpha=0.8)
            
            # R² value
            from scipy.stats import pearsonr
            r, _ = pearsonr(x_data, y_data)
            self.ax.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top', color=ModernTheme.TEXT_PRIMARY,
                        bbox=dict(boxstyle='round', facecolor=ModernTheme.BG_LIGHT, alpha=0.8))
        
        self.ax.set_xlabel(x_label, fontsize=11, color=ModernTheme.TEXT_PRIMARY)
        self.ax.set_ylabel(y_label, fontsize=11, color=ModernTheme.TEXT_PRIMARY)
        if title:
            self.ax.set_title(title, fontsize=12, fontweight='bold', color=ModernTheme.TEXT_PRIMARY)
        
        self.ax.grid(True, alpha=0.2, color=ModernTheme.TEXT_SECONDARY)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def clear(self):
        """Clear the plot"""
        self.ax.clear()
        self._style_axes()
        self.canvas.draw()
    
    def show_figure(self, fig):
        """Display an external matplotlib figure"""
        # Clear current figure
        self.figure.clear()
        
        # Copy axes from external figure
        for ax in fig.axes:
            new_ax = self.figure.add_subplot(111)
            # Copy the plot
            for line in ax.lines:
                new_ax.plot(line.get_xdata(), line.get_ydata())
        
        self.canvas.draw()


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
class AdvancedDataAnalysisGUI:
    """Main application class with modern GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Analysis Toolkit v7.0")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Configure root background
        self.root.configure(bg=ModernTheme.BG_DARK)
        
        # Apply modern theme
        self.style = ModernTheme.configure_style(root)
        
        # Initialize analysis modules
        self.data_loader = DataLoader()
        self.stats_analysis = StatisticalAnalysis()
        self.ml_models = MLModels()
        self.bayesian = BayesianAnalysis()
        self.uncertainty = UncertaintyAnalysis()
        self.nonlinear = NonLinearAnalysis()
        self.timeseries = TimeSeriesAnalysis()
        self.causality = CausalityAnalysis()
        self.visualization = VisualizationMethods()
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager()
        
        # Variables
        self.target_var = tk.StringVar()
        self.model_var = tk.StringVar(value="Linear Regression")
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.bootstrap_samples = tk.IntVar(value=1000)
        self.max_lag_var = tk.IntVar(value=10)
        
        # Build GUI
        self._create_main_layout()
        self._create_tabs()
        
    def _create_main_layout(self):
        """Create the main application layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="📊 Advanced Data Analysis Toolkit",
                               style="Header.TLabel",
                               font=(ModernTheme.FONT_FAMILY, 18, "bold"))
        title_label.pack(side="left")
        
        # Backend toggle section (right side, before version)
        backend_frame = ttk.Frame(header_frame)
        backend_frame.pack(side="right", padx=(0, 20))
        
        # Backend status indicator
        self.backend_var = tk.BooleanVar(value=AccelerationSettings._use_rust)
        self.rust_available = is_rust_available()
        
        # Status label
        self.backend_label = ttk.Label(backend_frame, text="", style="Info.TLabel")
        self.backend_label.pack(side="left", padx=(0, 10))
        
        # Toggle checkbox
        self.backend_toggle = ttk.Checkbutton(
            backend_frame, 
            text="🦀 Rust Acceleration",
            variable=self.backend_var,
            command=self._toggle_backend,
            style="TCheckbutton"
        )
        self.backend_toggle.pack(side="left")
        
        # Disable toggle if Rust not available
        if not self.rust_available:
            self.backend_toggle.configure(state="disabled")
        
        # Update status display
        self._update_backend_status()
        
        # Version info
        version_label = ttk.Label(header_frame, text="v7.0 - Unified Edition",
                                 style="Info.TLabel")
        version_label.pack(side="right")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)
    
    def _create_tabs(self):
        """Create all application tabs"""
        self._create_data_tab()
        self._create_analysis_tab()
        self._create_ml_tab()
        self._create_bayesian_tab()
        self._create_uncertainty_tab()
        self._create_nonlinear_tab()
        self._create_timeseries_tab()
        self._create_causality_tab()
        self._create_visualization_tab()
        self._create_plugins_tab()
    
    def _create_data_tab(self):
        """Tab for data loading and preview"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 Data Loading")
        
        # Top section - File loading
        top_frame = ttk.Frame(tab)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        # Load button
        load_btn = ttk.Button(top_frame, text="📂 Load CSV File", 
                             style="Accent.TButton", command=self.pick_file)
        load_btn.pack(side="left", padx=(0, 20))
        
        # File info panel
        self.file_info_panel = FileInfoPanel(top_frame)
        self.file_info_panel.pack(side="left", fill="x", expand=True)
        
        # Middle section - Split view
        middle_frame = ttk.Frame(tab)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Column selection
        left_frame = ttk.LabelFrame(middle_frame, text="Column Selection")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Feature columns list
        ttk.Label(left_frame, text="Feature Columns (Ctrl+Click for multi-select):").pack(anchor="w", padx=5, pady=5)
        
        feature_frame = ttk.Frame(left_frame)
        feature_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.feature_list = tk.Listbox(feature_frame, selectmode="extended", 
                                       bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                       selectbackground=ModernTheme.ACCENT,
                                       font=(ModernTheme.FONT_FAMILY, 10),
                                       borderwidth=0, highlightthickness=1,
                                       highlightcolor=ModernTheme.ACCENT)
        feature_scroll = ttk.Scrollbar(feature_frame, orient="vertical", command=self.feature_list.yview)
        self.feature_list.configure(yscrollcommand=feature_scroll.set)
        self.feature_list.pack(side="left", fill="both", expand=True)
        feature_scroll.pack(side="right", fill="y")
        
        # Target column
        target_frame = ttk.Frame(left_frame)
        target_frame.pack(fill="x", padx=5, pady=10)
        ttk.Label(target_frame, text="Target Column:").pack(anchor="w")
        self.target_dropdown = ttk.Combobox(target_frame, textvariable=self.target_var, 
                                           state="readonly", width=30)
        self.target_dropdown.pack(fill="x", pady=5)
        
        # Right side - Data preview and plot
        right_frame = ttk.Frame(middle_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Data preview table
        preview_frame = ttk.LabelFrame(right_frame, text="Data Preview (First 100 rows)")
        preview_frame.pack(fill="both", expand=True, pady=(0, 5))
        
        self.data_table = DataPreviewTable(preview_frame)
        self.data_table.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Quick X-Y plot
        plot_frame = ttk.LabelFrame(right_frame, text="Quick X→Y Plot")
        plot_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        # Plot controls
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(plot_controls, text="X:").pack(side="left")
        self.x_col_var = tk.StringVar()
        self.x_dropdown = ttk.Combobox(plot_controls, textvariable=self.x_col_var, 
                                       state="readonly", width=15)
        self.x_dropdown.pack(side="left", padx=5)
        
        ttk.Label(plot_controls, text="Y:").pack(side="left")
        self.y_col_var = tk.StringVar()
        self.y_dropdown = ttk.Combobox(plot_controls, textvariable=self.y_col_var,
                                       state="readonly", width=15)
        self.y_dropdown.pack(side="left", padx=5)
        
        ttk.Button(plot_controls, text="Plot", command=self.quick_plot).pack(side="left", padx=10)
        ttk.Button(plot_controls, text="Open in Window", 
                  command=self.open_plot_window).pack(side="left")
        
        # Embedded plot
        self.preview_plot = PlotPanel(plot_frame)
        self.preview_plot.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_analysis_tab(self):
        """Tab for statistical analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📈 Statistical Analysis")
        
        # Button frame
        btn_frame = ttk.LabelFrame(tab, text="Analysis Tools")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Descriptive Stats", self.descriptive_stats),
            ("Correlation Matrix", self.correlation_matrix),
            ("Cross-Correlation", self.cross_correlation),
            ("Lag Analysis", self.lag_analysis),
            ("Distribution Analysis", self.distribution_analysis),
            ("Outlier Detection", self.outlier_detection),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
        
        # Results area
        results_frame = ttk.LabelFrame(tab, text="Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, wrap="word", 
                                   bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                   font=(ModernTheme.FONT_FAMILY, 10),
                                   borderwidth=0, padx=10, pady=10)
        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", 
                                      command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scroll.pack(side="right", fill="y")
    
    def _create_ml_tab(self):
        """Tab for machine learning"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🤖 Machine Learning")
        
        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Model Selection")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        models = self.ml_models.get_available_models()
        
        ttk.Label(model_frame, text="Select Model:").pack(side="left", padx=10, pady=10)
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var,
                                     values=models, state="readonly", width=30)
        model_dropdown.pack(side="left", padx=5)
        
        ttk.Button(model_frame, text="Train Model", command=self.train_model).pack(side="left", padx=10)
        ttk.Button(model_frame, text="Cross-Validation", command=self.cross_validation).pack(side="left", padx=5)
        
        # Dimensionality reduction
        dim_frame = ttk.LabelFrame(tab, text="Dimensionality Reduction")
        dim_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(dim_frame, text="PCA Analysis", command=self.pca_analysis).pack(side="left", padx=10, pady=10)
        ttk.Button(dim_frame, text="ICA Analysis", command=self.ica_analysis).pack(side="left", padx=5)
        ttk.Button(dim_frame, text="Feature Importance", command=self.feature_importance).pack(side="left", padx=5)
        
        # Results
        results_frame = ttk.LabelFrame(tab, text="Model Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.ml_results_text = tk.Text(results_frame, wrap="word",
                                      bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                      font=(ModernTheme.FONT_FAMILY, 10),
                                      borderwidth=0, padx=10, pady=10)
        ml_scroll = ttk.Scrollbar(results_frame, orient="vertical",
                                 command=self.ml_results_text.yview)
        self.ml_results_text.configure(yscrollcommand=ml_scroll.set)
        self.ml_results_text.pack(side="left", fill="both", expand=True)
        ml_scroll.pack(side="right", fill="y")
    
    def _create_bayesian_tab(self):
        """Tab for Bayesian analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎲 Bayesian")
        
        btn_frame = ttk.LabelFrame(tab, text="Bayesian Analysis Tools")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Bayesian Linear Regression", self.bayesian_regression),
            ("Credible Intervals", self.credible_intervals),
            ("Posterior Distributions", self.posterior_distributions),
            ("Bayesian Model Comparison", self.bayesian_model_comparison),
            ("Prior Sensitivity", self.prior_sensitivity),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
        
        results_frame = ttk.LabelFrame(tab, text="Bayesian Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.bayesian_results_text = tk.Text(results_frame, wrap="word",
                                            bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                            font=(ModernTheme.FONT_FAMILY, 10),
                                            borderwidth=0, padx=10, pady=10)
        scroll = ttk.Scrollbar(results_frame, orient="vertical",
                              command=self.bayesian_results_text.yview)
        self.bayesian_results_text.configure(yscrollcommand=scroll.set)
        self.bayesian_results_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
    
    def _create_uncertainty_tab(self):
        """Tab for uncertainty analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Uncertainty")
        
        btn_frame = ttk.LabelFrame(tab, text="Uncertainty Quantification")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Bootstrap CI", self.bootstrap_ci),
            ("Prediction Intervals", self.prediction_intervals),
            ("Confidence Bands", self.confidence_bands),
            ("Error Propagation", self.error_propagation),
            ("Monte Carlo Simulation", self.monte_carlo_analysis),
            ("Residual Analysis", self.residual_analysis),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
        
        # Parameters
        param_frame = ttk.LabelFrame(tab, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(param_frame, text="Confidence Level:").pack(side="left", padx=10, pady=10)
        ttk.Entry(param_frame, textvariable=self.confidence_level, width=10).pack(side="left", padx=5)
        
        ttk.Label(param_frame, text="Bootstrap Samples:").pack(side="left", padx=10)
        ttk.Entry(param_frame, textvariable=self.bootstrap_samples, width=10).pack(side="left", padx=5)
        
        results_frame = ttk.LabelFrame(tab, text="Uncertainty Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.uncertainty_results_text = tk.Text(results_frame, wrap="word",
                                               bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                               font=(ModernTheme.FONT_FAMILY, 10),
                                               borderwidth=0, padx=10, pady=10)
        scroll = ttk.Scrollbar(results_frame, orient="vertical",
                              command=self.uncertainty_results_text.yview)
        self.uncertainty_results_text.configure(yscrollcommand=scroll.set)
        self.uncertainty_results_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
    
    def _create_nonlinear_tab(self):
        """Tab for non-linear analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🌀 Non-Linear")
        
        btn_frame = ttk.LabelFrame(tab, text="Non-Linear Analysis")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Mutual Information", self.mutual_information),
            ("Distance Correlation", self.distance_correlation),
            ("Maximal Info Coefficient", self.maximal_information),
            ("Gaussian Process", self.gaussian_process),
            ("Polynomial Regression", self.polynomial_regression),
            ("Spline Regression", self.spline_regression),
            ("Neural Network", self.neural_network_regression),
            ("SVM Regression", self.svm_regression),
            ("Transfer Entropy", self.transfer_entropy),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
        
        results_frame = ttk.LabelFrame(tab, text="Non-Linear Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.nonlinear_results_text = tk.Text(results_frame, wrap="word",
                                             bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                             font=(ModernTheme.FONT_FAMILY, 10),
                                             borderwidth=0, padx=10, pady=10)
        scroll = ttk.Scrollbar(results_frame, orient="vertical",
                              command=self.nonlinear_results_text.yview)
        self.nonlinear_results_text.configure(yscrollcommand=scroll.set)
        self.nonlinear_results_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
    
    def _create_timeseries_tab(self):
        """Tab for time series analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⏱️ Time Series")
        
        btn_frame = ttk.LabelFrame(tab, text="Time Series Analysis")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("ACF/PACF", self.acf_pacf_analysis),
            ("Stationarity Test", self.stationarity_test),
            ("ARIMA Model", self.arima_model),
            ("Decomposition", self.time_decomposition),
            ("Rolling Statistics", self.rolling_stats),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=0, column=i, padx=10, pady=8, sticky="ew")
        
        for i in range(5):
            btn_frame.columnconfigure(i, weight=1)
        
        results_frame = ttk.LabelFrame(tab, text="Time Series Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.ts_results_text = tk.Text(results_frame, wrap="word",
                                      bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                      font=(ModernTheme.FONT_FAMILY, 10),
                                      borderwidth=0, padx=10, pady=10)
        scroll = ttk.Scrollbar(results_frame, orient="vertical",
                              command=self.ts_results_text.yview)
        self.ts_results_text.configure(yscrollcommand=scroll.set)
        self.ts_results_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
    
    def _create_causality_tab(self):
        """Tab for causality analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔗 Causality")
        
        btn_frame = ttk.LabelFrame(tab, text="Causality Analysis")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Granger Causality", self.granger_causality),
            ("Lead-Lag Analysis", self.lead_lag_analysis),
            ("Correlation at Different Lags", self.correlation_lags),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=0, column=i, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
        
        # Lag parameters
        param_frame = ttk.LabelFrame(tab, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(param_frame, text="Max Lag:").pack(side="left", padx=10, pady=10)
        ttk.Entry(param_frame, textvariable=self.max_lag_var, width=10).pack(side="left", padx=5)
        
        results_frame = ttk.LabelFrame(tab, text="Causality Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.causality_results_text = tk.Text(results_frame, wrap="word",
                                             bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.TEXT_PRIMARY,
                                             font=(ModernTheme.FONT_FAMILY, 10),
                                             borderwidth=0, padx=10, pady=10)
        scroll = ttk.Scrollbar(results_frame, orient="vertical",
                              command=self.causality_results_text.yview)
        self.causality_results_text.configure(yscrollcommand=scroll.set)
        self.causality_results_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
    
    def _create_visualization_tab(self):
        """Tab for visualizations"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Visualizations")
        
        btn_frame = ttk.LabelFrame(tab, text="Visualization Tools")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("Scatter Matrix", self.scatter_matrix),
            ("Heatmap", self.correlation_heatmap),
            ("Box Plots", self.box_plots),
            ("FFT Spectrum", self.fft_analysis),
            ("Noise Filtering", self.noise_filter),
            ("3D Scatter", self.scatter_3d),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(btn_frame, text=text, command=cmd)
            btn.grid(row=i//3, column=i%3, padx=10, pady=8, sticky="ew")
        
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_selected_columns(self):
        """Get selected feature and target columns"""
        if self.data_loader.df is None:
            messagebox.showerror("Error", "Load a CSV file first!")
            return None, None
        
        feature_indices = self.feature_list.curselection()
        if not feature_indices:
            messagebox.showerror("Error", "Select at least one feature column.")
            return None, None
        
        features = [self.feature_list.get(i) for i in feature_indices]
        target = self.target_var.get()
        
        return features, target
    
    def _update_all_modules(self):
        """Update all analysis modules with current data"""
        df = self.data_loader.df
        if df is not None:
            self.stats_analysis.set_data(df)
            self.ml_models.set_data(df)
            self.bayesian.set_data(df)
            self.uncertainty.set_data(df)
            self.nonlinear.set_data(df)
            self.timeseries.set_data(df)
            self.causality.set_data(df)
            self.visualization.set_data(df)
    
    # ========================================================================
    # DATA LOADING METHODS
    # ========================================================================
    
    def pick_file(self):
        """Load a CSV file"""
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        if not path:
            return
        
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                success, msg = self.data_loader.load_excel(path)
            else:
                success, msg = self.data_loader.load_csv(path)
            
            if success:
                # Update all modules
                self._update_all_modules()
                
                # Update UI
                info = self.data_loader.get_data_info()
                self.file_info_panel.update_info(
                    info['file_name'],
                    info['rows'],
                    info['columns'],
                    info['memory_usage']
                )
                
                # Update column lists
                cols = self.data_loader.get_columns()
                numeric_cols = self.data_loader.get_numeric_columns()
                
                self.feature_list.delete(0, tk.END)
                for c in cols:
                    self.feature_list.insert(tk.END, c)
                
                self.target_dropdown["values"] = cols
                self.x_dropdown["values"] = numeric_cols
                self.y_dropdown["values"] = numeric_cols
                
                if cols:
                    self.target_dropdown.current(0)
                if len(numeric_cols) >= 2:
                    self.x_dropdown.current(0)
                    self.y_dropdown.current(1)
                    # Auto-plot first two columns
                    self.quick_plot()
                
                # Update data preview table
                self.data_table.load_data(self.data_loader.get_preview(100))
                
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def quick_plot(self):
        """Create quick X-Y plot"""
        if self.data_loader.df is None:
            return
        
        x_col = self.x_col_var.get()
        y_col = self.y_col_var.get()
        
        if not x_col or not y_col:
            return
        
        df = self.data_loader.df
        data = df[[x_col, y_col]].dropna()
        
        self.preview_plot.plot_xy(data[x_col].values, data[y_col].values,
                                  x_col, y_col, f"{x_col} vs {y_col}")
    
    def open_plot_window(self):
        """Open plot in a separate window"""
        if self.data_loader.df is None:
            return
        
        x_col = self.x_col_var.get()
        y_col = self.y_col_var.get()
        
        if not x_col or not y_col:
            return
        
        fig = self.visualization.scatter_plot(x_col, y_col)
        if fig:
            plt.show()
    
    # ========================================================================
    # STATISTICAL ANALYSIS METHODS
    # ========================================================================
    
    def descriptive_stats(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        stats = self.stats_analysis.descriptive_stats(features)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "═══ DESCRIPTIVE STATISTICS ═══\n\n")
        self.results_text.insert(tk.END, stats.to_string())
    
    def correlation_matrix(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.stats_analysis.plot_correlation_matrices(features)
        if fig:
            plt.show()
    
    def cross_correlation(self):
        features, _ = self.get_selected_columns()
        if features is None or len(features) < 2:
            messagebox.showerror("Error", "Select at least 2 features")
            return
        
        fig = self.stats_analysis.plot_cross_correlation(features[0], features[1])
        if fig:
            plt.show()
    
    def lag_analysis(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.stats_analysis.lag_analysis(features, target, self.max_lag_var.get())
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "═══ LAG CORRELATION ANALYSIS ═══\n\n")
        
        for feature, lag_data in results.items():
            self.results_text.insert(tk.END, f"\n{feature} vs {target}:\n")
            for item in lag_data:
                self.results_text.insert(tk.END, f"  Lag {item['lag']}: {item['correlation']:.4f}\n")
    
    def distribution_analysis(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.stats_analysis.plot_distributions(features)
        if fig:
            plt.show()
    
    def outlier_detection(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.stats_analysis.outlier_detection(features)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "═══ OUTLIER DETECTION (IQR Method) ═══\n\n")
        
        for col, data in results.items():
            status = '⚠️' if data['n_outliers'] > 0 else '✅'
            self.results_text.insert(tk.END, 
                f"{status} {col}: {data['n_outliers']} outliers ({data['percentage']:.2f}%)\n")
            if data['n_outliers'] > 0 and 'lower_bound' in data:
                self.results_text.insert(tk.END,
                    f"   Bounds: [{data['lower_bound']:.3f}, {data['upper_bound']:.3f}]\n")
        
        # Pass outlier results to plot for accurate visualization
        fig = self.stats_analysis.plot_boxplots(features, outlier_results=results)
        if fig:
            plt.show()
    
    # ========================================================================
    # MACHINE LEARNING METHODS
    # ========================================================================
    
    def train_model(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        model_name = self.model_var.get()
        results = self.ml_models.train_model(features, target, model_name)
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, f"═══ {model_name.upper()} RESULTS ═══\n\n")
        
        if 'error' in results:
            self.ml_results_text.insert(tk.END, f"Error: {results['error']}\n")
            return
        
        if 'classification_report' in results:
            self.ml_results_text.insert(tk.END, results['classification_report'])
        elif 'cluster_sizes' in results:
            self.ml_results_text.insert(tk.END, f"Number of clusters: {results['n_clusters']}\n\n")
            for cluster, size in results['cluster_sizes'].items():
                self.ml_results_text.insert(tk.END, f"Cluster {cluster}: {size} samples\n")
            
            fig = self.ml_models.plot_clusters()
            if fig:
                plt.show()
        else:
            self.ml_results_text.insert(tk.END, f"MSE: {results['mse']:.4f}\n")
            self.ml_results_text.insert(tk.END, f"RMSE: {results['rmse']:.4f}\n")
            self.ml_results_text.insert(tk.END, f"R² Score: {results['r2']:.4f}\n\n")
            
            if 'coefficients' in results:
                self.ml_results_text.insert(tk.END, "Coefficients:\n")
                for feat, coef in results['coefficients'].items():
                    self.ml_results_text.insert(tk.END, f"  {feat}: {coef:.4f}\n")
            
            fig = self.ml_models.plot_predictions_vs_actual()
            if fig:
                plt.show()
    
    def cross_validation(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.ml_models.cross_validation(features, target)
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, "═══ CROSS-VALIDATION RESULTS ═══\n\n")
        self.ml_results_text.insert(tk.END, f"5-Fold CV R² Scores: {results['scores']}\n")
        self.ml_results_text.insert(tk.END, f"Mean: {results['mean']:.4f}\n")
        self.ml_results_text.insert(tk.END, f"Std Dev: {results['std']:.4f}\n")
    
    def pca_analysis(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.ml_models.pca_analysis(features)
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, "═══ PCA ANALYSIS ═══\n\n")
        
        for i, var in enumerate(results['explained_variance'][:5], 1):
            self.ml_results_text.insert(tk.END, f"PC{i}: {var*100:.2f}% variance\n")
        
        self.ml_results_text.insert(tk.END, f"\nComponents for 95% variance: {results['n_components_95']}\n")
        
        fig = self.ml_models.plot_pca_results(results)
        if fig:
            plt.show()
    
    def ica_analysis(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.ml_models.ica_analysis(features)
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, "═══ ICA ANALYSIS ═══\n\n")
        self.ml_results_text.insert(tk.END, f"Number of components: {results['n_components']}\n")
        
        # Plot components
        components = results['components']
        fig, axes = plt.subplots(1, components.shape[1], figsize=(12, 4))
        if components.shape[1] == 1:
            axes = [axes]
        
        for i in range(components.shape[1]):
            axes[i].plot(components[:, i])
            axes[i].set_title(f'IC {i+1}')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        importances = self.ml_models.feature_importance(features, target)
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, "═══ FEATURE IMPORTANCE ═══\n\n")
        
        for feat, imp in importances.items():
            self.ml_results_text.insert(tk.END, f"{feat}: {imp:.4f}\n")
        
        fig = self.ml_models.plot_feature_importance(importances)
        if fig:
            plt.show()
    
    # ========================================================================
    # BAYESIAN ANALYSIS METHODS
    # ========================================================================
    
    def bayesian_regression(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.bayesian.bayesian_regression(features, target)
        
        self.bayesian_results_text.delete(1.0, tk.END)
        self.bayesian_results_text.insert(tk.END, "═══ BAYESIAN LINEAR REGRESSION ═══\n\n")
        
        self.bayesian_results_text.insert(tk.END, "Posterior Mean (coefficients):\n")
        for i, feat in enumerate(results['features']):
            self.bayesian_results_text.insert(tk.END, f"  {feat}: {results['posterior_mean'][i]:.4f}\n")
        
        self.bayesian_results_text.insert(tk.END, "\n95% Credible Intervals:\n")
        for i, feat in enumerate(results['features']):
            self.bayesian_results_text.insert(tk.END,
                f"  {feat}: [{results['credible_intervals_lower'][i]:.4f}, "
                f"{results['credible_intervals_upper'][i]:.4f}]\n")
        
        fig = self.bayesian.plot_posterior_distributions(results)
        if fig:
            plt.show()
    
    def credible_intervals(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.bayesian.credible_intervals(features, target, self.confidence_level.get())
        
        self.bayesian_results_text.delete(1.0, tk.END)
        self.bayesian_results_text.insert(tk.END, "═══ CREDIBLE INTERVALS ═══\n\n")
        self.bayesian_results_text.insert(tk.END, f"95% CI Coverage: {results['coverage']*100:.2f}%\n")
        self.bayesian_results_text.insert(tk.END, f"Mean CI Width: {results['mean_ci_width']:.4f}\n")
        
        fig = self.bayesian.plot_credible_intervals(results)
        if fig:
            plt.show()
    
    def posterior_distributions(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.bayesian.posterior_distributions(features, target)
        
        self.bayesian_results_text.delete(1.0, tk.END)
        self.bayesian_results_text.insert(tk.END, "═══ POSTERIOR DISTRIBUTIONS ═══\n\n")
        self.bayesian_results_text.insert(tk.END, f"Alpha (precision): {results['alpha']:.4f}\n")
        self.bayesian_results_text.insert(tk.END, f"Lambda (regularization): {results['lambda']:.4f}\n\n")
        self.bayesian_results_text.insert(tk.END, "Coefficients:\n")
        for feat, coef in results['coefficients'].items():
            self.bayesian_results_text.insert(tk.END, f"  {feat}: {coef:.4f}\n")
    
    def bayesian_model_comparison(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.bayesian.bayesian_model_comparison(features, target)
        
        self.bayesian_results_text.delete(1.0, tk.END)
        self.bayesian_results_text.insert(tk.END, "═══ BAYESIAN MODEL COMPARISON (BIC) ═══\n\n")
        self.bayesian_results_text.insert(tk.END, "Top 5 Models (lower BIC is better):\n\n")
        
        for i, model in enumerate(results[:5], 1):
            self.bayesian_results_text.insert(tk.END, f"Model {i}:\n")
            self.bayesian_results_text.insert(tk.END, f"  BIC: {model['bic']:.2f}\n")
            self.bayesian_results_text.insert(tk.END, f"  R²: {model['r2']:.4f}\n")
            self.bayesian_results_text.insert(tk.END, f"  Features: {', '.join(model['features'])}\n\n")
    
    def prior_sensitivity(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.bayesian.prior_sensitivity(features, target)
        
        self.bayesian_results_text.delete(1.0, tk.END)
        self.bayesian_results_text.insert(tk.END, "═══ PRIOR SENSITIVITY ANALYSIS ═══\n\n")
        
        for r in results[:10]:
            self.bayesian_results_text.insert(tk.END,
                f"α={r['alpha_init']:.2e}, λ={r['lambda_init']:.2e}: R²={r['r2']:.4f}\n")
    
    # ========================================================================
    # UNCERTAINTY ANALYSIS METHODS
    # ========================================================================
    
    def bootstrap_ci(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.uncertainty.bootstrap_ci(features, target,
                                               self.bootstrap_samples.get(),
                                               self.confidence_level.get())
        
        self.uncertainty_results_text.delete(1.0, tk.END)
        conf = self.confidence_level.get() * 100
        self.uncertainty_results_text.insert(tk.END, f"═══ BOOTSTRAP CONFIDENCE INTERVALS ({conf:.0f}%) ═══\n\n")
        
        for i, feat in enumerate(results['features']):
            self.uncertainty_results_text.insert(tk.END, f"{feat}:\n")
            self.uncertainty_results_text.insert(tk.END, f"  Mean: {results['mean_coefs'][i]:.4f}\n")
            self.uncertainty_results_text.insert(tk.END, f"  Std: {results['std_coefs'][i]:.4f}\n")
            self.uncertainty_results_text.insert(tk.END,
                f"  CI: [{results['ci_lower'][i]:.4f}, {results['ci_upper'][i]:.4f}]\n\n")
        
        fig = self.uncertainty.plot_bootstrap_distributions(results)
        if fig:
            plt.show()
    
    def prediction_intervals(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.uncertainty.prediction_intervals(features, target, self.confidence_level.get())
        
        self.uncertainty_results_text.delete(1.0, tk.END)
        conf = self.confidence_level.get() * 100
        self.uncertainty_results_text.insert(tk.END, f"═══ PREDICTION INTERVALS ({conf:.0f}%) ═══\n\n")
        self.uncertainty_results_text.insert(tk.END, f"Residual Standard Error: {results['rse']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"Prediction Interval Width: {results['pi_width']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"Empirical Coverage: {results['coverage']*100:.2f}%\n")
        
        fig = self.uncertainty.plot_prediction_intervals(results)
        if fig:
            plt.show()
    
    def confidence_bands(self):
        features, target = self.get_selected_columns()
        if features is None or len(features) != 1:
            messagebox.showerror("Error", "Select exactly 1 feature for confidence bands")
            return
        
        results = self.uncertainty.confidence_bands(features[0], target, self.confidence_level.get())
        
        fig = self.uncertainty.plot_confidence_bands(results)
        if fig:
            plt.show()
    
    def error_propagation(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.uncertainty.error_propagation(features, target)
        
        self.uncertainty_results_text.delete(1.0, tk.END)
        self.uncertainty_results_text.insert(tk.END, "═══ ERROR PROPAGATION ═══\n\n")
        self.uncertainty_results_text.insert(tk.END, "Input Uncertainties:\n")
        for feat, unc in results['input_uncertainties'].items():
            self.uncertainty_results_text.insert(tk.END, f"  σ({feat}): {unc:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"\nOutput Uncertainty: {results['output_uncertainty']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"Propagated σ: {results['propagated_uncertainty']:.4f}\n")
    
    def monte_carlo_analysis(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.uncertainty.monte_carlo_analysis(features, target,
                                                       self.bootstrap_samples.get(),
                                                       self.confidence_level.get())
        
        self.uncertainty_results_text.delete(1.0, tk.END)
        self.uncertainty_results_text.insert(tk.END, 
            f"═══ MONTE CARLO ANALYSIS ({results['n_simulations']} simulations) ═══\n\n")
        self.uncertainty_results_text.insert(tk.END, f"Mean Prediction Uncertainty: {results['mean_uncertainty']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"Max Prediction Uncertainty: {results['max_uncertainty']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"Mean CI Width: {results['mean_ci_width']:.4f}\n")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(results['std_distribution'], bins=50, alpha=0.7)
        ax.set_xlabel('Prediction Uncertainty')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Uncertainties (Monte Carlo)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def residual_analysis(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.uncertainty.residual_analysis(features, target)
        
        self.uncertainty_results_text.delete(1.0, tk.END)
        self.uncertainty_results_text.insert(tk.END, "═══ RESIDUAL ANALYSIS ═══\n\n")
        self.uncertainty_results_text.insert(tk.END, "Diagnostic Tests:\n")
        self.uncertainty_results_text.insert(tk.END, 
            f"  Durbin-Watson: {results['durbin_watson']:.4f} (2=no autocorrelation)\n")
        self.uncertainty_results_text.insert(tk.END,
            f"  Breusch-Pagan p-value: {results['bp_pvalue']:.4f} (>0.05=homoscedastic)\n")
        self.uncertainty_results_text.insert(tk.END,
            f"  Shapiro-Wilk p-value: {results['shapiro_pvalue']:.4f} (>0.05=normal)\n\n")
        self.uncertainty_results_text.insert(tk.END, "Residual Statistics:\n")
        self.uncertainty_results_text.insert(tk.END, f"  Mean: {results['mean']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"  Std Dev: {results['std']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"  Skewness: {results['skewness']:.4f}\n")
        self.uncertainty_results_text.insert(tk.END, f"  Kurtosis: {results['kurtosis']:.4f}\n")
        
        fig = self.uncertainty.plot_residual_diagnostics(results)
        if fig:
            plt.show()
    
    # ========================================================================
    # NON-LINEAR ANALYSIS METHODS
    # ========================================================================
    
    def mutual_information(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.mutual_information(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ MUTUAL INFORMATION ═══\n\n")
        
        for feat, score in results.items():
            self.nonlinear_results_text.insert(tk.END, f"{feat}: {score:.4f}\n")
        
        fig = self.nonlinear.plot_mutual_information(results)
        if fig:
            plt.show()
    
    def distance_correlation(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.distance_correlation(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ DISTANCE CORRELATION ═══\n\n")
        
        for feat, score in results.items():
            self.nonlinear_results_text.insert(tk.END, f"{feat} vs {target}: {score:.4f}\n")
    
    def maximal_information(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.maximal_information_coefficient(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ MAXIMAL INFORMATION COEFFICIENT ═══\n\n")
        
        for feat, score in results.items():
            self.nonlinear_results_text.insert(tk.END, f"{feat} vs {target}: {score:.4f}\n")
    
    def gaussian_process(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.gaussian_process_regression(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ GAUSSIAN PROCESS REGRESSION ═══\n\n")
        self.nonlinear_results_text.insert(tk.END, f"R² Score: {results['r2']:.4f}\n")
        self.nonlinear_results_text.insert(tk.END, f"RMSE: {results['rmse']:.4f}\n")
        self.nonlinear_results_text.insert(tk.END, f"Mean Uncertainty: {results['mean_uncertainty']:.4f}\n")
        
        fig = self.nonlinear.plot_gp_predictions(results)
        if fig:
            plt.show()
    
    def polynomial_regression(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.polynomial_regression(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ POLYNOMIAL REGRESSION ═══\n\n")
        
        for degree, metrics in results.items():
            self.nonlinear_results_text.insert(tk.END,
                f"Degree {degree}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}\n")
        
        fig = self.nonlinear.plot_polynomial_comparison(results)
        if fig:
            plt.show()
    
    def spline_regression(self):
        features, target = self.get_selected_columns()
        if features is None or len(features) > 1:
            messagebox.showinfo("Info", "Spline regression works best with 1 feature")
            return
        
        results = self.nonlinear.spline_regression(features[0], target)
        
        if 'error' in results:
            messagebox.showerror("Error", results['error'])
            return
        
        fig = self.nonlinear.plot_spline_regression(results, features[0], target)
        if fig:
            plt.show()
    
    def neural_network_regression(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.neural_network_regression(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ NEURAL NETWORK REGRESSION ═══\n\n")
        self.nonlinear_results_text.insert(tk.END, f"Architecture: {results['architecture']}\n")
        self.nonlinear_results_text.insert(tk.END, f"Iterations: {results['iterations']}\n")
        self.nonlinear_results_text.insert(tk.END, f"R² Score: {results['r2']:.4f}\n")
        self.nonlinear_results_text.insert(tk.END, f"RMSE: {results['rmse']:.4f}\n")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results['y_test'], results['y_pred'], alpha=0.5)
        ax.plot([results['y_test'].min(), results['y_test'].max()],
               [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Neural Network: Predictions vs Actual')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def svm_regression(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.svm_regression(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ SVM REGRESSION ═══\n\n")
        
        for kernel, metrics in results.items():
            self.nonlinear_results_text.insert(tk.END,
                f"{kernel.upper()} Kernel: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}\n")
    
    def transfer_entropy(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.nonlinear.transfer_entropy(features, target)
        
        self.nonlinear_results_text.delete(1.0, tk.END)
        self.nonlinear_results_text.insert(tk.END, "═══ TRANSFER ENTROPY ═══\n\n")
        self.nonlinear_results_text.insert(tk.END, "(Measures directed information flow)\n\n")
        
        for feat, data in results.items():
            for key, val in data.items():
                if key != 'direction':
                    self.nonlinear_results_text.insert(tk.END, f"{key}: {val:.4f}\n")
            self.nonlinear_results_text.insert(tk.END, f"Direction: {data['direction']}\n\n")
    
    # ========================================================================
    # TIME SERIES METHODS
    # ========================================================================
    
    def acf_pacf_analysis(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.timeseries.plot_acf_pacf(features[0])
        if fig:
            plt.show()
    
    def stationarity_test(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.timeseries.stationarity_test(features)
        
        self.ts_results_text.delete(1.0, tk.END)
        self.ts_results_text.insert(tk.END, "═══ AUGMENTED DICKEY-FULLER TEST ═══\n\n")
        
        for col, data in results.items():
            if 'error' in data:
                self.ts_results_text.insert(tk.END, f"{col}: Error - {data['error']}\n")
                continue
            
            self.ts_results_text.insert(tk.END, f"\n{col}:\n")
            self.ts_results_text.insert(tk.END, f"  ADF Statistic: {data['adf_statistic']:.4f}\n")
            self.ts_results_text.insert(tk.END, f"  p-value: {data['p_value']:.4f}\n")
            self.ts_results_text.insert(tk.END, f"  Critical Values:\n")
            for key, val in data['critical_values'].items():
                self.ts_results_text.insert(tk.END, f"    {key}: {val:.4f}\n")
            status = "Stationary" if data['is_stationary'] else "Non-stationary"
            self.ts_results_text.insert(tk.END, f"  Result: {status}\n")
    
    def arima_model(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.timeseries.arima_model(features[0])
        
        if 'error' in results:
            messagebox.showerror("Error", f"ARIMA failed: {results['error']}")
            return
        
        self.ts_results_text.delete(1.0, tk.END)
        self.ts_results_text.insert(tk.END, "═══ ARIMA MODEL ═══\n\n")
        self.ts_results_text.insert(tk.END, results['summary'])
        
        fig = self.timeseries.plot_arima_fit(results)
        if fig:
            plt.show()
    
    def time_decomposition(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.timeseries.time_decomposition(features[0])
        
        if 'error' in results:
            messagebox.showerror("Error", f"Decomposition failed: {results['error']}")
            return
        
        fig = self.timeseries.plot_decomposition(results)
        if fig:
            plt.show()
    
    def rolling_stats(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        results = self.timeseries.rolling_statistics(features[0])
        
        fig = self.timeseries.plot_rolling_stats(results, features[0])
        if fig:
            plt.show()
    
    # ========================================================================
    # CAUSALITY METHODS
    # ========================================================================
    
    def granger_causality(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.causality.granger_causality(features, target, self.max_lag_var.get())
        
        self.causality_results_text.delete(1.0, tk.END)
        self.causality_results_text.insert(tk.END, "═══ GRANGER CAUSALITY TEST ═══\n\n")
        
        for feat, lag_data in results.items():
            self.causality_results_text.insert(tk.END, f"\n{feat} -> {target}:\n")
            
            if 'error' in lag_data:
                self.causality_results_text.insert(tk.END, f"  Error: {lag_data['error']}\n")
                continue
            
            for lag, data in lag_data.items():
                pval = data['ssr_ftest_pvalue']
                sig = " *significant*" if data['is_significant'] else ""
                self.causality_results_text.insert(tk.END, f"  Lag {lag}: p-value = {pval:.4f}{sig}\n")
        
        fig = self.causality.plot_granger_results(results, target)
        if fig:
            plt.show()
    
    def lead_lag_analysis(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.causality.lead_lag_analysis(features, target, self.max_lag_var.get())
        
        fig = self.causality.plot_lead_lag(results, target)
        if fig:
            plt.show()
    
    def correlation_lags(self):
        features, target = self.get_selected_columns()
        if features is None:
            return
        
        results = self.causality.correlation_at_lags(features, target, self.max_lag_var.get())
        
        self.causality_results_text.delete(1.0, tk.END)
        self.causality_results_text.insert(tk.END, "═══ CORRELATION AT DIFFERENT LAGS ═══\n\n")
        
        for feat, data in results.items():
            self.causality_results_text.insert(tk.END, f"\n{feat} vs {target}:\n")
            
            for item in data['lag_correlations']:
                self.causality_results_text.insert(tk.END,
                    f"  Lag {item['lag']:3d}: {item['correlation']:7.4f}\n")
            
            self.causality_results_text.insert(tk.END,
                f"  Best: Lag {data['best_lag']} with correlation {data['best_correlation']:.4f}\n")
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def scatter_matrix(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.visualization.scatter_matrix(features)
        if fig:
            plt.show()
    
    def correlation_heatmap(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.visualization.correlation_heatmap(features)
        if fig:
            plt.show()
    
    def box_plots(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.visualization.box_plots(features)
        if fig:
            plt.show()
    
    def fft_analysis(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.visualization.fft_analysis(features[0])
        if fig:
            plt.show()
    
    def noise_filter(self):
        features, _ = self.get_selected_columns()
        if features is None:
            return
        
        fig = self.visualization.noise_filter(features[0])
        if fig:
            plt.show()
    
    def scatter_3d(self):
        features, _ = self.get_selected_columns()
        if features is None or len(features) < 3:
            messagebox.showerror("Error", "Select at least 3 features for 3D plot")
            return
        
        fig = self.visualization.scatter_3d(features[:3])
        if fig:
            plt.show()
    
    # ========================================================================
    # PLUGINS TAB
    # ========================================================================
    
    def _create_plugins_tab(self):
        """Tab for plugin management"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔌 Plugins")
        
        # Main horizontal split
        main_paned = ttk.PanedWindow(tab, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Plugin list and controls
        left_frame = ttk.LabelFrame(tab, text="Loaded Plugins")
        main_paned.add(left_frame, weight=1)
        
        # Plugin listbox
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.plugin_listbox = tk.Listbox(list_frame, height=12,
                                         bg="#ffffff", fg=ModernTheme.TEXT_PRIMARY,
                                         selectbackground=ModernTheme.ACCENT,
                                         font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL))
        self.plugin_listbox.pack(side="left", fill="both", expand=True)
        self.plugin_listbox.bind('<<ListboxSelect>>', self._on_plugin_select)
        
        plugin_scroll = ttk.Scrollbar(list_frame, orient="vertical", 
                                      command=self.plugin_listbox.yview)
        plugin_scroll.pack(side="right", fill="y")
        self.plugin_listbox.configure(yscrollcommand=plugin_scroll.set)
        
        # Plugin buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="📁 Load File", 
                  command=self._load_plugin_file).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="📋 New/Paste", 
                  command=self._new_plugin).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="❌ Remove", 
                  command=self._remove_plugin).pack(side="left", padx=2)
        
        # Example plugins dropdown
        example_frame = ttk.Frame(left_frame)
        example_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(example_frame, text="Examples:").pack(side="left")
        self.example_var = tk.StringVar()
        examples = list(get_example_plugins().keys())
        self.example_combo = ttk.Combobox(example_frame, textvariable=self.example_var,
                                          values=examples, state="readonly", width=25)
        self.example_combo.pack(side="left", padx=5)
        ttk.Button(example_frame, text="Load", 
                  command=self._load_example_plugin).pack(side="left")
        
        # Plugin info display
        info_frame = ttk.LabelFrame(left_frame, text="Plugin Info")
        info_frame.pack(fill="x", padx=5, pady=5)
        
        self.plugin_info_label = ttk.Label(info_frame, text="Select a plugin to view details",
                                           wraplength=250, style="Info.TLabel")
        self.plugin_info_label.pack(padx=5, pady=5)
        
        # Right side - Code editor and execution
        right_frame = ttk.Frame(tab)
        main_paned.add(right_frame, weight=2)
        
        # Code editor
        code_frame = ttk.LabelFrame(right_frame, text="Plugin Code")
        code_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.plugin_code_text = tk.Text(code_frame, height=15, wrap="none",
                                        bg="#ffffff", fg=ModernTheme.TEXT_PRIMARY,
                                        insertbackground=ModernTheme.TEXT_PRIMARY,
                                        font=("Consolas", 10))
        self.plugin_code_text.pack(side="left", fill="both", expand=True)
        
        code_scroll = ttk.Scrollbar(code_frame, orient="vertical",
                                    command=self.plugin_code_text.yview)
        code_scroll.pack(side="right", fill="y")
        self.plugin_code_text.configure(yscrollcommand=code_scroll.set)
        
        # Code buttons
        code_btn_frame = ttk.Frame(right_frame)
        code_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(code_btn_frame, text="💾 Load/Update Plugin",
                  command=self._load_plugin_from_editor, 
                  style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(code_btn_frame, text="📝 Template",
                  command=self._insert_template).pack(side="left", padx=2)
        ttk.Button(code_btn_frame, text="💾 Save to File",
                  command=self._save_plugin_to_file).pack(side="left", padx=2)
        
        # Execution frame
        exec_frame = ttk.LabelFrame(right_frame, text="Execute Plugin")
        exec_frame.pack(fill="x", padx=5, pady=5)
        
        # Parameter frame (will be populated dynamically)
        self.plugin_params_frame = ttk.Frame(exec_frame)
        self.plugin_params_frame.pack(fill="x", padx=5, pady=5)
        self.plugin_param_widgets = {}
        
        exec_btn_frame = ttk.Frame(exec_frame)
        exec_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(exec_btn_frame, text="▶️ Run Plugin",
                  command=self._execute_plugin,
                  style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(exec_btn_frame, text="📊 Apply to Data",
                  command=self._apply_plugin_result).pack(side="left", padx=2)
        
        # Results display
        results_frame = ttk.LabelFrame(right_frame, text="Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.plugin_results_text = tk.Text(results_frame, height=6, wrap="word",
                                           bg="#ffffff", fg=ModernTheme.TEXT_PRIMARY,
                                           font=(ModernTheme.FONT_FAMILY, ModernTheme.FONT_SIZE_SMALL))
        self.plugin_results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Track current plugin and result
        self.current_plugin_id = None
        self.current_plugin_result = None
        
        # Update plugin list
        self._update_plugin_list()
        
        # Register for plugin changes
        self.plugin_manager.add_listener(self._update_plugin_list)
    
    def _update_plugin_list(self):
        """Update the plugin listbox"""
        self.plugin_listbox.delete(0, tk.END)
        
        for plugin in self.plugin_manager.get_all_plugins():
            category_icon = {
                'preprocessing': '🔧',
                'analysis': '📊',
                'postprocessing': '📤',
                'visualization': '📈',
            }.get(plugin.info.category, '🔌')
            
            self.plugin_listbox.insert(tk.END, 
                f"{category_icon} {plugin.info.name} (v{plugin.info.version})")
    
    def _on_plugin_select(self, event):
        """Handle plugin selection"""
        selection = self.plugin_listbox.curselection()
        if not selection:
            return
        
        plugins = self.plugin_manager.get_all_plugins()
        if selection[0] < len(plugins):
            plugin = plugins[selection[0]]
            self.current_plugin_id = plugin.id
            
            # Update info display
            info_text = (f"📌 {plugin.info.name}\n"
                        f"📝 {plugin.info.description}\n"
                        f"👤 Author: {plugin.info.author}\n"
                        f"📂 Category: {plugin.info.category}\n"
                        f"📥 Inputs: {', '.join(plugin.info.inputs)}\n"
                        f"📤 Outputs: {', '.join(plugin.info.outputs)}")
            self.plugin_info_label.configure(text=info_text)
            
            # Show code
            self.plugin_code_text.delete("1.0", tk.END)
            self.plugin_code_text.insert("1.0", plugin.source_code)
            
            # Update parameter widgets
            self._update_param_widgets(plugin)
    
    def _update_param_widgets(self, plugin):
        """Update parameter input widgets for a plugin"""
        # Clear existing widgets
        for widget in self.plugin_params_frame.winfo_children():
            widget.destroy()
        self.plugin_param_widgets.clear()
        
        if not plugin.parameters:
            ttk.Label(self.plugin_params_frame, text="No parameters",
                     style="Info.TLabel").pack()
            return
        
        for name, param in plugin.parameters.items():
            frame = ttk.Frame(self.plugin_params_frame)
            frame.pack(fill="x", pady=2)
            
            ttk.Label(frame, text=f"{name}:", width=15).pack(side="left")
            
            if param.type == 'bool':
                var = tk.BooleanVar(value=param.default)
                widget = ttk.Checkbutton(frame, variable=var)
            elif param.choices:
                var = tk.StringVar(value=param.default)
                widget = ttk.Combobox(frame, textvariable=var, 
                                     values=param.choices, state="readonly", width=15)
            else:
                var = tk.StringVar(value=str(param.default))
                widget = ttk.Entry(frame, textvariable=var, width=15)
            
            widget.pack(side="left", padx=5)
            
            if param.description:
                ttk.Label(frame, text=f"({param.description})", 
                         style="Info.TLabel").pack(side="left")
            
            self.plugin_param_widgets[name] = (param.type, var)
    
    def _get_plugin_params(self) -> dict:
        """Get current parameter values from widgets"""
        params = {}
        for name, (ptype, var) in self.plugin_param_widgets.items():
            value = var.get()
            if ptype == 'float':
                params[name] = float(value)
            elif ptype == 'int':
                params[name] = int(value)
            elif ptype == 'bool':
                params[name] = bool(value)
            else:
                params[name] = value
        return params
    
    def _load_plugin_file(self):
        """Load a plugin from a file"""
        filepath = filedialog.askopenfilename(
            title="Select Plugin File",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filepath:
            plugin, message = self.plugin_manager.load_from_file(filepath)
            self.plugin_results_text.delete("1.0", tk.END)
            self.plugin_results_text.insert("1.0", message)
    
    def _new_plugin(self):
        """Create a new plugin or paste code"""
        self.plugin_code_text.delete("1.0", tk.END)
        self.plugin_code_text.insert("1.0", get_plugin_template())
        self.current_plugin_id = None
        self.plugin_results_text.delete("1.0", tk.END)
        self.plugin_results_text.insert("1.0", "📝 Edit the template and click 'Load/Update Plugin'")
    
    def _insert_template(self):
        """Insert template at cursor"""
        self.plugin_code_text.delete("1.0", tk.END)
        self.plugin_code_text.insert("1.0", get_plugin_template())
    
    def _load_example_plugin(self):
        """Load an example plugin"""
        example_name = self.example_var.get()
        if not example_name:
            return
        
        examples = get_example_plugins()
        if example_name in examples:
            code = examples[example_name]
            self.plugin_code_text.delete("1.0", tk.END)
            self.plugin_code_text.insert("1.0", code)
            
            # Automatically load it
            plugin, message = self.plugin_manager.load_from_code(code)
            self.plugin_results_text.delete("1.0", tk.END)
            self.plugin_results_text.insert("1.0", message)
            
            if plugin:
                self.current_plugin_id = plugin.id
                self._update_param_widgets(plugin)
    
    def _load_plugin_from_editor(self):
        """Load or update plugin from the code editor"""
        code = self.plugin_code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to load")
            return
        
        plugin, message = self.plugin_manager.load_from_code(code)
        self.plugin_results_text.delete("1.0", tk.END)
        self.plugin_results_text.insert("1.0", message)
        
        if plugin:
            self.current_plugin_id = plugin.id
            self._update_param_widgets(plugin)
    
    def _remove_plugin(self):
        """Remove the selected plugin"""
        if self.current_plugin_id:
            self.plugin_manager.unload_plugin(self.current_plugin_id)
            self.current_plugin_id = None
            self.plugin_code_text.delete("1.0", tk.END)
            self.plugin_info_label.configure(text="Select a plugin to view details")
    
    def _save_plugin_to_file(self):
        """Save current plugin code to a file"""
        code = self.plugin_code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Plugin",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(code)
                messagebox.showinfo("Success", f"Saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _execute_plugin(self):
        """Execute the current plugin"""
        if not self.current_plugin_id:
            messagebox.showwarning("Warning", "No plugin selected")
            return
        
        if self.data_loader.df is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        features, target = self.get_selected_columns()
        params = self._get_plugin_params()
        
        result, message = self.plugin_manager.execute_plugin(
            self.current_plugin_id,
            self.data_loader.df,
            columns=features,
            target=target,
            **params
        )
        
        self.current_plugin_result = result
        
        self.plugin_results_text.delete("1.0", tk.END)
        self.plugin_results_text.insert("1.0", message + "\n\n")
        
        if result is not None:
            if isinstance(result, pd.DataFrame):
                self.plugin_results_text.insert(tk.END, 
                    f"Output: DataFrame with {len(result)} rows, {len(result.columns)} columns\n")
                self.plugin_results_text.insert(tk.END, 
                    f"New columns: {[c for c in result.columns if c not in self.data_loader.df.columns]}")
            elif isinstance(result, dict):
                self.plugin_results_text.insert(tk.END, "Output:\n")
                for key, value in result.items():
                    self.plugin_results_text.insert(tk.END, f"  {key}: {value}\n")
            elif isinstance(result, plt.Figure):
                self.plugin_results_text.insert(tk.END, "Output: Figure (displayed)")
                plt.show()
    
    def _apply_plugin_result(self):
        """Apply plugin result to the main data"""
        if self.current_plugin_result is None:
            messagebox.showwarning("Warning", "No result to apply. Run a plugin first.")
            return
        
        if isinstance(self.current_plugin_result, pd.DataFrame):
            # Replace or merge with current data
            if messagebox.askyesno("Apply Result", 
                                   "Replace current data with plugin output?"):
                self.data_loader.df = self.current_plugin_result
                self._update_all_modules()
                self._refresh_column_list()
                self.data_preview.load_data(self.current_plugin_result)
                messagebox.showinfo("Success", "Data updated with plugin result!")
        else:
            messagebox.showinfo("Info", 
                "This result type cannot be applied to data. "
                "Only DataFrame results can update the data.")
    
    # ========================================================================
    # BACKEND TOGGLE METHODS
    # ========================================================================
    
    def _toggle_backend(self):
        """Toggle between Rust and Python backends"""
        use_rust = self.backend_var.get()
        AccelerationSettings.set_use_rust(use_rust)
        self._update_backend_status()
    
    def _update_backend_status(self):
        """Update the backend status display"""
        if self.rust_available:
            if AccelerationSettings.use_rust():
                status = "⚡ Backend: Rust (Fast)"
                self.backend_label.configure(foreground=ModernTheme.SUCCESS)
            else:
                status = "🐍 Backend: Python"
                self.backend_label.configure(foreground=ModernTheme.TEXT_SECONDARY)
        else:
            status = "🐍 Backend: Python (Rust not compiled)"
            self.backend_label.configure(foreground=ModernTheme.TEXT_SECONDARY)
        
        self.backend_label.configure(text=status)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = AdvancedDataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
