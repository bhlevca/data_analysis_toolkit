"""
Report Generator Module
========================
Automated generation of reproducible scientific reports including:
- PDF and HTML report generation
- Analysis pipeline export as Python scripts
- Figure management with publication formats
- Method descriptions with citations
- Statistical results tables
- Data provenance tracking

Version: 1.0

Requirements:
    pip install jinja2 pdfkit weasyprint markdown
"""

import json
import os
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Optional imports for PDF generation
try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class ReportGenerator:
    """Generate reproducible scientific reports"""
    
    def __init__(self, title: str = "Data Analysis Report", 
                 author: str = "Data Analysis Toolkit",
                 output_dir: str = "./reports"):
        """
        Initialize report generator
        
        Args:
            title: Report title
            author: Author name
            output_dir: Output directory for reports
        """
        self.title = title
        self.author = author
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sections = []
        self.figures = []
        self.tables = []
        self.code_blocks = []
        self.analysis_log = []
        self.data_provenance = []
        self.random_seeds = {}
        
        self.created_at = datetime.now()
        self._analysis_start = None
    
    def start_analysis(self, description: str = "Analysis session"):
        """Mark the start of an analysis session for logging"""
        self._analysis_start = datetime.now()
        self.log_event("analysis_start", description)
    
    def end_analysis(self):
        """Mark the end of an analysis session"""
        if self._analysis_start:
            duration = datetime.now() - self._analysis_start
            self.log_event("analysis_end", f"Duration: {duration}")
    
    def log_event(self, event_type: str, description: str, metadata: Dict = None):
        """Log an analysis event for reproducibility"""
        self.analysis_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description,
            'metadata': metadata or {}
        })
    
    def set_seed(self, seed: int, component: str = "global"):
        """Record random seed for reproducibility"""
        self.random_seeds[component] = seed
        np.random.seed(seed)
        self.log_event("seed_set", f"Random seed set to {seed} for {component}")
    
    # =========================================================================
    # DATA PROVENANCE
    # =========================================================================
    
    def add_data_source(self, name: str, path: str = None, description: str = None,
                        n_rows: int = None, n_cols: int = None,
                        columns: List[str] = None, transformations: List[str] = None):
        """
        Record data source for provenance tracking
        
        Args:
            name: Dataset name
            path: File path or URL
            description: Description of data
            n_rows, n_cols: Dataset dimensions
            columns: Column names
            transformations: List of transformations applied
        """
        self.data_provenance.append({
            'name': name,
            'path': path,
            'description': description,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'columns': columns,
            'transformations': transformations or [],
            'loaded_at': datetime.now().isoformat()
        })
        self.log_event("data_loaded", f"Loaded dataset: {name}")
    
    def add_transformation(self, dataset_name: str, transformation: str, 
                           details: Dict = None):
        """Add transformation to data provenance"""
        for source in self.data_provenance:
            if source['name'] == dataset_name:
                source['transformations'].append({
                    'transformation': transformation,
                    'details': details,
                    'timestamp': datetime.now().isoformat()
                })
                break
        self.log_event("transformation", f"{transformation} applied to {dataset_name}")
    
    # =========================================================================
    # CONTENT ADDITION
    # =========================================================================
    
    def add_section(self, title: str, content: str, level: int = 2):
        """
        Add a section to the report
        
        Args:
            title: Section title
            content: Section content (Markdown supported)
            level: Heading level (1-4)
        """
        self.sections.append({
            'type': 'section',
            'title': title,
            'content': content,
            'level': level
        })
    
    def add_text(self, content: str):
        """Add plain text paragraph"""
        self.sections.append({
            'type': 'text',
            'content': content
        })
    
    def add_figure(self, fig: plt.Figure, caption: str, label: str = None,
                   dpi: int = 150, formats: List[str] = None) -> str:
        """
        Add a matplotlib figure to the report
        
        Args:
            fig: Matplotlib figure
            caption: Figure caption
            label: Reference label (e.g., 'fig:scatter')
            dpi: Resolution for raster formats
            formats: Export formats ['png', 'svg', 'eps', 'pdf']
            
        Returns:
            Figure filename
        """
        if formats is None:
            formats = ['png']
        
        fig_num = len(self.figures) + 1
        if label is None:
            label = f"fig_{fig_num}"
        
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        filenames = {}
        for fmt in formats:
            filename = f"{label}.{fmt}"
            filepath = fig_dir / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=fmt)
            filenames[fmt] = str(filepath)
        
        self.figures.append({
            'num': fig_num,
            'label': label,
            'caption': caption,
            'filenames': filenames,
            'primary': filenames.get('png', list(filenames.values())[0])
        })
        
        self.sections.append({
            'type': 'figure',
            'ref': label
        })
        
        plt.close(fig)
        return label
    
    def add_table(self, df: pd.DataFrame, caption: str, label: str = None,
                  precision: int = 4, max_rows: int = 50) -> str:
        """
        Add a DataFrame table to the report
        
        Args:
            df: Pandas DataFrame
            caption: Table caption
            label: Reference label
            precision: Decimal precision for floats
            max_rows: Maximum rows to display
            
        Returns:
            Table label
        """
        table_num = len(self.tables) + 1
        if label is None:
            label = f"tab_{table_num}"
        
        # Format DataFrame
        df_display = df.head(max_rows).copy()
        for col in df_display.select_dtypes(include=[np.floating]):
            df_display[col] = df_display[col].round(precision)
        
        self.tables.append({
            'num': table_num,
            'label': label,
            'caption': caption,
            'data': df_display,
            'html': df_display.to_html(classes='table table-striped', index=True),
            'latex': df_display.to_latex(index=True),
            'truncated': len(df) > max_rows
        })
        
        self.sections.append({
            'type': 'table',
            'ref': label
        })
        
        return label
    
    def add_data_provenance(self, df: pd.DataFrame, name: str = "Dataset"):
        """
        Add data provenance information from a DataFrame
        
        Args:
            df: DataFrame to document
            name: Name for the dataset
        """
        self.add_data_source(
            name=name,
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=list(df.columns),
            description=f"DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        
        # Add summary section
        summary = {
            'Rows': len(df),
            'Columns': len(df.columns),
            'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'Missing Values': int(df.isnull().sum().sum()),
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        }
        
        self.add_statistics_summary(summary, title="Data Summary")
    
    def add_statistics_table(self, results: Dict[str, Any], title: str = "Results"):
        """
        Add statistical results as a formatted table
        
        Args:
            results: Dictionary of results
            title: Table title
        """
        # Convert dict to DataFrame for table display
        rows = []
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool)):
                if isinstance(value, float):
                    rows.append({'Metric': key, 'Value': f"{value:.4f}"})
                else:
                    rows.append({'Metric': key, 'Value': str(value)})
        
        if rows:
            df_results = pd.DataFrame(rows)
            self.add_table(df_results, caption=title)
        else:
            self.add_statistics_summary(results, title=title)
    
    def add_statistics_summary(self, results: Dict[str, Any], title: str = "Statistical Results"):
        """
        Add a formatted statistical results section
        
        Args:
            results: Dictionary of statistical results
            title: Section title
        """
        content = self._format_statistics(results)
        self.add_section(title, content, level=3)
    
    def _format_statistics(self, results: Dict[str, Any], indent: int = 0) -> str:
        """Format statistical results as Markdown"""
        lines = []
        prefix = "  " * indent
        
        for key, value in results.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(self._format_statistics(value, indent + 1))
            elif isinstance(value, (list, np.ndarray)):
                if len(value) <= 5:
                    formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in value]
                    lines.append(f"{prefix}- **{key}**: [{', '.join(formatted)}]")
                else:
                    lines.append(f"{prefix}- **{key}**: [array with {len(value)} elements]")
            elif isinstance(value, float):
                lines.append(f"{prefix}- **{key}**: {value:.4f}")
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
        
        return "\n".join(lines)
    
    def add_method_description(self, method: str, description: str = None,
                               parameters: Dict = None, citation: str = None):
        """
        Add a method description with optional citation
        
        Args:
            method: Method name
            description: Method description
            parameters: Parameters used
            citation: Citation reference
        """
        content = f"### {method}\n\n"
        
        if description:
            content += f"{description}\n\n"
        
        if parameters:
            content += "**Parameters:**\n"
            for param, value in parameters.items():
                content += f"- `{param}`: {value}\n"
            content += "\n"
        
        if citation:
            content += f"*Reference: {citation}*\n"
        
        self.add_text(content)
        self.log_event("method_used", method, {'parameters': parameters, 'citation': citation})
    
    def add_code_block(self, code: str, language: str = "python", 
                       description: str = None):
        """Add a code block for reproducibility"""
        self.code_blocks.append({
            'code': code,
            'language': language,
            'description': description
        })
        
        content = ""
        if description:
            content += f"*{description}*\n\n"
        content += f"```{language}\n{code}\n```"
        
        self.add_text(content)
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def generate_html(self, filename: str = None) -> str:
        """
        Generate HTML report
        
        Args:
            filename: Output filename (without extension)
            
        Returns:
            Path to generated HTML file
        """
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        html_content = self._build_html()
        
        output_path = self.output_dir / f"{filename}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log_event("report_generated", f"HTML report: {output_path}")
        return str(output_path)
    
    def _build_html(self) -> str:
        """Build HTML report content"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #3498db; color: white; }}
        .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; height: auto; }}
        .figure-caption {{ font-style: italic; color: #666; margin-top: 10px; }}
        .code {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
        .metadata {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .provenance {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
        blockquote {{ border-left: 4px solid #3498db; padding-left: 15px; color: #666; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="metadata">
        <strong>Author:</strong> {self.author}<br>
        <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>Toolkit Version:</strong> Data Analysis Toolkit
    </div>
"""
        
        # Table of Contents
        html += "<h2>Table of Contents</h2>\n<ul>\n"
        section_num = 0
        for item in self.sections:
            if item['type'] == 'section':
                section_num += 1
                html += f"<li><a href='#section-{section_num}'>{item['title']}</a></li>\n"
        html += "</ul>\n"
        
        # Data Provenance
        if self.data_provenance:
            html += "<h2>Data Provenance</h2>\n<div class='provenance'>\n"
            for source in self.data_provenance:
                html += f"<h4>{source['name']}</h4>\n"
                if source['path']:
                    html += f"<p><strong>Source:</strong> {source['path']}</p>\n"
                if source['n_rows'] and source['n_cols']:
                    html += f"<p><strong>Dimensions:</strong> {source['n_rows']} rows × {source['n_cols']} columns</p>\n"
                if source['transformations']:
                    html += "<p><strong>Transformations:</strong></p><ul>\n"
                    for t in source['transformations']:
                        if isinstance(t, dict):
                            html += f"<li>{t.get('transformation', t)}</li>\n"
                        else:
                            html += f"<li>{t}</li>\n"
                    html += "</ul>\n"
            html += "</div>\n"
        
        # Content sections
        section_num = 0
        for item in self.sections:
            if item['type'] == 'section':
                section_num += 1
                level = item.get('level', 2)
                html += f"<h{level} id='section-{section_num}'>{item['title']}</h{level}>\n"
                if MARKDOWN_AVAILABLE:
                    html += markdown.markdown(item['content'])
                else:
                    html += f"<p>{item['content']}</p>\n"
            
            elif item['type'] == 'text':
                if MARKDOWN_AVAILABLE:
                    html += markdown.markdown(item['content'])
                else:
                    html += f"<p>{item['content']}</p>\n"
            
            elif item['type'] == 'figure':
                fig_info = next((f for f in self.figures if f['label'] == item['ref']), None)
                if fig_info:
                    html += f"""<div class="figure">
                        <img src="{fig_info['primary']}" alt="{fig_info['label']}">
                        <div class="figure-caption">Figure {fig_info['num']}: {fig_info['caption']}</div>
                    </div>\n"""
            
            elif item['type'] == 'table':
                table_info = next((t for t in self.tables if t['label'] == item['ref']), None)
                if table_info:
                    html += f"<h4>Table {table_info['num']}: {table_info['caption']}</h4>\n"
                    html += table_info['html'] + "\n"
                    if table_info['truncated']:
                        html += "<p><em>(Table truncated)</em></p>\n"
        
        # Random seeds
        if self.random_seeds:
            html += "<h2>Reproducibility Information</h2>\n"
            html += "<div class='metadata'>\n<h4>Random Seeds</h4>\n<ul>\n"
            for component, seed in self.random_seeds.items():
                html += f"<li><strong>{component}:</strong> {seed}</li>\n"
            html += "</ul></div>\n"
        
        # Analysis log
        if self.analysis_log:
            html += "<h2>Analysis Log</h2>\n<table class='table'>\n"
            html += "<tr><th>Timestamp</th><th>Event</th><th>Description</th></tr>\n"
            for event in self.analysis_log[-20:]:  # Last 20 events
                html += f"<tr><td>{event['timestamp']}</td><td>{event['type']}</td><td>{event['description']}</td></tr>\n"
            html += "</table>\n"
        
        html += """
</body>
</html>"""
        
        return html
    
    def generate_markdown(self, filename: str = None) -> str:
        """
        Generate Markdown report
        
        Returns:
            Path to generated Markdown file
        """
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        md_content = self._build_markdown()
        
        output_path = self.output_dir / f"{filename}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(output_path)
    
    def _build_markdown(self) -> str:
        """Build Markdown report content"""
        lines = [
            f"# {self.title}",
            "",
            f"**Author:** {self.author}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
            "---",
            ""
        ]
        
        # Data Provenance
        if self.data_provenance:
            lines.extend(["## Data Provenance", ""])
            for source in self.data_provenance:
                lines.append(f"### {source['name']}")
                if source['path']:
                    lines.append(f"- **Source:** `{source['path']}`")
                if source['n_rows']:
                    lines.append(f"- **Dimensions:** {source['n_rows']} × {source['n_cols']}")
                lines.append("")
        
        # Content
        for item in self.sections:
            if item['type'] == 'section':
                level = item.get('level', 2)
                lines.append("#" * level + " " + item['title'])
                lines.append("")
                lines.append(item['content'])
                lines.append("")
            
            elif item['type'] == 'text':
                lines.append(item['content'])
                lines.append("")
            
            elif item['type'] == 'figure':
                fig_info = next((f for f in self.figures if f['label'] == item['ref']), None)
                if fig_info:
                    lines.append(f"![{fig_info['caption']}]({fig_info['primary']})")
                    lines.append(f"*Figure {fig_info['num']}: {fig_info['caption']}*")
                    lines.append("")
            
            elif item['type'] == 'table':
                table_info = next((t for t in self.tables if t['label'] == item['ref']), None)
                if table_info:
                    lines.append(f"**Table {table_info['num']}: {table_info['caption']}**")
                    lines.append("")
                    lines.append(table_info['data'].to_markdown())
                    lines.append("")
        
        return "\n".join(lines)
    
    def export_analysis_script(self, filename: str = None) -> str:
        """
        Export analysis as reproducible Python script
        
        Returns:
            Path to generated Python script
        """
        if filename is None:
            filename = f"analysis_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        script_lines = [
            '"""',
            f'Reproducible Analysis Script',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'Title: {self.title}',
            f'Author: {self.author}',
            '"""',
            '',
            '# Standard imports',
            'import numpy as np',
            'import pandas as pd',
            'import matplotlib.pyplot as plt',
            'from data_toolkit import *',
            '',
        ]
        
        # Add random seeds
        if self.random_seeds:
            script_lines.append('# Random seeds for reproducibility')
            for component, seed in self.random_seeds.items():
                if component == 'global':
                    script_lines.append(f'np.random.seed({seed})')
                else:
                    script_lines.append(f'# {component}: {seed}')
            script_lines.append('')
        
        # Add data loading
        if self.data_provenance:
            script_lines.append('# Data loading')
            for source in self.data_provenance:
                if source['path']:
                    script_lines.append(f"# Load {source['name']}")
                    script_lines.append(f"df = pd.read_csv('{source['path']}')")
            script_lines.append('')
        
        # Add code blocks
        for block in self.code_blocks:
            if block['description']:
                script_lines.append(f"# {block['description']}")
            script_lines.append(block['code'])
            script_lines.append('')
        
        # Add analysis log as comments
        script_lines.append('# Analysis log:')
        for event in self.analysis_log:
            script_lines.append(f"# {event['timestamp']}: {event['type']} - {event['description']}")
        
        script_content = '\n'.join(script_lines)
        
        output_path = self.output_dir / f"{filename}.py"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(output_path)
    
    def export_latex_tables(self, filename: str = None) -> str:
        """
        Export all tables as LaTeX
        
        Returns:
            Path to LaTeX file
        """
        if filename is None:
            filename = "tables"
        
        latex_lines = [
            "% LaTeX tables generated by Data Analysis Toolkit",
            f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        for table in self.tables:
            latex_lines.extend([
                f"% Table {table['num']}: {table['caption']}",
                f"\\begin{{table}}[htbp]",
                f"\\centering",
                f"\\caption{{{table['caption']}}}",
                f"\\label{{{table['label']}}}",
                table['latex'],
                f"\\end{{table}}",
                ""
            ])
        
        output_path = self.output_dir / f"{filename}.tex"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))
        
        return str(output_path)
    
    def save_session(self, filename: str = None) -> str:
        """
        Save complete session state as JSON
        
        Returns:
            Path to JSON file
        """
        if filename is None:
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            'title': self.title,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'saved_at': datetime.now().isoformat(),
            'random_seeds': self.random_seeds,
            'data_provenance': self.data_provenance,
            'analysis_log': self.analysis_log,
            'code_blocks': self.code_blocks,
            'figures': [{'num': f['num'], 'label': f['label'], 'caption': f['caption']} 
                       for f in self.figures],
            'tables': [{'num': t['num'], 'label': t['label'], 'caption': t['caption']} 
                      for t in self.tables]
        }
        
        output_path = self.output_dir / f"{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return str(output_path)


# =============================================================================
# CITATION HELPER
# =============================================================================

COMMON_CITATIONS = {
    'cohens_d': "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
    'hedges_g': "Hedges, L. V., & Olkin, I. (1985). Statistical Methods for Meta-Analysis. Academic Press.",
    'bootstrap': "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    'anova': "Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver and Boyd.",
    'granger_causality': "Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models. Econometrica, 37(3), 424-438.",
    'mann_whitney': "Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other. Annals of Mathematical Statistics, 18(1), 50-60.",
    'shapiro_wilk': "Shapiro, S. S., & Wilk, M. B. (1965). An Analysis of Variance Test for Normality. Biometrika, 52(3-4), 591-611.",
    'fdr': "Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. Journal of the Royal Statistical Society B, 57(1), 289-300.",
    'shap': "Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
    'kaplan_meier': "Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. JASA, 53(282), 457-481.",
    'cox': "Cox, D. R. (1972). Regression Models and Life-Tables. Journal of the Royal Statistical Society B, 34(2), 187-220.",
    'prophet': "Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. The American Statistician, 72(1), 37-45.",
    'adf': "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the Estimators for Autoregressive Time Series. JASA, 74(366), 427-431."
}


def get_citation(method: str) -> str:
    """Get citation for a statistical method"""
    return COMMON_CITATIONS.get(method.lower(), f"Citation for {method} not available.")
