"""
Data Loading Module
Handles file loading, data preview, and column management
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


class DataLoader:
    """Handles data loading and basic data operations"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.file_name: Optional[str] = None
        self.loaded_files: List[Dict[str, Any]] = []
    
    def load_csv(self, path: str) -> Tuple[bool, str]:
        """
        Load a CSV file into a DataFrame
        
        Args:
            path: Path to the CSV file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.df = pd.read_csv(path)
            self.file_path = path
            self.file_name = Path(path).name
            
            file_info = {
                'name': self.file_name,
                'path': path,
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': list(self.df.columns)
            }
            self.loaded_files.append(file_info)
            
            return True, f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns"
        except Exception as e:
            return False, str(e)
    
    def load_excel(self, path: str, sheet_name: int = 0) -> Tuple[bool, str]:
        """Load an Excel file"""
        try:
            self.df = pd.read_excel(path, sheet_name=sheet_name)
            self.file_path = path
            self.file_name = Path(path).name
            
            file_info = {
                'name': self.file_name,
                'path': path,
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': list(self.df.columns)
            }
            self.loaded_files.append(file_info)
            
            return True, f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns"
        except Exception as e:
            return False, str(e)
    
    def get_columns(self) -> List[str]:
        """Get list of column names"""
        if self.df is None:
            return []
        return list(self.df.columns)
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names"""
        if self.df is None:
            return []
        return list(self.df.select_dtypes(include=[np.number]).columns)
    
    def get_preview(self, n_rows: int = 20) -> pd.DataFrame:
        """Get preview of first n rows"""
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(n_rows)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive data info"""
        if self.df is None:
            return {}
        
        return {
            'file_name': self.file_name,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': dict(self.df.dtypes.astype(str)),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'null_counts': dict(self.df.isnull().sum()),
            'numeric_columns': self.get_numeric_columns()
        }
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get statistics for a specific column"""
        if self.df is None or column not in self.df.columns:
            return {}
        
        col = self.df[column]
        stats = {
            'dtype': str(col.dtype),
            'non_null': col.count(),
            'null': col.isnull().sum(),
            'unique': col.nunique()
        }
        
        if np.issubdtype(col.dtype, np.number):
            stats.update({
                'mean': col.mean(),
                'std': col.std(),
                'min': col.min(),
                'max': col.max(),
                'median': col.median()
            })
        
        return stats
    
    def get_selected_data(self, features: List[str], target: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Get feature matrix and target vector with aligned indices (NaN removed)
        
        Args:
            features: List of feature column names
            target: Target column name
            
        Returns:
            Tuple of (X DataFrame, y Series) or (None, None) if data unavailable
        """
        if self.df is None:
            return None, None
        
        try:
            X = self.df[features].dropna()
            y = self.df[target].loc[X.index]
            return X, y
        except Exception:
            return None, None
    
    def clear_data(self):
        """Clear loaded data"""
        self.df = None
        self.file_path = None
        self.file_name = None
        self.loaded_files = []
