"""
Shared imports and constants for all tab modules.
Import this at the top of each tab file.
"""

import os
import sys
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Plotly theme configuration
PLOTLY_TEMPLATE = "plotly_white"

# Ensure package directory is in path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
