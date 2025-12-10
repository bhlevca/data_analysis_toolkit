#!/usr/bin/env python
"""
Quick Run Script
================

Run this script to launch the Data Analysis Toolkit GUI
without installing the package.

Usage:
    python run.py
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_toolkit.main_gui import main

if __name__ == '__main__':
    print("Starting Advanced Data Analysis Toolkit...")
    main()
