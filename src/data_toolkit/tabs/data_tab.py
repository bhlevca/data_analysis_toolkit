"""
Data Loading Tab for the Data Analysis Toolkit
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

PLOTLY_TEMPLATE = "plotly_white"

# Import datetime utilities
from utils.datetime_utils import detect_and_convert_datetime_columns


def render_data_tab():
    """Render the data loading tab"""
    st.header("📁 Data Loading & Column Selection")
    st.caption("Upload CSV or Excel files, preview data, and select feature/target columns for analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)

                # Detect and convert date/time columns to numeric
                st.session_state.df = detect_and_convert_datetime_columns(st.session_state.df)

                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col2:
        st.markdown("### Or use sample data:")
        if st.button("🎲 Generate Sample Data"):
            np.random.seed(42)
            n = 200
            x1 = np.random.normal(50, 10, n)
            x2 = 0.7 * x1 + np.random.normal(0, 5, n)
            x3 = np.random.uniform(0, 100, n)
            target = 2.5 * x1 + 1.8 * x2 - 0.5 * x3 + np.random.normal(0, 15, n)

            st.session_state.df = pd.DataFrame({
                'feature_1': x1,
                'feature_2': x2,
                'feature_3': x3,
                'target': target
            })

            # Detect and convert any date/time columns in sample data
            st.session_state.df = detect_and_convert_datetime_columns(st.session_state.df)

            st.success("✅ Sample data generated!")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("---")

        # Data info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        col4.metric("Missing", df.isnull().sum().sum())

        st.markdown("---")

        # Column selection
        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Initialize feature_cols in session state if not set or if columns changed
            if 'feature_cols_widget' not in st.session_state:
                initial_features = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                st.session_state.feature_cols_widget = initial_features
            else:
                st.session_state.feature_cols_widget = [
                    f for f in st.session_state.feature_cols_widget if f in numeric_cols
                ]

            st.multiselect(
                "📊 Select Feature Columns",
                options=numeric_cols,
                key="feature_cols_widget",
                help="Select one or more columns to use as input features for analysis"
            )
            st.session_state.feature_cols = st.session_state.feature_cols_widget

        with col2:
            target_options = ['None'] + numeric_cols

            if 'target_col_widget' not in st.session_state:
                st.session_state.target_col_widget = 'None'
            elif st.session_state.target_col_widget not in target_options:
                st.session_state.target_col_widget = 'None'

            st.selectbox(
                "🎯 Select Target Column",
                options=target_options,
                key="target_col_widget",
                help="Select the variable you want to predict (for supervised learning)"
            )
            st.session_state.target_col = st.session_state.target_col_widget if st.session_state.target_col_widget != 'None' else None

        # Validation
        validation_errors = []
        validation_warnings = []

        if st.session_state.target_col and st.session_state.target_col in st.session_state.feature_cols:
            validation_errors.append(
                f"⚠️ **Column Overlap Detected**: '{st.session_state.target_col}' is selected as BOTH a feature AND the target."
            )

        if len(st.session_state.feature_cols) == 0:
            validation_warnings.append("💡 No feature columns selected.")

        if st.session_state.target_col is None:
            validation_warnings.append("💡 No target column selected.")

        for error in validation_errors:
            st.error(error)
        for warning in validation_warnings:
            st.info(warning)

        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), width='stretch')

        # Quick plot
        if len(st.session_state.feature_cols) >= 1:
            st.markdown("### Quick Visualization (Interactive!)")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                all_numeric = numeric_cols
                x_col = st.selectbox("X-axis", all_numeric, index=0, key="quick_viz_x")

            with col2:
                default_y_idx = 1 if len(all_numeric) > 1 else 0
                if st.session_state.target_col and st.session_state.target_col in all_numeric:
                    default_y_idx = all_numeric.index(st.session_state.target_col)
                y_col = st.selectbox("Y-axis", all_numeric, index=default_y_idx, key="quick_viz_y")

            if x_col and y_col:
                try:
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        trendline="ols",
                        title=f'{y_col} vs {x_col}',
                        template=PLOTLY_TEMPLATE
                    )
                    fig.update_layout(height=500, xaxis_title=x_col, yaxis_title=y_col)
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.warning(f"Could not generate quick visualization: {e}")
