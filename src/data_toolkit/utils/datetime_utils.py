"""
Date/Time parsing utilities for the Data Analysis Toolkit
"""

import pandas as pd
import streamlit as st


def detect_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect columns with date/time data and convert them to numeric timestamps.
    Handles various formats including:
    - MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD
    - HH:MM, HH:MM:SS (24h and 12h with AM/PM)
    - Combined datetime formats

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with datetime columns converted to numeric timestamps
    """
    df_copy = df.copy()

    # Common date/time format patterns to try
    date_formats = [
        '%Y-%m-%d',           # YYYY-MM-DD
        '%Y/%m/%d',           # YYYY/MM/DD
        '%d/%m/%Y',           # DD/MM/YYYY
        '%m/%d/%Y',           # MM/DD/YYYY
        '%d-%m-%Y',           # DD-MM-YYYY
        '%m-%d-%Y',           # MM-DD-YYYY
        '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
        '%Y/%m/%d %H:%M:%S',  # YYYY/MM/DD HH:MM:SS
        '%d/%m/%Y %H:%M:%S',  # DD/MM/YYYY HH:MM:SS
        '%m/%d/%Y %H:%M:%S',  # MM/DD/YYYY HH:MM:SS
        '%Y-%m-%d %H:%M',     # YYYY-MM-DD HH:MM
        '%d/%m/%Y %H:%M',     # DD/MM/YYYY HH:MM
        '%m/%d/%Y %H:%M',     # MM/DD/YYYY HH:MM
        '%H:%M:%S',           # HH:MM:SS
        '%H:%M',              # HH:MM
        '%I:%M:%S %p',        # HH:MM:SS AM/PM
        '%I:%M %p',           # HH:MM AM/PM
    ]

    converted_cols = []

    for col in df_copy.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            continue

        # Skip if column has too many nulls (>50%)
        if df_copy[col].isnull().sum() / len(df_copy) > 0.5:
            continue

        # Try to convert with pandas auto-detection first
        try:
            # Attempt automatic datetime parsing
            temp_series = pd.to_datetime(df_copy[col], errors='coerce', infer_datetime_format=True)

            # Check if conversion was successful for most values (>70%)
            success_rate = temp_series.notna().sum() / len(temp_series)

            if success_rate > 0.7:
                # Convert to Unix timestamp (seconds since epoch) for numeric analysis
                df_copy[col] = temp_series.astype('int64') / 10**9  # Convert nanoseconds to seconds
                converted_cols.append(col)
                continue
        except:
            pass

        # If auto-detection failed, try specific formats
        converted = False
        for fmt in date_formats:
            try:
                temp_series = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                success_rate = temp_series.notna().sum() / len(temp_series)

                if success_rate > 0.7:
                    # Convert to Unix timestamp
                    df_copy[col] = temp_series.astype('int64') / 10**9
                    converted_cols.append(col)
                    converted = True
                    break
            except:
                continue

        # If still not converted, check for time-only formats
        if not converted:
            try:
                # Try parsing as time only, convert to seconds since midnight
                temp_series = pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.time
                if temp_series.notna().sum() / len(temp_series) > 0.7:
                    df_copy[col] = pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.hour * 3600 + \
                                   pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.minute * 60 + \
                                   pd.to_datetime(df_copy[col], format='%H:%M:%S', errors='coerce').dt.second
                    converted_cols.append(col)
            except:
                pass

    # Log conversion info if any columns were converted
    if converted_cols:
        st.info(f"📅 Detected and converted {len(converted_cols)} date/time column(s) to numeric: {', '.join(converted_cols)}")

    return df_copy
