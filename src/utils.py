"""
Utility functions for column detection and feature recommendations
"""
import pandas as pd
import numpy as np


def detect_column_types(df):
    """
    Detect numeric, categorical, and columns to drop from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict with keys: 'numeric', 'categorical', 'drop', 'date'
    """
    numeric_cols = []
    categorical_cols = []
    drop_cols = []
    date_cols = []
    
    for col in df.columns:
        # Always drop ID column
        if col.upper() == 'ID':
            drop_cols.append(col)
            continue
        
        # Check for date columns
        if 'dt_' in col.lower() or 'date' in col.lower():
            date_cols.append(col)
            drop_cols.append(col)
            continue
        
        # Drop constant columns (Z_CostContact, Z_Revenue typically)
        if col.startswith('Z_'):
            drop_cols.append(col)
            continue
        
        # Check if numeric
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if it's actually categorical (low cardinality integers)
            if df[col].nunique() < 10 and df[col].nunique() < len(df) * 0.1:
                # Could be categorical, but for clustering, keep numeric
                numeric_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            # Assume categorical
            categorical_cols.append(col)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'drop': drop_cols,
        'date': date_cols
    }


def get_recommended_features(df):
    """
    Get recommended feature list for clustering.
    Includes demographics, spend, and activity columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    list of recommended column names
    """
    col_types = detect_column_types(df)
    recommended = []
    
    # Demographics
    demo_keywords = ['income', 'year_birth', 'age', 'kidhome', 'teenhome']
    for col in df.columns:
        if any(kw in col.lower() for kw in demo_keywords):
            if col not in col_types['drop']:
                recommended.append(col)
    
    # Spending columns
    for col in df.columns:
        if col.startswith('Mnt') or 'spend' in col.lower() or 'amount' in col.lower():
            if col not in col_types['drop']:
                recommended.append(col)
    
    # Activity columns
    activity_keywords = ['recency', 'num', 'purchase', 'visit', 'deal']
    for col in df.columns:
        if any(kw in col.lower() for kw in activity_keywords):
            if col not in col_types['drop']:
                recommended.append(col)
    
    # Campaign responses
    for col in df.columns:
        if 'accepted' in col.lower() or col.lower() == 'response':
            if col not in col_types['drop']:
                recommended.append(col)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result
