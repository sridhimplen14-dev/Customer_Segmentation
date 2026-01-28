"""
Data loading and I/O utilities
"""
import pandas as pd
import streamlit as st
import os


@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """
    Load customer segmentation data from file path or uploaded file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to CSV file
    uploaded_file : streamlit UploadedFile, optional
        Uploaded file object
    
    Returns:
    --------
    pd.DataFrame or None if error
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows from uploaded file")
            return df
        elif file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.success(f"Loaded {len(df)} rows from {file_path}")
            return df
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def save_clustered_data(df, output_path):
    """
    Save clustered dataframe to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels
    output_path : str
        Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False
