"""
Prediction utilities for new customers
"""
import pandas as pd
import numpy as np


def create_customer_input_form(df, available_fields):
    """
    Create a dictionary of customer inputs from form.
    This is a helper to structure the input data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe (for reference)
    available_fields : list
        List of available field names
    
    Returns:
    --------
    dict: Customer data dictionary
    """
    # This function is mainly for documentation
    # Actual form will be created in app.py
    pass


def predict_customer_segment(customer_data, pipeline):
    """
    Predict cluster segment for a new customer.
    
    Parameters:
    -----------
    customer_data : pd.DataFrame
        Single-row dataframe with customer features
    pipeline : Pipeline
        Trained pipeline (preprocessor + model)
    
    Returns:
    --------
    int: Predicted cluster label
    """
    try:
        # Pipeline will handle preprocessing and prediction
        cluster_label = pipeline.predict(customer_data)
        return cluster_label[0]
    except Exception as e:
        raise Exception(f"Error predicting segment: {str(e)}")
