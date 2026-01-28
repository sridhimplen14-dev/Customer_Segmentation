"""
Data preprocessing utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import detect_column_types


def create_preprocessing_pipeline(df, feature_cols, random_state=42):
    """
    Create a preprocessing pipeline for the selected features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to use
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    ColumnTransformer pipeline
    """
    col_types = detect_column_types(df)
    
    # Get numeric and categorical columns from selected features
    numeric_cols = [col for col in feature_cols if col in col_types['numeric']]
    categorical_cols = [col for col in feature_cols if col in col_types['categorical']]
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_cols, categorical_cols


def preprocess_data(df, feature_cols, preprocessor):
    """
    Preprocess data using the fitted preprocessor.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    preprocessor : ColumnTransformer
        Fitted preprocessor
    
    Returns:
    --------
    np.ndarray of transformed features
    """
    X = df[feature_cols].copy()
    X_transformed = preprocessor.transform(X)
    return X_transformed


def get_feature_names_after_encoding(preprocessor, numeric_cols, categorical_cols):
    """
    Get feature names after one-hot encoding.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    list of feature names
    """
    feature_names = []
    
    # Add numeric feature names
    feature_names.extend(numeric_cols)
    
    # Add categorical feature names (after one-hot encoding)
    if categorical_cols:
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'named_steps'):
            encoder = cat_transformer.named_steps['encoder']
            if hasattr(encoder, 'get_feature_names_out'):
                cat_features = encoder.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_features)
    
    return feature_names
