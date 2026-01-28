"""
Clustering model utilities
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.base import BaseEstimator, ClusterMixin


def compute_k_metrics(X, min_k=2, max_k=10, random_state=42):
    """
    Compute inertia and silhouette scores for different K values.
    
    Parameters:
    -----------
    X : np.ndarray
        Preprocessed feature matrix
    min_k : int
        Minimum number of clusters
    max_k : int
        Maximum number of clusters
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict with 'k_values', 'inertias', 'silhouettes'
    """
    k_values = list(range(min_k, max_k + 1))
    inertias = []
    silhouettes = []
    
    for k in k_values:
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            # Silhouette score (only if valid)
            if len(np.unique(labels)) > 1 and len(X) > k:
                try:
                    sil_score = silhouette_score(X, labels)
                    silhouettes.append(sil_score)
                except:
                    silhouettes.append(np.nan)
            else:
                silhouettes.append(np.nan)
        except Exception as e:
            inertias.append(np.nan)
            silhouettes.append(np.nan)
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes
    }


def train_kmeans(X, n_clusters, random_state=42):
    """
    Train KMeans model.
    
    Parameters:
    -----------
    X : np.ndarray
        Preprocessed feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    KMeans model
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def train_agglomerative(X, n_clusters):
    """
    Train Agglomerative Clustering model.
    
    Parameters:
    -----------
    X : np.ndarray
        Preprocessed feature matrix
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    AgglomerativeClustering model
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X)
    return model


class AgglomerativeWrapper(BaseEstimator, ClusterMixin):
    """Wrapper to make AgglomerativeClustering work in pipeline with predict()"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.kmeans_fallback = None
    
    def fit(self, X, y=None):
        self.base_model.fit(X)
        # Use KMeans as fallback for prediction on new data
        # since AgglomerativeClustering doesn't support predict() for new samples
        self.kmeans_fallback = KMeans(n_clusters=self.base_model.n_clusters, 
                                     random_state=42, n_init=10)
        self.kmeans_fallback.fit(X)
        return self
    
    def predict(self, X):
        # Use KMeans fallback for prediction
        if self.kmeans_fallback is not None:
            return self.kmeans_fallback.predict(X)
        else:
            return self.base_model.fit_predict(X)
    
    def fit_predict(self, X, y=None):
        return self.base_model.fit_predict(X)


def create_full_pipeline(preprocessor, model):
    """
    Create a full pipeline combining preprocessor and model.
    Note: AgglomerativeClustering doesn't support predict() for new data,
    so we use a wrapper that falls back to KMeans for predictions.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    model : sklearn model
        Trained clustering model
    
    Returns:
    --------
    Pipeline object
    """
    # Wrap AgglomerativeClustering if needed
    if isinstance(model, AgglomerativeClustering):
        model = AgglomerativeWrapper(model)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return full_pipeline


def save_model(pipeline, file_path):
    """
    Save model pipeline to file.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Full pipeline to save
    file_path : str
        Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(pipeline, file_path)
        return True
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")


def load_model(file_path):
    """
    Load model pipeline from file.
    
    Parameters:
    -----------
    file_path : str
        Model file path
    
    Returns:
    --------
    Pipeline object or None if error
    """
    try:
        if os.path.exists(file_path):
            pipeline = joblib.load(file_path)
            return pipeline
        else:
            return None
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
