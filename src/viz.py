"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_elbow_silhouette(k_values, inertias, silhouettes):
    """
    Plot elbow and silhouette plots side by side.
    
    Parameters:
    -----------
    k_values : list
        List of K values
    inertias : list
        List of inertia values
    silhouettes : list
        List of silhouette scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (SSE)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Silhouette plot
    valid_sil = [(k, s) for k, s in zip(k_values, silhouettes) if not np.isnan(s)]
    if valid_sil:
        k_vals, sil_vals = zip(*valid_sil)
        ax2.plot(k_vals, sil_vals, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_vals)
    else:
        ax2.text(0.5, 0.5, 'No valid silhouette scores', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_clusters_2d(X, labels, n_clusters):
    """
    Plot clusters in 2D using PCA projection.
    
    Parameters:
    -----------
    X : np.ndarray
        Preprocessed feature matrix
    labels : np.ndarray
        Cluster labels
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    matplotlib figure
    """
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[colors[i]], label=f'Cluster {i}', 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
                  fontsize=11)
    ax.set_ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
                  fontsize=11)
    ax.set_title('Customer Clusters (2D PCA Projection)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
