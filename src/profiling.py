"""
Segment profiling utilities
"""
import pandas as pd
import numpy as np


def create_segment_profiles(df, cluster_labels, feature_cols):
    """
    Create detailed profiles for each cluster segment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    cluster_labels : np.ndarray
        Cluster labels for each customer
    feature_cols : list
        List of feature column names used
    
    Returns:
    --------
    pd.DataFrame with segment profiles
    """
    df_profiled = df.copy()
    df_profiled['Cluster'] = cluster_labels
    
    # Get numeric and categorical columns
    numeric_cols = df_profiled.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Cluster']
    
    # Build profile summary
    profiles = []
    
    for cluster_id in sorted(df_profiled['Cluster'].unique()):
        cluster_data = df_profiled[df_profiled['Cluster'] == cluster_id]
        
        profile = {
            'Cluster': cluster_id,
            'Count': len(cluster_data)
        }
        
        # Add mean/median for numeric columns
        for col in numeric_cols:
            if col in cluster_data.columns:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_median'] = cluster_data[col].median()
        
        # Add campaign response rates if present
        campaign_cols = [col for col in df_profiled.columns 
                        if 'accepted' in col.lower() or col.lower() == 'response']
        for col in campaign_cols:
            if col in cluster_data.columns:
                profile[f'{col}_rate'] = cluster_data[col].mean()
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    return profiles_df, df_profiled


def suggest_segment_names(df_profiled, cluster_id):
    """
    Suggest a segment name based on cluster characteristics.
    
    Parameters:
    -----------
    df_profiled : pd.DataFrame
        Dataframe with cluster labels
    cluster_id : int
        Cluster ID to name
    
    Returns:
    --------
    str: Suggested segment name
    """
    if 'Cluster' not in df_profiled.columns:
        return f"Segment {cluster_id}"

    cluster_data = df_profiled[df_profiled['Cluster'] == cluster_id]
    if cluster_data.empty:
        return f"Segment {cluster_id}"
    
    # Calculate key metrics
    spend_cols = [col for col in df_profiled.columns if col.startswith('Mnt')]
    total_spend = cluster_data[spend_cols].sum(axis=1).mean() if spend_cols else 0
    
    num_purchase_cols = [col for col in df_profiled.columns if col.startswith('Num')]
    total_purchases = cluster_data[num_purchase_cols].sum(axis=1).mean() if num_purchase_cols else 0
    
    income = cluster_data['Income'].mean() if 'Income' in cluster_data.columns else 0
    
    web_purchases = cluster_data['NumWebPurchases'].mean() if 'NumWebPurchases' in cluster_data.columns else 0
    store_purchases = cluster_data['NumStorePurchases'].mean() if 'NumStorePurchases' in cluster_data.columns else 0
    
    recency = cluster_data['Recency'].mean() if 'Recency' in cluster_data.columns else 0
    
    # Compare to overall averages
    overall_spend = df_profiled[spend_cols].sum(axis=1).mean() if spend_cols else 0
    overall_income = df_profiled['Income'].mean() if 'Income' in df_profiled.columns else 0
    overall_web = df_profiled['NumWebPurchases'].mean() if 'NumWebPurchases' in df_profiled.columns else 0
    overall_store = df_profiled['NumStorePurchases'].mean() if 'NumStorePurchases' in df_profiled.columns else 0
    
    # Build name based on characteristics
    name_parts = []
    
    # Income level
    if income > overall_income * 1.2:
        name_parts.append("High-Value")
    elif income < overall_income * 0.8:
        name_parts.append("Budget")
    else:
        name_parts.append("Mid-Value")
    
    # Spending behavior
    if total_spend > overall_spend * 1.3:
        name_parts.append("Spenders")
    elif total_spend < overall_spend * 0.7:
        name_parts.append("Savers")
    
    # Channel preference
    if web_purchases > overall_web * 1.5 and web_purchases > store_purchases:
        name_parts.append("Web-Focused")
    elif store_purchases > overall_store * 1.5:
        name_parts.append("Store-Loyal")
    
    # Recency (engagement)
    if recency < 30:
        name_parts.append("Active")
    elif recency > 60:
        name_parts.append("Inactive")
    
    # Default if no strong characteristics
    if len(name_parts) == 0:
        return f"Segment {cluster_id}"
    
    return " ".join(name_parts[:3])  # Limit to 3 parts max
