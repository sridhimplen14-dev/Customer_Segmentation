"""
Customer Segmentation Streamlit App
Main application file for clustering and segmenting customers
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_io import load_data, save_clustered_data
from src.preprocess import create_preprocessing_pipeline, preprocess_data
from src.modeling import (
    compute_k_metrics, train_kmeans, train_agglomerative, 
    create_full_pipeline, save_model, load_model
)
from src.viz import plot_elbow_silhouette, plot_clusters_2d
from src.profiling import create_segment_profiles, suggest_segment_names
from src.predict import predict_customer_segment
from src.utils import detect_column_types, get_recommended_features
from sklearn.decomposition import PCA

# Page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'X_transformed' not in st.session_state:
    st.session_state.X_transformed = None
if 'segment_names' not in st.session_state:
    st.session_state.segment_names = {}


def main():
    st.title("📊 Customer Segmentation (Clustering)")
    st.markdown("---")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File input
        file_path = st.text_input(
            "CSV File Path",
            value="data/customer_segmentation.csv",
            help="Path to the customer segmentation CSV file"
        )
        
        uploaded_file = st.file_uploader(
            "Or Upload CSV File",
            type=['csv'],
            help="Upload a CSV file to override the file path"
        )
        
        # Load data button
        if st.button("🔄 Load Data", type="primary"):
            if uploaded_file is not None:
                st.session_state.df = load_data(uploaded_file=uploaded_file)
            elif file_path:
                st.session_state.df = load_data(file_path=file_path)
            else:
                st.error("Please provide a file path or upload a file")
        
        if st.session_state.df is not None:
            st.success(f"✅ Data loaded: {len(st.session_state.df)} rows")
            
            # Feature selection mode
            feature_mode = st.radio(
                "Feature Selection Mode",
                ["Recommended features", "Manual select features"],
                help="Choose how to select features for clustering"
            )
            
            col_types = detect_column_types(st.session_state.df)
            all_features = [col for col in st.session_state.df.columns 
                          if col not in col_types['drop']]
            
            if feature_mode == "Recommended features":
                recommended = get_recommended_features(st.session_state.df)
                st.session_state.feature_cols = [f for f in recommended if f in all_features]
                st.info(f"Using {len(st.session_state.feature_cols)} recommended features")
            else:
                st.session_state.feature_cols = st.multiselect(
                    "Select Features",
                    options=all_features,
                    default=st.session_state.feature_cols if st.session_state.feature_cols else [],
                    help="Select features to use for clustering"
                )
            
            # Algorithm selection
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["KMeans", "Agglomerative"],
                help="Choose clustering algorithm"
            )
            
            # K range
            col1, col2 = st.columns(2)
            with col1:
                min_k = st.number_input("Min K", min_value=2, max_value=10, value=2)
            with col2:
                max_k = st.number_input("Max K", min_value=2, max_value=20, value=10)
            
            # Training K
            training_k = st.slider(
                "Training K",
                min_value=min_k,
                max_value=max_k,
                value=4,
                help="Number of clusters to use for training"
            )
            
            # Random seed
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=1000,
                value=42,
                help="Random seed for reproducibility"
            )
    
    # Main content area
    if st.session_state.df is None:
        st.info("👈 Please load data using the sidebar controls")
        return
    
    if not st.session_state.feature_cols:
        st.warning("⚠️ Please select at least one feature")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", "Preprocessing", "Model Selection", 
        "Train & Visualize", "Segment Profiles", "Predict Segment", "Export"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            missing_count = st.session_state.df.isnull().sum().sum()
            st.metric("Missing Values", missing_count)
        
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.astype(str),
            'Non-Null Count': st.session_state.df.count().values,
            'Null Count': st.session_state.df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        st.subheader("Missing Values Summary")
        missing_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Missing Count': st.session_state.df.isnull().sum().values,
            'Missing %': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.info("No missing values found")
        
        st.subheader("Descriptive Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Tab 2: Preprocessing
    with tab2:
        st.header("🔧 Data Preprocessing")
        
        if st.button("🔄 Run Preprocessing", type="primary"):
            with st.spinner("Preprocessing data..."):
                try:
                    preprocessor, numeric_cols, categorical_cols = create_preprocessing_pipeline(
                        st.session_state.df,
                        st.session_state.feature_cols,
                        random_state=int(random_seed)
                    )
                    
                    # Fit preprocessor
                    X = st.session_state.df[st.session_state.feature_cols].copy()
                    preprocessor.fit(X)
                    
                    # Transform data
                    X_transformed = preprocess_data(st.session_state.df, st.session_state.feature_cols, preprocessor)
                    
                    st.session_state.preprocessor = preprocessor
                    st.session_state.X_transformed = X_transformed
                    
                    st.success("✅ Preprocessing completed!")
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
        
        if st.session_state.preprocessor is not None:
            st.subheader("Preprocessing Steps Applied")
            steps = [
                "✅ Dropped non-feature columns: ID and Dt_Customer",
                "✅ Handled missing Income with median imputation",
                "✅ One-hot encoded categorical columns",
                "✅ Scaled numeric features with StandardScaler"
            ]
            for step in steps:
                st.markdown(f"- {step}")
            
            st.subheader("Preprocessing Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Features", len(st.session_state.feature_cols))
            with col2:
                st.metric("Transformed Features", st.session_state.X_transformed.shape[1])
            
            st.info(f"Feature matrix shape: {st.session_state.X_transformed.shape}")
            
            # Show feature breakdown
            col_types = detect_column_types(st.session_state.df)
            numeric_cols = [col for col in st.session_state.feature_cols if col in col_types['numeric']]
            categorical_cols = [col for col in st.session_state.feature_cols if col in col_types['categorical']]
            
            st.write(f"**Numeric features:** {len(numeric_cols)}")
            st.write(f"**Categorical features:** {len(categorical_cols)}")
            st.write(f"**After one-hot encoding:** {st.session_state.X_transformed.shape[1]} features")
        else:
            st.info("👆 Click 'Run Preprocessing' to start")
    
    # Tab 3: Model Selection
    with tab3:
        st.header("🎯 Model Selection (K Selection)")
        
        if st.session_state.X_transformed is None:
            st.warning("⚠️ Please run preprocessing first")
        else:
            if st.button("🔍 Compute K Metrics", type="primary"):
                with st.spinner("Computing metrics for different K values..."):
                    try:
                        metrics = compute_k_metrics(
                            st.session_state.X_transformed,
                            min_k=int(min_k),
                            max_k=int(max_k),
                            random_state=int(random_seed)
                        )
                        st.session_state.k_metrics = metrics
                        st.success("✅ Metrics computed!")
                    except Exception as e:
                        st.error(f"Error computing metrics: {str(e)}")
            
            if 'k_metrics' in st.session_state:
                metrics = st.session_state.k_metrics
                
                # Plot metrics
                fig = plot_elbow_silhouette(
                    metrics['k_values'],
                    metrics['inertias'],
                    metrics['silhouettes']
                )
                st.pyplot(fig)
                
                # Find best K
                valid_sil = [(k, s) for k, s in zip(metrics['k_values'], metrics['silhouettes']) 
                           if not np.isnan(s)]
                if valid_sil:
                    best_k, best_sil = max(valid_sil, key=lambda x: x[1])
                    st.success(f"🎯 **Recommended K: {best_k}** (Silhouette Score: {best_sil:.4f})")
                    st.info("The best K is selected based on the highest silhouette score, which measures how similar objects are to their own cluster compared to other clusters.")
                else:
                    st.warning("Could not compute valid silhouette scores")
                
                # Show metrics table
                st.subheader("Metrics Table")
                metrics_df = pd.DataFrame({
                    'K': metrics['k_values'],
                    'Inertia (SSE)': metrics['inertias'],
                    'Silhouette Score': metrics['silhouettes']
                })
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("👆 Click 'Compute K Metrics' to analyze different K values")
    
    # Tab 4: Train & Visualize
    with tab4:
        st.header("🚀 Train Model & Visualize")
        
        if st.session_state.X_transformed is None:
            st.warning("⚠️ Please run preprocessing first")
        else:
            if st.button("🎓 Train Model", type="primary"):
                with st.spinner("Training clustering model..."):
                    try:
                        if algorithm == "KMeans":
                            model = train_kmeans(
                                st.session_state.X_transformed,
                                n_clusters=int(training_k),
                                random_state=int(random_seed)
                            )
                        else:
                            model = train_agglomerative(
                                st.session_state.X_transformed,
                                n_clusters=int(training_k)
                            )
                        
                        # Get cluster labels
                        labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(st.session_state.X_transformed)
                        st.session_state.cluster_labels = labels
                        st.session_state.model = model
                        
                        # Create and save full pipeline
                        pipeline = create_full_pipeline(st.session_state.preprocessor, model)
                        st.session_state.pipeline = pipeline
                        
                        # Save model
                        save_model(pipeline, "artifacts/model.joblib")
                        
                        st.success("✅ Model trained and saved!")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
            if st.session_state.cluster_labels is not None:
                # Cluster counts
                st.subheader("Cluster Distribution")
                cluster_counts = pd.Series(st.session_state.cluster_labels).value_counts().sort_index()
                col1, col2, col3, col4 = st.columns(4)
                for i, (cluster_id, count) in enumerate(cluster_counts.items()):
                    with [col1, col2, col3, col4][i % 4]:
                        st.metric(f"Cluster {cluster_id}", count)
                
                # Visualization
                st.subheader("Cluster Visualization (2D PCA)")
                fig = plot_clusters_2d(
                    st.session_state.X_transformed,
                    st.session_state.cluster_labels,
                    int(training_k)
                )
                st.pyplot(fig)
            else:
                st.info("👆 Click 'Train Model' to train the clustering model")
    
    # Tab 5: Segment Profiles
    with tab5:
        st.header("👥 Segment Profiles")
        
        if st.session_state.cluster_labels is None:
            st.warning("⚠️ Please train the model first")
        else:
            # Create profiles first (adds Cluster column needed for naming)
            profiles_df, df_profiled = create_segment_profiles(
                st.session_state.df,
                st.session_state.cluster_labels,
                st.session_state.feature_cols
            )

            # Generate segment names
            if st.button("🏷️ Generate Segment Names", type="primary"):
                with st.spinner("Analyzing segments..."):
                    for cluster_id in sorted(np.unique(st.session_state.cluster_labels)):
                        name = suggest_segment_names(df_profiled, cluster_id)
                        st.session_state.segment_names[cluster_id] = name
                    st.success("✅ Segment names generated!")
            
            st.subheader("Segment Summary")
            summary_data = []
            for cluster_id in sorted(np.unique(st.session_state.cluster_labels)):
                cluster_data = df_profiled[df_profiled['Cluster'] == cluster_id]
                summary = {
                    'Cluster': cluster_id,
                    'Segment Name': st.session_state.segment_names.get(cluster_id, f"Segment {cluster_id}"),
                    'Count': len(cluster_data)
                }
                
                # Add key metrics
                if 'Income' in cluster_data.columns:
                    summary['Avg Income'] = cluster_data['Income'].mean()
                if 'Recency' in cluster_data.columns:
                    summary['Avg Recency'] = cluster_data['Recency'].mean()
                
                spend_cols = [col for col in cluster_data.columns if col.startswith('Mnt')]
                if spend_cols:
                    summary['Total Spend'] = cluster_data[spend_cols].sum(axis=1).mean()
                
                summary_data.append(summary)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed profiles
            st.subheader("Detailed Segment Profiles")
            selected_cluster = st.selectbox(
                "Select Cluster to View",
                options=sorted(np.unique(st.session_state.cluster_labels)),
                format_func=lambda x: f"Cluster {x}: {st.session_state.segment_names.get(x, 'Unnamed')}"
            )
            
            cluster_data = df_profiled[df_profiled['Cluster'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Customers in Segment", len(cluster_data))
            with col2:
                st.metric("Segment Name", st.session_state.segment_names.get(selected_cluster, "Unnamed"))
            
            # Key statistics
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Cluster']
            
            if numeric_cols:
                st.write("**Key Statistics:**")
                stats_df = cluster_data[numeric_cols].describe().T
                st.dataframe(stats_df, use_container_width=True)
            
            # Sample customers
            st.subheader("Sample Customers")
            st.dataframe(cluster_data.head(10), use_container_width=True)
    
    # Tab 6: Predict Segment
    with tab6:
        st.header("🔮 Predict Segment for New Customer")
        
        # Check if model exists
        pipeline = load_model("artifacts/model.joblib")
        if pipeline is None:
            st.warning("⚠️ No trained model found. Please train a model first.")
        else:
            st.info("Enter customer details below to predict their segment")
            
            # Create form
            with st.form("customer_form"):
                col1, col2 = st.columns(2)
                
                customer_data = {}
                
                with col1:
                    if 'Income' in st.session_state.df.columns:
                        customer_data['Income'] = st.number_input("Income", min_value=0.0, value=50000.0)
                    
                    if 'Recency' in st.session_state.df.columns:
                        customer_data['Recency'] = st.number_input("Recency", min_value=0, value=30)
                    
                    if 'MntWines' in st.session_state.df.columns:
                        customer_data['MntWines'] = st.number_input("MntWines", min_value=0.0, value=100.0)
                    
                    if 'MntMeatProducts' in st.session_state.df.columns:
                        customer_data['MntMeatProducts'] = st.number_input("MntMeatProducts", min_value=0.0, value=50.0)
                
                with col2:
                    if 'NumWebPurchases' in st.session_state.df.columns:
                        customer_data['NumWebPurchases'] = st.number_input("NumWebPurchases", min_value=0, value=3)
                    
                    if 'NumStorePurchases' in st.session_state.df.columns:
                        customer_data['NumStorePurchases'] = st.number_input("NumStorePurchases", min_value=0, value=5)
                    
                    if 'NumWebVisitsMonth' in st.session_state.df.columns:
                        customer_data['NumWebVisitsMonth'] = st.number_input("NumWebVisitsMonth", min_value=0, value=4)
                    
                    if 'Education' in st.session_state.df.columns:
                        edu_options = st.session_state.df['Education'].unique().tolist()
                        customer_data['Education'] = st.selectbox("Education", options=edu_options)
                    
                    if 'Marital_Status' in st.session_state.df.columns:
                        marital_options = st.session_state.df['Marital_Status'].unique().tolist()
                        customer_data['Marital_Status'] = st.selectbox("Marital_Status", options=marital_options)
                
                # Fill missing columns with defaults
                for col in st.session_state.feature_cols:
                    if col not in customer_data:
                        if col in st.session_state.df.columns:
                            if st.session_state.df[col].dtype in ['int64', 'float64']:
                                customer_data[col] = st.session_state.df[col].median()
                            else:
                                customer_data[col] = st.session_state.df[col].mode()[0] if len(st.session_state.df[col].mode()) > 0 else None
                
                submitted = st.form_submit_button("🔮 Predict Segment", type="primary")
                
                if submitted:
                    try:
                        # Create dataframe
                        customer_df = pd.DataFrame([customer_data])
                        
                        # Predict
                        predicted_cluster = predict_customer_segment(customer_df, pipeline)
                        
                        # Get segment name
                        segment_name = st.session_state.segment_names.get(predicted_cluster, f"Segment {predicted_cluster}")
                        
                        st.success(f"✅ **Predicted Cluster: {predicted_cluster}**")
                        st.info(f"📌 **Segment Name: {segment_name}**")
                        
                        # Show customer data
                        st.subheader("Customer Data")
                        st.dataframe(customer_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error predicting segment: {str(e)}")
    
    # Tab 7: Export
    with tab7:
        st.header("💾 Export Results")
        
        if st.session_state.cluster_labels is None:
            st.warning("⚠️ Please train the model first")
        else:
            # Add cluster labels to dataframe
            df_export = st.session_state.df.copy()
            df_export['Cluster'] = st.session_state.cluster_labels
            
            # Add segment names if available
            if st.session_state.segment_names:
                df_export['Segment_Name'] = df_export['Cluster'].map(st.session_state.segment_names)
            
            st.subheader("Export Options")
            
            # Save to file
            output_path = "downloads/segmented_customers.csv"
            if st.button("💾 Save to File", type="primary"):
                if save_clustered_data(df_export, output_path):
                    st.success(f"✅ Data saved to {output_path}")
            
            # Download button
            st.subheader("Download CSV")
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Segmented Customers CSV",
                data=csv,
                file_name="segmented_customers.csv",
                mime="text/csv"
            )
            
            # Preview
            st.subheader("Preview Exported Data")
            st.dataframe(df_export.head(20), use_container_width=True)
            
            st.info(f"Total rows: {len(df_export)} | Total columns: {len(df_export.columns)}")


if __name__ == "__main__":
    main()
