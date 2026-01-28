# Customer Segmentation (Clustering) - Streamlit App

A comprehensive Streamlit web application for customer segmentation using clustering algorithms. This app allows you to load customer data, preprocess it, train clustering models, visualize segments, profile customer groups, and predict segments for new customers.

## Features

- **Data Loading**: Load CSV files from path or upload directly
- **Data Preprocessing**: Automatic handling of missing values, encoding, and scaling
- **Model Selection**: Evaluate optimal number of clusters using elbow method and silhouette scores
- **Clustering**: Train KMeans or Agglomerative clustering models
- **Visualization**: 2D PCA projections of customer clusters
- **Segment Profiling**: Detailed analysis and naming of customer segments
- **Prediction**: Predict segment for new customers
- **Export**: Download segmented customer data as CSV

## Project Structure

```
Customer Segmentation/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_io.py             # Data loading utilities
│   ├── preprocess.py          # Data preprocessing
│   ├── modeling.py            # Clustering models
│   ├── viz.py                 # Visualizations
│   ├── profiling.py           # Segment profiling
│   ├── predict.py             # Prediction utilities
│   └── utils.py               # Utility functions
├── data/                       # Data directory
│   └── customer_segmentation.csv
├── artifacts/                  # Model artifacts
│   └── model.joblib           # Saved model (created by app)
└── downloads/                  # Export directory
    └── segmented_customers.csv # Exported data (created by app)
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your customer segmentation CSV file in the `data/` directory:

```bash
# The file should be named: customer_segmentation.csv
# Or update the default path in the app sidebar
```

**Expected CSV Format:**
- Contains an ID column (e.g., `ID`)
- Has numeric spend columns (e.g., `MntWines`, `MntFruits`, `MntMeatProducts`)
- Has categorical columns (e.g., `Education`, `Marital_Status`)
- Has campaign response columns (e.g., `AcceptedCmp1..5`, `Response`)
- May have a date column (e.g., `Dt_Customer`)
- May have missing values in `Income` or other columns

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Step 1: Load Data
1. In the sidebar, enter the path to your CSV file (default: `data/customer_segmentation.csv`)
2. Or upload a CSV file using the file uploader
3. Click "🔄 Load Data"

### Step 2: Configure Features
1. Choose feature selection mode:
   - **Recommended features**: Automatically selects demographics, spend, and activity columns
   - **Manual select features**: Manually choose which features to use
2. Select clustering algorithm (KMeans or Agglomerative)
3. Set K range for evaluation (min and max)
4. Choose training K (number of clusters)
5. Set random seed for reproducibility

### Step 3: Explore Data (Overview Tab)
- View dataset shape, data types, missing values
- See descriptive statistics
- Preview the data

### Step 4: Preprocess Data (Preprocessing Tab)
- Click "🔄 Run Preprocessing"
- View preprocessing steps applied:
  - Dropped ID and date columns
  - Median imputation for missing Income
  - One-hot encoding for categorical features
  - StandardScaler for numeric features

### Step 5: Select Optimal K (Model Selection Tab)
- Click "🔍 Compute K Metrics"
- Review elbow plot and silhouette scores
- App recommends best K based on highest silhouette score
- Examine metrics table for all K values

### Step 6: Train Model (Train & Visualize Tab)
- Click "🎓 Train Model"
- View cluster distribution
- See 2D PCA visualization of clusters
- Model is automatically saved to `artifacts/model.joblib`

### Step 7: Analyze Segments (Segment Profiles Tab)
- Click "🏷️ Generate Segment Names" to get automatic naming
- View segment summary with key metrics
- Select a cluster to see detailed statistics
- View sample customers in each segment

### Step 8: Predict New Customer (Predict Segment Tab)
- Fill in customer details in the form
- Click "🔮 Predict Segment"
- View predicted cluster and segment name

### Step 9: Export Results (Export Tab)
- Click "💾 Save to File" to save to `downloads/segmented_customers.csv`
- Use download button to get CSV file
- Preview exported data

## App Tabs Overview

### 📋 Overview
- Dataset shape and basic statistics
- Data types and missing values summary
- Descriptive statistics
- Data preview

### 🔧 Preprocessing
- Lists all preprocessing steps
- Shows feature matrix shape before and after encoding
- Displays numeric and categorical feature counts

### 🎯 Model Selection
- Computes inertia (SSE) and silhouette scores for K range
- Elbow plot (K vs inertia)
- Silhouette plot (K vs silhouette score)
- Recommends best K with explanation

### 🚀 Train & Visualize
- Trains clustering model with selected K
- Shows cluster distribution
- 2D PCA scatter plot colored by cluster
- Saves model pipeline to `artifacts/model.joblib`

### 👥 Segment Profiles
- Segment summary table with key metrics
- Detailed statistics per cluster
- Automatic segment naming based on characteristics
- Sample customers per segment
- Filter by cluster

### 🔮 Predict Segment
- Form to input new customer data
- Predicts cluster assignment
- Shows segment name
- Handles missing fields gracefully

### 💾 Export
- Adds cluster labels to original dataframe
- Adds segment names (if generated)
- Download button for CSV
- Saves to `downloads/segmented_customers.csv`

## Troubleshooting

### File Not Found Error
- **Issue**: "File not found: data/customer_segmentation.csv"
- **Solution**: 
  - Ensure the CSV file exists in the `data/` directory
  - Or upload the file using the file uploader in the sidebar
  - Or update the file path in the sidebar

### No Features Selected
- **Issue**: "Please select at least one feature"
- **Solution**: Select features using either "Recommended features" or "Manual select features" mode

### Preprocessing Error
- **Issue**: Error during preprocessing
- **Solution**: 
  - Ensure selected features exist in the dataset
  - Check for data type issues (non-numeric values in numeric columns)
  - Verify missing value patterns

### Model Training Error
- **Issue**: Error training model
- **Solution**: 
  - Ensure preprocessing has been completed first
  - Check that K value is within valid range (2 to number of samples)
  - Verify feature matrix is not empty

### Silhouette Score Not Available
- **Issue**: "Could not compute valid silhouette scores"
- **Solution**: 
  - This can happen if K equals number of samples or if clusters are too small
  - Try a different K range
  - Ensure you have enough data points

### Prediction Error
- **Issue**: Error predicting segment
- **Solution**: 
  - Ensure model has been trained and saved
  - Check that all required fields are filled in the form
  - Verify field names match the training data

## Technical Details

### Preprocessing Pipeline
- **Numeric Features**: Median imputation + StandardScaler
- **Categorical Features**: OneHotEncoder with `handle_unknown='ignore'`
- **Dropped Columns**: ID, date columns (Dt_Customer), constant columns (Z_*)

### Clustering Algorithms
- **KMeans**: Default algorithm with k-means++ initialization
- **Agglomerative**: Hierarchical clustering (optional)

### Model Persistence
- Full pipeline (preprocessor + model) saved using joblib
- Location: `artifacts/model.joblib`
- Automatically loaded for predictions

### Segment Naming Logic
Segments are named based on:
- Income level (High-Value, Mid-Value, Budget)
- Spending behavior (Spenders, Savers)
- Channel preference (Web-Focused, Store-Loyal)
- Engagement level (Active, Inactive)

## Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (clustering, preprocessing)
- **matplotlib**: Plotting and visualization
- **joblib**: Model persistence

## License

This project is open source and available for educational and commercial use.

## Support

For issues or questions, please check the troubleshooting section or review the code comments in the source files.
