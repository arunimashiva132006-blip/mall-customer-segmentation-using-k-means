# save_model.py
import sys
import os
sys.path.append('..')

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    # Load your processed data
    df = pd.read_csv('../processed/rfm_data.csv')  # or cleaned_data.csv
    
    # Prepare features (adjust based on your actual columns)
    # Common RFM features:
    features = ['Recency', 'Frequency', 'Monetary']  # Update with your actual column names
    
    # If these columns don't exist, use whatever features you have
    if not all(col in df.columns for col in features):
        features = df.select_dtypes(include=['number']).columns.tolist()
        features = [f for f in features if f not in ['CustomerID', 'segment']]  # exclude IDs and targets
    
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a model
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_scaled)
    
    # Save the model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/trained_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    
    print(f"Model trained using features: {features}")
    print("Model saved successfully to ../models/trained_model.pkl")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have the processed data files in the '../processed/' folder")