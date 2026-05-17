import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

def find_optimal_clusters(X_scaled, max_k=8):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    # Choose optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, inertia, silhouette_scores, k_range

def train_segmentation_model(X, optimal_k):
    """Train the final segmentation model"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    segments = final_kmeans.fit_predict(X_scaled)
    
    return final_kmeans, scaler, segments, X_scaled

def save_model_artifacts(model, scaler, segment_mapping, file_paths):
    """Save model artifacts to files"""
    joblib.dump(model, file_paths['model'])
    joblib.dump(scaler, file_paths['scaler'])
    joblib.dump(segment_mapping, file_paths['mapping'])