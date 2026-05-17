import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(file_path):
    """Load and preprocess the customer shopping data"""
    df = pd.read_csv(file_path)
    
    # Convert date format
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    
    # Currency conversion to INR
    exchange_rate = 2.70
    df['price_inr'] = df['price'] * exchange_rate
    
    return df

def calculate_rfm_metrics(df):
    """Calculate RFM metrics from the dataset"""
    latest_date = df['invoice_date'].max()
    
    rfm_data = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (latest_date - x.max()).days,
        'invoice_no': 'count',
        'price_inr': 'sum'
    }).reset_index()
    
    rfm_data.columns = ['customer_id', 'recency', 'frequency', 'monetary_inr']
    
    # Add customer demographics
    customer_demographics = df[['customer_id', 'gender', 'age', 'shopping_mall']].drop_duplicates()
    rfm_full = rfm_data.merge(customer_demographics, on='customer_id', how='left')
    
    return rfm_full

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    clean_df = df.copy()
    for col in columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df