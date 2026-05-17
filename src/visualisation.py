import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_segment_distribution(segment_counts, segment_names):
    """Plot segment distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax.bar(segment_names, segment_counts, color=colors[:len(segment_counts)])
    ax.set_title('Customer Segment Distribution')
    ax.set_ylabel('Number of Customers')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_rfm_scatter(df, x_col, y_col, color_col, title):
    """Create RFM scatter plot"""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=title,
        hover_data=['frequency', 'age', 'shopping_mall']
    )
    return fig