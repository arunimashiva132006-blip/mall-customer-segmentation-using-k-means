import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .segment-card {
        background-color: #FF53cd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header"> Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Enhanced Analysis with RFM + Behavioral Features")

# Load the segmented data with behavioral features
@st.cache_data
def load_data():
    try:
        # Try to load enhanced data first
        df = pd.read_csv('../data/processed/segmented_customers_with_behavior.csv')
        st.success(" Loaded enhanced dataset with behavioral features")
        return df
    except:
        # Fall back to basic data
        df = pd.read_csv('../data/processed/segmented_customers.csv')
        st.info("ℹ Using basic RFM dataset")
        return df

df = load_data()

# Check if behavioral features exist
has_behavioral_features = 'category_variety' in df.columns

# Sidebar for filters
st.sidebar.header(" Dashboard Filters")

# Segment selection
selected_segments = st.sidebar.multiselect(
    "Select Segments to Display:",
    options=sorted(df['segment'].unique()),
    default=sorted(df['segment'].unique())
)

# Shopping mall filter
mall_options = ['All'] + sorted(df['shopping_mall'].unique().tolist())
selected_mall = st.sidebar.selectbox("Filter by Shopping Mall:", mall_options)

# Apply filters
filtered_df = df[df['segment'].isin(selected_segments)]
if selected_mall != 'All':
    filtered_df = filtered_df[filtered_df['shopping_mall'] == selected_mall]

# Main dashboard layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Customers", f"{len(filtered_df):,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_spending = filtered_df['monetary_inr'].mean()
    st.metric("Average Spending", f"₹{avg_spending:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_recency = filtered_df['recency'].mean()
    st.metric("Average Recency", f"{avg_recency:.0f} days")
    st.markdown('</div>', unsafe_allow_html=True)

# Segment distribution
st.markdown("---")
st.subheader(" Customer Segments Overview")

col1, col2 = st.columns(2)

# IMPROVED DYNAMIC SEGMENT NAMES - Ensures unique names for all segments
def get_segment_names(segment_df):
    segment_profiles = {}
    segments = sorted(segment_df['segment'].unique())
    
    # Calculate metrics for all segments
    segment_metrics = []
    for segment in segments:
        seg_data = segment_df[segment_df['segment'] == segment]
        metrics = {
            'segment': segment,
            'recency': seg_data['recency'].mean(),
            'spending': seg_data['monetary_inr'].mean(),
            'frequency': seg_data['frequency'].mean(),
            'size': len(seg_data),
            'size_pct': len(seg_data) / len(segment_df) * 100
        }
        segment_metrics.append(metrics)
    
    # Sort by spending (highest first) for priority naming
    segment_metrics.sort(key=lambda x: x['spending'], reverse=True)
    
    # Assign unique names based on combined characteristics
    name_templates = [
        " Premium High-Spenders",      # Highest spending
        " Strategic Shoppers",         # Second highest spending
        " Regular Value Shoppers",     # Medium spending
        " Budget Conscious",           # Lower spending
        " At-Risk Customers",          # High recency
        " Recent Engaged",             # Low recency, newer
        " Growth Potential",           # Medium everything
        " Diverse Shoppers"           # High category variety
    ]
    
    for i, metrics in enumerate(segment_metrics):
        segment = metrics['segment']
        
        if i < len(name_templates):
            # Use predefined names in order of spending priority
            segment_profiles[segment] = name_templates[i]
        else:
            # Fallback for more segments than templates
            if metrics['recency'] > segment_df['recency'].quantile(0.75):
                segment_profiles[segment] = f"⚠️ Inactive Segment {segment}"
            elif metrics['spending'] > segment_df['monetary_inr'].quantile(0.75):
                segment_profiles[segment] = f"💰 High-Value Segment {segment}"
            else:
                segment_profiles[segment] = f"🛒 Segment {segment}"
    
    return segment_profiles

segment_names = get_segment_names(df)

with col1:
    # Segment distribution pie chart
    segment_counts = filtered_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    segment_counts['segment_name'] = segment_counts['segment'].map(segment_names)
    
    fig_pie = px.pie(
        segment_counts, 
        values='count', 
        names='segment_name',
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # RFM scatter plot
    fig_scatter = px.scatter(
        filtered_df,
        x='recency',
        y='monetary_inr',
        color='segment',
        title="RFM Analysis: Recency vs Monetary Value",
        labels={
            'recency': 'Days Since Last Purchase',
            'monetary_inr': 'Total Spending (₹)',
            'segment': 'Customer Segment'
        },
        color_continuous_scale='viridis',
        hover_data=['frequency', 'age', 'shopping_mall']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Behavioral Analysis Section
if has_behavioral_features:
    st.markdown("---")
    st.subheader(" Behavioral Analysis")
    
    # Get actual segments from data
    actual_segments = sorted(filtered_df['segment'].unique())
    segment_display_names = []
    
    for seg in actual_segments:
        if seg == 0:
            segment_display_names.append(" Loyal High-Spenders")
        elif seg == 1:
            segment_display_names.append(" At-Risk Customers")
        elif seg == 2:
            segment_display_names.append(" Budget Shoppers")
        else:
            segment_display_names.append(f"Segment {seg}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category variety by segment
        fig_category = px.box(
            filtered_df, 
            x='segment', 
            y='category_variety',
            title="Category Variety by Segment",
            labels={'segment': 'Customer Segment', 'category_variety': 'Number of Categories Shopped'}
        )
        fig_category.update_xaxes(ticktext=segment_display_names, tickvals=actual_segments)
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Payment flexibility by segment
        fig_payment = px.box(
            filtered_df,
            x='segment',
            y='payment_flexibility', 
            title="Payment Flexibility by Segment",
            labels={'segment': 'Customer Segment', 'payment_flexibility': 'Number of Payment Methods Used'}
        )
        fig_payment.update_xaxes(ticktext=segment_display_names, tickvals=actual_segments)
        st.plotly_chart(fig_payment, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Shopping cart size by segment
        fig_cart = px.box(
            filtered_df,
            x='segment',
            y='avg_items_per_order',
            title="Average Items per Order by Segment",
            labels={'segment': 'Customer Segment', 'avg_items_per_order': 'Average Items per Order'}
        )
        fig_cart.update_xaxes(ticktext=segment_display_names, tickvals=actual_segments)
        st.plotly_chart(fig_cart, use_container_width=True)
    
    with col4:
        # Weekend shopping ratio
        fig_weekend = px.box(
            filtered_df,
            x='segment',
            y='weekend_ratio',
            title="Weekend Shopping Ratio by Segment",
            labels={'segment': 'Customer Segment', 'weekend_ratio': 'Ratio of Weekend Shopping'}
        )
        fig_weekend.update_xaxes(ticktext=segment_display_names, tickvals=actual_segments)
        st.plotly_chart(fig_weekend, use_container_width=True)

# Segment details
st.markdown("---")
st.subheader(" Segment Characteristics")

# Create columns for each segment
segments = sorted(filtered_df['segment'].unique())
cols = st.columns(len(segments))

for i, segment in enumerate(segments):
    with cols[i]:
        seg_data = filtered_df[filtered_df['segment'] == segment]
        segment_name = segment_names[segment]
        
       
        st.markdown(f"### {segment_name}")
        st.markdown(f"**Customers:** {len(seg_data):,}")
        st.markdown(f"**Avg Recency:** {seg_data['recency'].mean():.0f} days")
        st.markdown(f"**Avg Spending:** ₹{seg_data['monetary_inr'].mean():.0f}")
        st.markdown(f"**Avg Age:** {seg_data['age'].mean():.0f} years")
        
        if has_behavioral_features:
            st.markdown(f"**Category Variety:** {seg_data['category_variety'].mean():.1f}")
            st.markdown(f"**Payment Methods:** {seg_data['payment_flexibility'].mean():.1f}")
            st.markdown(f"**Avg Cart Size:** {seg_data['avg_items_per_order'].mean():.1f} items")
            st.markdown(f"**Weekend Shopping:** {seg_data['weekend_ratio'].mean()*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Business insights
st.markdown("---")
st.subheader(" Enhanced Business Insights & Recommendations")

insights = {
    0: {
        "title": " Loyal High-Spenders",
        "description": "Your most valuable customers with high spending power",
        "basic_recommendations": [
            "VIP treatment & exclusive offers",
            "Early access to new products",
            "Personalized premium services"
        ],
        "behavioral_insights": [
            "Focus on their preferred categories for cross-selling",
            "Leverage their payment preferences for targeted promotions",
            "Use their shopping patterns for optimal campaign timing"
        ]
    },
    1: {
        "title": " At-Risk Customers", 
        "description": "Haven't purchased in a long time - high churn risk",
        "basic_recommendations": [
            "Win-back campaigns with special discounts",
            "Re-engagement emails with personalized offers",
            "Survey to understand why they left"
        ],
        "behavioral_insights": [
            "Analyze their last shopping patterns for clues",
            "Target based on their previous category preferences",
            "Consider their preferred shopping times for outreach"
        ]
    },
    2: {
        "title": " Budget Shoppers",
        "description": "Recent shoppers with moderate spending",
        "basic_recommendations": [
            "Loyalty programs to increase frequency",
            "Bundle deals and volume discounts", 
            "Regular engagement through newsletters"
        ],
        "behavioral_insights": [
            "Leverage their category variety for bundle opportunities",
            "Use their payment flexibility for payment-specific offers",
            "Target based on their consistent shopping patterns"
        ]
    }
}

for segment, info in insights.items():
    if segment in selected_segments:
        with st.expander(f"{info['title']} - {len(filtered_df[filtered_df['segment'] == segment]):,} customers"):
            st.write(info['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Core Recommendations:**")
                for rec in info['basic_recommendations']:
                    st.markdown(f"• {rec}")
            
            with col2:
                if has_behavioral_features:
                    st.markdown("**Behavioral Insights:**")
                    for insight in info['behavioral_insights']:
                        st.markdown(f"• {insight}")

# Additional visualizations
st.markdown("---")
st.subheader(" Detailed Analysis")

col1, col2 = st.columns(2)

with col1:
    # Age distribution by segment
    fig_age = px.box(
        filtered_df,
        x='segment',
        y='age',
        color='segment',
        title="Age Distribution by Segment",
        labels={'segment': 'Customer Segment', 'age': 'Age'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Update segment names on x-axis
    fig_age.update_xaxes(ticktext=[segment_names[i] for i in sorted(filtered_df['segment'].unique())],
                        tickvals=sorted(filtered_df['segment'].unique()))
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    # Shopping mall distribution
    mall_dist = filtered_df['shopping_mall'].value_counts().head(10)
    fig_mall = px.bar(
        x=mall_dist.values,
        y=mall_dist.index,
        orientation='h',
        title="Top 10 Shopping Malls by Customer Count",
        labels={'x': 'Number of Customers', 'y': 'Shopping Mall'},
        color=mall_dist.values,
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig_mall, use_container_width=True)

# Data export
st.markdown("---")
st.subheader(" Export Data")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label=" Download Segment Analysis",
        data=filtered_df.to_csv(index=False),
        file_name="customer_segments_analysis.csv",
        mime="text/csv"
    )

with col2:
    if st.button(" Refresh Analysis"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Enhanced Customer Segmentation Dashboard • Built with Streamlit • "
    f"Total Customers Analyzed: {len(df):,} • "
    f"{'With Behavioral Features' if has_behavioral_features else 'Basic RFM Analysis'}"
    "</div>",
    unsafe_allow_html=True
)