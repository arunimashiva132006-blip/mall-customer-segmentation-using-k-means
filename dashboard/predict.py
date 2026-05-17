import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segment Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .segment-0 { background-color: #FF6B6B; color: white; padding: 1rem; border-radius: 10px; }
    .segment-1 { background-color: #4ECDC4; color: white; padding: 1rem; border-radius: 10px; }
    .segment-2 { background-color: #45B7D1; color: white; padding: 1rem; border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 style="text-align: center; color: #1f77b4;">Enhanced Customer Segment Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict segments using RFM + Behavioral features")

# Initialize model variables
model = None
scaler = None
segment_mapping = None

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Use relative path to go up one level from dashboard folder
        model_path = '../models/kmeans_model.pkl'
        scaler_path = '../models/scaler.pkl'
        mapping_path = '../models/segment_mapping.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        segment_mapping = joblib.load(mapping_path)
        
        st.success(" Enhanced models loaded successfully! (RFM + Behavioral)")
        return model, scaler, segment_mapping
        
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please make sure the model files exist in the models folder")
        return None, None, None

# Load models
model, scaler, segment_mapping = load_model()

# Segment information with behavioral insights
segment_info = {
    0: {
        "name": "Loyal High-Spenders",
        "description": "High-value customers who spend significantly more than average",
        "behavioral_traits": [
            "Shops multiple categories",
            "Prefers specific payment methods", 
            "Larger average order sizes",
            "Consistent shopping patterns"
        ],
        "strategy": "VIP treatment, exclusive offers, premium services"
    },
    1: {
        "name": "At-Risk Customers", 
        "description": "Customers who haven't purchased in a long time",
        "behavioral_traits": [
            "Limited recent activity",
            "May have changed shopping habits",
            "Specific category preferences",
            "Inactive shopping patterns"
        ],
        "strategy": "Win-back campaigns, special discounts, re-engagement"
    },
    2: {
        "name": "Budget Shoppers",
        "description": "Recent shoppers with average spending patterns",
        "behavioral_traits": [
            "Diverse category shopping",
            "Flexible payment methods",
            "Consistent order sizes", 
            "Regular shopping frequency"
        ],
        "strategy": "Loyalty programs, bundle deals, regular engagement"
    }
}

# Input form
st.markdown("---")
st.subheader(" Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("####  RFM Metrics")
    days_since_last_purchase = st.slider(
        "Days since last purchase:",
        min_value=0,
        max_value=800,
        value=200,
        help="How many days ago did the customer make their last purchase?"
    )
    
    purchase_frequency = st.selectbox(
        "Purchase frequency:",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=0,
        help="How often does the customer typically purchase? (number of purchases)"
    )

    total_spending = st.number_input(
        "Total spending (₹):",
        min_value=0,
        max_value=50000,
        value=5000,
        step=100,
        help="Customer's total spending in Indian Rupees"
    )

with col2:
    st.markdown("####  Demographics")
    customer_age = st.slider(
        "Customer age:",
        min_value=18,
        max_value=80,
        value=35,
        help="Customer's age"
    )

    shopping_mall = st.selectbox(
        "Preferred shopping mall:",
        options=['Kanyon', 'Mall of Istanbul', 'Metrocity', 'Metropol AVM', 'Istinye Park',
                'Viaport Outlet', 'Zorlu Center', 'Cevahir AVM', 'Forum Istanbul']
    )
    
    gender = st.radio(
        "Gender:",
        options=['Male', 'Female']
    )

# Behavioral Features Section
st.markdown("---")
st.markdown('<div class="behavioral-section">', unsafe_allow_html=True)
st.subheader(" Behavioral Patterns")
st.markdown("Provide additional behavioral insights for more accurate segmentation")

col3, col4 = st.columns(2)

with col3:
    category_variety = st.slider(
        "Category Variety:",
        min_value=1,
        max_value=10, 
        value=3,
        help="How many different product categories do they shop from?"
    )
    
    is_fashion_shopper = st.selectbox(
        "Fashion Shopper:",
        options=["Yes", "No"],
        index=0,
        help="Do they frequently shop fashion categories (Clothing, Shoes, Cosmetics)?"
    )

with col4:
    payment_flexibility = st.slider(
        "Payment Flexibility:",
        min_value=1,
        max_value=5,
        value=2,
        help="How many different payment methods do they use?"
    )
    
    avg_items_per_order = st.slider(
        "Average Items per Order:",
        min_value=1,
        max_value=20,
        value=5,
        help="Typical number of items in their shopping cart"
    )

col5, col6 = st.columns(2)

with col5:
    max_items_per_order = st.slider(
        "Maximum Items per Order:",
        min_value=1,
        max_value=50,
        value=8,
        help="Largest number of items they've purchased in a single order"
    )
    
    avg_order_value = st.number_input(
        "Average Order Value (₹):",
        min_value=0,
        max_value=10000,
        value=2500,
        step=100,
        help="Average spending per transaction"
    )

with col6:
    preferred_hour = st.slider(
        "Preferred Shopping Hour:",
        min_value=0,
        max_value=23,
        value=14,
        help="Typical time of day they shop (24-hour format)"
    )
    
    weekend_ratio = st.slider(
        "Weekend Shopping Ratio:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="What percentage of their shopping happens on weekends?",
        format="%.2f"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Convert fashion shopper to numeric
is_fashion_numeric = 1 if is_fashion_shopper == "Yes" else 0

# Prediction button
st.markdown("---")
if st.button(" Predict Customer Segment", use_container_width=True):
    if model is not None and scaler is not None:
        # Prepare input data with ALL features (RFM + Behavioral)
        input_data = np.array([[
            days_since_last_purchase,     # recency
            purchase_frequency,           # frequency  
            total_spending,               # monetary_inr
            category_variety,             # category_variety
            is_fashion_numeric,           # is_fashion_shopper
            payment_flexibility,          # payment_flexibility
            avg_items_per_order,          # avg_items_per_order
            max_items_per_order,          # max_items_per_order
            avg_order_value,              # avg_order_value
            preferred_hour,               # preferred_hour
            weekend_ratio                 # weekend_ratio
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        segment_data = segment_info[prediction]
        
        # Display prediction result
        st.markdown("---")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("##  Prediction Result")
        st.markdown(f"### {segment_data['name']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced segment details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("####  Customer Profile")
            st.write(f"**Segment:** {segment_data['name']}")
            st.write(f"**Description:** {segment_data['description']}")
            
            st.markdown("####  Behavioral Traits")
            for trait in segment_data['behavioral_traits']:
                st.markdown(f"• {trait}")
            
            # Display input summary
            st.markdown("####  Input Summary")
            st.write(f"• **RFM**: {days_since_last_purchase} days recency, {purchase_frequency} purchases, ₹{total_spending:,} spending")
            st.write(f"• **Behavioral**: {category_variety} categories, {payment_flexibility} payment methods")
            st.write(f"• **Shopping**: {avg_items_per_order} avg items, {weekend_ratio*100:.0f}% weekend shopping")
            st.write(f"• **Demographics**: {customer_age} years, {gender}, {shopping_mall}")
        
        with col2:
            st.markdown("####  Recommended Strategy")
            st.info(segment_data['strategy'])
            
            # Enhanced recommendations based on behavioral inputs
            st.markdown("####  Actionable Insights")
            
            if prediction == 0:  # High-Spenders
                if category_variety > 5:
                    st.success(" Cross-category premium bundles would work well")
                if avg_items_per_order > 10:
                    st.success(" Volume-based VIP discounts recommended")
                if weekend_ratio > 0.5:
                    st.info(" Weekend exclusive events would engage this customer")
                    
            elif prediction == 1:  # At-Risk
                if days_since_last_purchase > 600:
                    st.error(" Immediate win-back campaign needed!")
                elif category_variety < 3:
                    st.warning(" Target with their preferred category offers")
                else:
                    st.warning(" Personalized re-engagement based on past behavior")
                    
            else:  # Budget Shoppers
                if payment_flexibility > 2:
                    st.info(" Multiple payment method promotions would appeal")
                if avg_order_value < 1000:
                    st.info(" Small-ticket bundle deals recommended")
                else:
                    st.success(" Loyalty program with tiered benefits")
        
        # Probability scores
        st.markdown("---")
        st.subheader(" Segment Similarity Scores")
        
        # Calculate distances to all cluster centers
        distances = model.transform(input_scaled)[0]
        # Convert distances to similarity scores (closer = higher similarity)
        similarities = 1 / (1 + distances)
        similarities = similarities / similarities.sum()  # Normalize to percentages
        
        # Display similarity scores
        sim_cols = st.columns(len(segment_info))
        for i, (segment, data) in enumerate(segment_info.items()):
            with sim_cols[i]:
                similarity_pct = similarities[i] * 100
                st.metric(
                    label=data["name"],
                    value=f"{similarity_pct:.1f}%",
                    delta="High match" if similarity_pct > 40 else "Moderate match"
                )
        
        # Export prediction
        st.markdown("---")
        st.subheader(" Export Prediction")
        
        prediction_data = {
            'predicted_segment': prediction,
            'segment_name': segment_data['name'],
            'confidence': f"{similarities[prediction]*100:.1f}%",
            # RFM metrics
            'days_since_last_purchase': days_since_last_purchase,
            'purchase_frequency': purchase_frequency,
            'total_spending': total_spending,
            # Behavioral metrics
            'category_variety': category_variety,
            'is_fashion_shopper': is_fashion_shopper,
            'payment_flexibility': payment_flexibility,
            'avg_items_per_order': avg_items_per_order,
            'max_items_per_order': max_items_per_order,
            'avg_order_value': avg_order_value,
            'preferred_hour': preferred_hour,
            'weekend_ratio': weekend_ratio,
            # Demographics
            'age': customer_age,
            'shopping_mall': shopping_mall,
            'gender': gender
        }
        
        prediction_df = pd.DataFrame([prediction_data])
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="💾 Download Prediction Report",
                data=prediction_df.to_csv(index=False),
                file_name=f"enhanced_customer_prediction_{prediction}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button(" Predict Another Customer", use_container_width=True):
                st.rerun()
    
    else:
        st.error(" Model not loaded properly. Please check the model files.")

# Enhanced batch prediction section
st.markdown("---")
st.subheader(" Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file with multiple customers for batch prediction:",
    type=['csv'],
    help="File should contain all RFM + Behavioral features for accurate predictions"
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(batch_data)} customers")
        
        # Check required columns for enhanced model
        required_cols = [
            'days_since_last_purchase', 'purchase_frequency', 'total_spending',  # RFM
            'category_variety', 'is_fashion_shopper', 'payment_flexibility',     # Behavioral
            'avg_items_per_order', 'max_items_per_order', 'avg_order_value',     # Cart behavior
            'preferred_hour', 'weekend_ratio'                                    # Time behavior
        ]
        
        if all(col in batch_data.columns for col in required_cols):
            
            if st.button("🔮 Predict All Customers", use_container_width=True):
                # Prepare data
                X_batch = batch_data[required_cols]
                X_batch_scaled = scaler.transform(X_batch)
                
                # Make predictions
                predictions = model.predict(X_batch_scaled)
                batch_data['predicted_segment'] = predictions
                batch_data['segment_name'] = batch_data['predicted_segment'].map(segment_mapping)
                
                # Show results
                st.subheader("📈 Batch Prediction Results")
                st.dataframe(batch_data[['predicted_segment', 'segment_name'] + required_cols].head(10))
                
                # Download results
                st.download_button(
                    label="📥 Download All Predictions",
                    data=batch_data.to_csv(index=False),
                    file_name="enhanced_batch_customer_predictions.csv",
                    mime="text/csv"
                )
        else:
            missing_cols = [col for col in required_cols if col not in batch_data.columns]
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("💡 Download a template file with all required columns")
            
            # Create template
            template_data = {col: [] for col in required_cols}
            template_df = pd.DataFrame(template_data)
            st.download_button(
                label="📋 Download Template CSV",
                data=template_df.to_csv(index=False),
                file_name="prediction_template.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Enhanced Customer Segment Predictor • Built with Streamlit • "
    "Uses RFM + Behavioral features for accurate predictions"
    "</div>",
    unsafe_allow_html=True
)