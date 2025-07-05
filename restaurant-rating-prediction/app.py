import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("Restaurant Rating Prediction\decision_tree_model.pk", "rb") as f:
    model = pickle.load(f)

# Scaling parameters from your training data (StandardScaler)
SCALING_PARAMS = {
    'Average Cost for two': {'mean': 5.808669, 'std': 0.855754},
    'Votes': {'mean': 2.724629, 'std': 1.695945},
    'Price range': {'mean': 1.552635, 'std': 0.708941},
    'City_freq': {'mean': 3102.932075, 'std': 2009.281798}
}

def standardize_features(avg_cost, votes, price_range, city_freq):
    """Apply StandardScaler transformation using training data statistics"""
    scaled_avg_cost = (avg_cost - SCALING_PARAMS['Average Cost for two']['mean']) / SCALING_PARAMS['Average Cost for two']['std']
    scaled_votes = (votes - SCALING_PARAMS['Votes']['mean']) / SCALING_PARAMS['Votes']['std']
    scaled_price_range = (price_range - SCALING_PARAMS['Price range']['mean']) / SCALING_PARAMS['Price range']['std']
    scaled_city_freq = (city_freq - SCALING_PARAMS['City_freq']['mean']) / SCALING_PARAMS['City_freq']['std']
    
    return scaled_avg_cost, scaled_votes, scaled_price_range, scaled_city_freq

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")
st.title("🍽️ Restaurant Aggregate Rating Predictor")
st.markdown("Provide restaurant features to estimate the expected user rating.")

# Numerical inputs - RAW VALUES (what users understand)
avg_cost = st.number_input("🧾 Average Cost for Two (₹)", min_value=0, value=1200, step=50,
                          help="Enter the average cost for two people in rupees")

votes = st.number_input("🗳️ Number of Votes", min_value=0, value=500, step=10,
                       help="Total number of votes/reviews the restaurant has received")

price_range = st.slider("💰 Price Range", 1, 4, value=2,
                       help="1=Budget, 2=Mid-range, 3=Expensive, 4=Very Expensive")

table_booking = st.selectbox("🪑 Table Booking Available", ["No", "Yes"])

online_delivery = st.selectbox("📦 Online Delivery Available", ["No", "Yes"])

city_freq = st.number_input("📍 City Frequency Score", min_value=0, value=3000, step=100,
                           help="Higher values indicate restaurants in more popular/frequent cities")

# Cuisine multiselect
cuisines = [
    "American", "Bakery", "Beverages", "Biryani", "Burger", "Cafe", "Chinese",
    "Continental", "Desserts", "Fast Food", "Healthy Food", "Ice Cream", 
    "Italian", "Mithai", "Mughlai", "North Indian", "Other", "Pizza", 
    "Raw Meats", "South Indian", "Street Food"
]
selected_cuisines = st.multiselect("🍱 Select Cuisines", cuisines)

# Define all cuisine features (must match your model's training features)
cuisine_features = [
    'Cuisines_American', 'Cuisines_Bakery', 'Cuisines_Beverages', 'Cuisines_Biryani',
    'Cuisines_Burger', 'Cuisines_Cafe', 'Cuisines_Chinese', 'Cuisines_Continental',
    'Cuisines_Desserts', 'Cuisines_Fast Food', 'Cuisines_Healthy Food',
    'Cuisines_Ice Cream', 'Cuisines_Italian', 'Cuisines_Mithai', 'Cuisines_Mughlai',
    'Cuisines_North Indian', 'Cuisines_Other', 'Cuisines_Pizza', 'Cuisines_Raw Meats',
    'Cuisines_South Indian', 'Cuisines_Street Food'
]

# Predict and display
if st.button("🎯 Predict Rating"):
    # Apply StandardScaler transformation to numerical features
    scaled_avg_cost, scaled_votes, scaled_price_range, scaled_city_freq = standardize_features(
        avg_cost, votes, price_range, city_freq
    )
    
    # Prepare input data with scaled numerical features
    input_data = {
        'Average Cost for two': scaled_avg_cost,
        'Votes': scaled_votes,
        'Price range': scaled_price_range,
        'Has Table booking': int(table_booking == "Yes"),
        'Has Online delivery': int(online_delivery == "Yes"),
        'City_freq': scaled_city_freq,
    }
    
    # Add cuisine one-hot encodings
    for cuisine_col in cuisine_features:
        readable_name = cuisine_col.split('_', 1)[1]
        input_data[cuisine_col] = int(readable_name in selected_cuisines)
    
    # Create DataFrame and predict
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    # Display result
    st.success(f"⭐ Estimated Aggregate Rating: {round(prediction, 2)}")
    
    # Optional: Show what the model actually received (for debugging)
    with st.expander("🔍 Debug: Scaled Values Used by Model"):
        st.write("**Standardized Values (z-scores):**")
        st.write(f"- Average Cost: {scaled_avg_cost:.4f}")
        st.write(f"- Votes: {scaled_votes:.4f}")
        st.write(f"- Price Range: {scaled_price_range:.4f}")
        st.write(f"- City Frequency: {scaled_city_freq:.4f}")
        
        st.write("**Raw Input Values:**")
        st.write(f"- Average Cost: ₹{avg_cost}")
        st.write(f"- Votes: {votes}")
        st.write(f"- Price Range: {price_range}")
        st.write(f"- City Frequency: {city_freq}")

# Add some helpful information
st.markdown("---")
st.markdown("### 💡 Tips:")
st.markdown("- **Average Cost**: Typical cost for two people dining")
st.markdown("- **Votes**: More votes usually indicate popular restaurants")
st.markdown("- **Price Range**: 1=Budget friendly, 4=Premium dining")
st.markdown("- **City Frequency**: Restaurants in popular cities tend to have higher scores")



#  Local URL: http://localhost:8507
# Network URL: http://192.168.68.116:8507
# cd Interview\Cognify_ml_task\"Restaurant Rating Prediction"
# streamlit run app.py
