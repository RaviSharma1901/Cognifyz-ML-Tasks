    
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# Load Dataset
@st.cache_data # Cache the dataset loading function
# This decorator caches the output of the function to avoid reloading the dataset on every interaction
def load_data():
    df = pd.read_csv("RaviSharma1901/Cognifyz-ML-Tasks/restaurant-recommendation-system/Dataset.csv")
    df['Cuisines'] = df['Cuisines'].fillna(df['Cuisines'].mode()[0]).str.lower()
    # Clean restaurant names - remove garbled characters
    df['Restaurant Name'] = df['Restaurant Name'].str.replace(r'[^\w\s&\'-]', '', regex=True)
    return df

df = load_data()

# Load or Fit Models
def get_vectorizer_and_scaler(df):
    # Check if the models already exist
    if os.path.exists("tfidf_vectorizer.pickle") and os.path.exists("cost_scaler.pickle"):
    # Load the pre-trained models
        # Load the TF-IDF vectorizer
        with open("tfidf_vectorizer.pickle", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        # Load the MinMaxScaler
        with open("cost_scaler.pickle", "rb") as f:
            # Load the cost scaler
            cost_scaler = pickle.load(f)
    else: # If models do not exist, fit them
        
        # Fit the TF-IDF vectorizer on 'Cuisines'
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(df['Cuisines'])
        # Fit the MinMaxScaler on 'Average Cost for two'
        cost_scaler = MinMaxScaler()
        cost_scaler.fit(df[['Average Cost for two']])
        
        # Save the fitted models
        with open("tfidf_vectorizer.pickle", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        with open("cost_scaler.pickle", "wb") as f:
            pickle.dump(cost_scaler, f)
        st.info("Models trained and saved automatically.")

    return tfidf_vectorizer, cost_scaler

tfidf_vectorizer, cost_scaler = get_vectorizer_and_scaler(df)

# Feature Engineering
# Build combined features for the dataset
def build_combined_features(df):
    # Transform 'Cuisines' using TF-IDF
    tfidf_matrix = tfidf_vectorizer.transform(df['Cuisines'])
    # Scale 'Average Cost for two' using MinMaxScaler
    cost_scaled = cost_scaler.transform(df[['Average Cost for two']])
    # Convert scaled cost to sparse matrix
    cost_sparse = csr_matrix(cost_scaled)
    # hstack combines the TF-IDF matrix and the scaled cost into a single sparse matrix
    return hstack([tfidf_matrix, cost_sparse])
# Combine TF-IDF and scaled cost into a single sparse matrix
combined_features = build_combined_features(df)

# Filter restaurants based on user preferences
def filter_by_preferences(df, cuisine, budget, rating):
    # Filter restaurants based on cuisine, budget, and rating
    return df[
        df['Cuisines'].str.contains(cuisine.lower(), na=False) &
        (df['Average Cost for two'] <= budget) &
        (df['Aggregate rating'] >= rating)
    ]
    
# Recommend similar restaurants based on cosine similarity
def recommend_similar(df, combined_features, base_df, budget, rating, top_n=5):
    # Check if base_df is empty
    if base_df.empty:
        return pd.DataFrame(), None # No recommendations if no restaurants match the filter
    
# Check if there are enough candidates for similarity
    if len(base_df) <= top_n:
        # Not enough candidates for similarity — fallback to Option 1
        sorted_df = base_df.sort_values(by='Aggregate rating', ascending=False).head(top_n)
        best_anchor = sorted_df.iloc[0] # Get the best anchor restaurant
        return sorted_df, best_anchor # Return the top N restaurants sorted by rating

    # Else: compute similarity normally
    best_anchor = base_df.loc[base_df['Aggregate rating'].idxmax()]
    idx = best_anchor.name

    # Get the index of the best anchor(Top) restaurant
    anchor_vector = combined_features[idx]
    # Compute cosine similarity between the anchor restaurant and all others 
    sim_scores = cosine_similarity(anchor_vector, combined_features).flatten() # Flatten to get a 1D array of similarity scores
    # Sort the restaurants based on similarity scores, excluding the anchor(top) restaurant itself
    ranked_indices = sim_scores.argsort()[::-1][1:]

# Filter the ranked indices based on budget and rating
    # Initialize an empty list to store the indices of recommended restaurants
    recommendations = []
    # Iterate through the ranked indices and check if they meet the budget and rating criteria
    for i in ranked_indices:
        # Get the restaurant at the current index
        restaurant = df.iloc[i]
        # Check if the restaurant meets the budget and rating criteria
        if (
            restaurant['Average Cost for two'] <= budget and
            restaurant['Aggregate rating'] >= rating
        ):
            recommendations.append(i)
        # If we have enough recommendations, break the loop
        if len(recommendations) >= top_n:
            break
    return df.iloc[recommendations], best_anchor # Return the recommended restaurants and the best anchor restaurant

# Streamlit UI
# Set up the Streamlit app

# Set the title of the app
st.title("🍽️Restaurant Recommendation System")
# Display the dataset information
unique_cuisines = sorted(df['Cuisines'].str.title().unique())
# Create a selectbox for cuisine preference
selected_cuisine = st.selectbox("Preferred Cuisine", unique_cuisines)
# Create sliders for budget and rating preferences
budget_input = st.slider("Max Cost for Two(₹)", min_value=50,max_value=10000,value=1500,step=50)
rating_input = st.slider("Minimum Rating", 0.0, 5.0, 4.5, 0.1)
# Create a slider for the number of recommendations
top_n = st.slider("Number of Recommendations", 1, 10, 5)

# Display the selected preferences
if st.button("Recommend Restaurants"):
    # Filter the dataset based on user preferences
    filtered_df = filter_by_preferences(df, selected_cuisine, budget_input, rating_input)
    # Display the number of restaurants found
    results, anchor = recommend_similar(df, combined_features, filtered_df, budget_input, rating_input, top_n)
# Display the top match and similar restaurants
    if anchor is not None:
        st.markdown(
            f"**Top Match**: `{anchor['Restaurant Name']}` — {anchor['Cuisines']} | "
            f"₹{anchor['Average Cost for two']} | {anchor['Aggregate rating']}"
        )
    else: 
        st.warning("No restaurants matched your preferences.")
# Display the results
    if not results.empty:
        # Display the number of similar restaurants found
        st.success(f"Top {len(results)} Similar Restaurants")
        # Display the results in a dataframe
        st.dataframe(results[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']].reset_index(drop=True))
    