## __Restaurant Recommendation System 🍽️__    
- A hybrid machine learning-based restaurant recommendation system that helps users discover restaurants based on their cuisine preferences, budget constraints, and quality expectations.

- __Objective__

    - To develop a restaurant recommendation system that helps users discover restaurants based on their:
        - Cuisine preferences
        - Budget constraints
        - Quality expectations (ratings)

- __Technical Approach__

    - Machine Learning Techniques Used:
        - TF-IDF Vectorization - For cuisine similarity analysis
        - MinMax Scaling - For cost normalization
        - Cosine Similarity - For content-based recommendations
        - Hybrid Filtering - Combines user preferences with similarity-based recommendations

- __Algorithm Flow:__

        User Input → Preference Filtering → Similarity Calculation → Recommendations

- __Dataset__
    - The system uses a restaurant dataset with key features:
        - Restaurant Name - Name of the restaurant
        - Cuisines - Types of food served
        - Average Cost for two - Pricing information
        - Aggregate rating - User ratings (0-5 scale)

- __Features__
    - Content-Based Filtering: Recommends restaurants similar to user preferences
    - Budget-Aware Recommendations: Filters based on cost constraints
    - Quality Assurance: Ensures minimum rating requirements
    - Hybrid Approach: Adapts recommendation strategy based on data availability
    - Error Handling: Manages edge cases and invalid inputs
- 📂 restaurant-recommendation-system/

        ├── Dataset.csv                     # Restaurant dataset used for recommendations
        ├── app.py                         # Streamlit app code
        ├── restaurant_recommendation_system.ipynb  # Jupyter notebook for development
        ├── requirements.txt               # List of required Python packages
        ├── tfidf_vectorizer.pickle        # Saved TF-IDF vectorizer
        ├── cost_scaler.pickle             # Saved cost scaler
        └── README.md                      # Project documentation

- __Requirements__

        pip install pandas numpy scikit-learn scipy
- __Live Demo__
    - Explore the restaurant recommendation system:
        - [Launch App](https://smart-menu-guide.streamlit.app/)

- __Test Cases__
    - Test Scenario 
        - Popular cuisine (Japanese, ₹1500, ≥4.5) Content-based similarity
        - Common cuisine (Mexican, ₹800, ≥4.0)- Content-based similarity
        - Niche cuisine (Lucknowi, ₹1000, ≥4.0)Direct recommendation
        - Invalid criteria (Martian, ₹50, 5.0)Clear error message

- __Performance Metrics__
    - Accuracy: Successfully handles diverse query types
    - Robustness: Zero system errors across test scenarios
    - Scalability: Efficient similarity computation using sparse matrices
    - User Experience: Clear feedback and formatted output
  
- __Algorithm Details__

    1. Data Preprocessing
        - Handle missing values in cuisines
        - Normalize text data (lowercase, strip whitespace)
        - Select relevant features for recommendation

    2. Feature Engineering
        - TF-IDF: Convert cuisine text to numerical vectors
        - MinMax Scaling: Normalize cost data (0-1 range)
        - Feature Combination: Merge cuisine and cost features

    3. Similarity Computation
        - Calculate cosine similarity between all restaurants
        - Create similarity matrix for fast lookups

    4. Recommendation Logic
        - Option 1: Direct filtering (≤ top_n matches)
        - Option 2: Similarity-based (> top_n matches)

    - __*Future Enhancements*__
        - Add collaborative filtering
        - Implement location-based recommendations

- This project is part of the Cognifyz ML Tasks series.

- __Links__
  * **Email**: [ravisharma1901@gmail.com](mailto:ravisharma1901@gmail.com)       
  * [LinkedIn](https://www.linkedin.com/in/ravi-sharma-ab8ba17a/)      
  * [GitHub Profile](https://github.com/RaviSharma1901)   
