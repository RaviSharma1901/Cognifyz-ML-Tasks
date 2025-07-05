## 🍽️ Restaurant Cuisine Classification using Multi-Label Random Forest

__Project Overview__

This repository contains a complete machine learning workflow for predicting cuisines offered by restaurants based on structured tabular data. The task is part of **Cognifyz ML Challenge – Task 3**. It explores multi-label classification using ensemble modeling and focuses on preprocessing, feature encoding, outlier handling, and model evaluation using label-specific and aggregate metrics.

---

__Repository Structure__

    
    restaurant-cuisine-classification/ 
    ├── Dataset.csv # Source data with restaurant attributes 
    ├── README.md # Project description and guidance 
    ├── restaurant_cuisine_classification.ipynb
    
---

__Features__

- **Multi-label prediction** using `MultiOutputClassifier` with `RandomForestClassifier`
- Data cleaning and preprocessing:
  - Handling missing values in the 'Cuisines' column
  - Dropping low-variance and irrelevant columns
  - Capping outliers for cost and votes
- **Label encoding & binary mapping** for categorical features
- **Cuisine label transformation** using `MultiLabelBinarizer`
- **Robust evaluation metrics**:
  - F1 (micro, macro, weighted)
  - Hamming Loss
  - Subset Accuracy

---

__Model Highlights__

- **Algorithm Used**: Random Forest (with default hyperparameters)
- **Number of Labels**: 145 cuisine categories
- **Best Predicted Cuisines**: North Indian, Chinese, Fast Food
- **Subset Accuracy**: ~6.5%
- **Hamming Loss**: ~1.47%

---

__Feature Importance Summary__

To understand how input features influence predictions, we visualized feature importance from Random Forest models trained on specific cuisines like **North Indian**, **Chinese**, and **Fast Food**.

- `Votes` and `Aggregate rating` consistently ranked as the most influential features.
- Features like `Average Cost for two`, `City`, and `Price range` also contributed moderately.
- Service-based features (e.g., `Online delivery`, `Table booking`) showed minimal impact.

Subplots were used to compare how different cuisine models prioritize features, helping to explain model behavior and user trends more clearly.


---

__Notes__

- Model file (`cuisine_rf_model.pkl`) is **not included** due to GitHub size constraints (~300MB).  
  You can regenerate it by running the notebook, or request an external link if needed.

---

__Future Enhancements__
    - To improve results and make the model even better:
        - Tune model settings using GridSearchCV
        - Group rare cuisines to reduce label imbalance
        - Try other models like XGBoost or LightGBM for performance
        - Use resampling to improve accuracy for less common cuisines
        - Add charts showing which features are most important
        - Build a small web app (like Streamlit) so others can test it

---

- __Links__
  * **Email**: [ravisharma1901@gmail.com](mailto:ravisharma1901@gmail.com)       
  * [LinkedIn](https://www.linkedin.com/in/ravi-sharma-ab8ba17a/)      
  * [GitHub Profile](https://github.com/RaviSharma1901)   


