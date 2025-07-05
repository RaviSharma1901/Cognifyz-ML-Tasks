# 🍽️ Restaurant Cuisine Classification using Multi-Label Random Forest

## 📌 Project Overview

This repository contains a complete machine learning workflow for predicting cuisines offered by restaurants based on structured tabular data. The task is part of **Cognifyz ML Challenge – Task 3**. It explores multi-label classification using ensemble modeling and focuses on preprocessing, feature encoding, outlier handling, and model evaluation using label-specific and aggregate metrics.

---

## 📁 Repository Structure


---

## 🔧 Features

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

## 📈 Model Highlights

- **Algorithm Used**: Random Forest (with default hyperparameters)
- **Number of Labels**: 145 cuisine categories
- **Best Predicted Cuisines**: North Indian, Chinese, Fast Food
- **Subset Accuracy**: ~6.5%
- **Hamming Loss**: ~1.47%

---

## 📝 Notes

- Model file (`cuisine_rf_model.pkl`) is **not included** due to GitHub size constraints (~300MB).  
  You can regenerate it by running the notebook, or request an external link if needed.

---

## 👤 Author

**Ravi Sharma**  
GitHub: [RaviSharma1901](https://github.com/RaviSharma1901)

---

## 📬 Contact

For queries or collaboration opportunities, feel free to reach out via GitHub Issues or LinkedIn!

