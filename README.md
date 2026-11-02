# 🏠 House Price Prediction Using Advanced Regression

**Live App:**
https://supervisedlearningpr2-778lr3hdf6frr8vwn2qkwi.streamlit.app/

## 📌 Project Overview

This project applies advanced supervised learning regression techniques to predict house prices and study **regularization, cross-validation, overfitting, and model generalization**.

The project compares linear, tree-based, and Support Vector Regression models to identify the best-performing model.

---

## 🎯 Objectives

- Apply Ridge and Lasso regularization.
- Implement different cross-validation strategies.
- Build Decision Tree and Random Forest models.
- Implement Support Vector Regression (SVR).
- Compare linear and non-linear models.
- Identify overfitting and underfitting.
- Select the best model based on unseen test data.

---

## 📊 Dataset

The dataset contains **3,800 real-estate records**.

### Target Variable

`house_price_inr`

### Main Features

- `area_sqft`
- `bedrooms`
- `bathrooms`
- `location_score`
- `property_age`
- `distance_city_km`
- `near_school`
- `near_metro`
- `crime_rate_index`
- `sale_date`

`property_id` is used only as an identifier and is excluded from model training.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🤖 Models Implemented

### Regularized Linear Models
- Ridge Regression (L2)
- Lasso Regression (L1)

### Tree-Based Models
- Decision Tree Regression
- Random Forest Regression

### Support Vector Regression
- Linear SVR
- RBF SVR

---

## 🔄 Cross-Validation

The following strategies were implemented:

- K-Fold Cross-Validation
- Stratified K-Fold Cross-Validation
- Leave-One-Out Cross-Validation (LOOCV)
- Time Series Split

---

## 📏 Evaluation Metrics

Models were evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📊 Model Results

| Model | RMSE | R² |
|---|---:|---:|
| Ridge | 2,539,952.40 | 0.9199 |
| Lasso | 2,539,728.19 | 0.9199 |
| Decision Tree | 2,953,891.38 | 0.8917 |
| **Random Forest** | **2,405,401.92** | **0.9282** |
| SVR Linear | 8,982,717.42 | -0.0019 |
| SVR RBF | 8,987,350.53 | -0.0029 |

---

## 🏆 Best Model

**Random Forest Regression** achieved the best overall performance with:

- **R² Score:** 0.9282
- **RMSE:** 2,405,401.92
- **MAE:** 1,757,755.01

It provided the best generalization among the tested models.

---

## 🔍 Key Findings

- Ridge and Lasso performed similarly and showed stable generalization.
- Lasso provided only a very small improvement over Ridge.
- The Decision Tree showed signs of overfitting.
- Random Forest performed better than a single Decision Tree.
- Both SVR models significantly underperformed on this dataset.
- Random Forest provided the best balance of predictive performance and generalization.

---

## ✅ Conclusion

- The project demonstrates how regularization, cross-validation, and different regression algorithms can be used to build reliable house price prediction models.

- Random Forest Regression was the best-performing model, achieving an R² score of 0.9282 on the test dataset.

- Explanation Video Link : https://drive.google.com/file/d/1mNM2ZxQ9GSMTT22E-Uvepi9f4Wj1I-pu/view?usp=drive_link
