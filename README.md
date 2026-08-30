# 🚗 Car Price Prediction using Machine Learning

A Machine Learning regression project designed to predict used car prices based on vehicle attributes such as vehicle age, fuel type, seller type, transmission, and brand/model categories.

This project implements **Leakage-Free Target Encoding** to process high-cardinality categorical features accurately without data leakage.

---

## 📌 Project Overview

- **Problem Type:** Supervised Learning (Regression)
- **Dataset:** Used Cars Dataset
- **Target Variable:** `Selling_Price`
- **Primary Algorithm:** Random Forest Regressor

---

## 🛠️ Key Technical Highlights

1. **Feature Engineering:** 
   - Derived `Car_Age` from the vehicle's manufacturing year to improve predictive signal.
   - Extracted primary model categories (`Model_Short`) by taking the core model name to reduce category dispersion.
2. **Leakage-Free Target Encoding:**
   - Applied Target Encoding on `Model_Short` and `Brand` strictly using the training set (`X_train` & `y_train`) to prevent data leakage.
   - Handled unseen test categories gracefully using a multi-tiered fallback hierarchy.
3. **Model Evaluation:**
   - Evaluated using standard regression metrics ($R^2$ Score and Mean Absolute Error).

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **$R^2$ Score (Accuracy)** | **~77.94%** |
| **Mean Absolute Error (MAE)** | **~112,300** |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MoBa-create/car-price-prediction.git
   cd car-price-prediction

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
3 - python car_price_prediction.py

## 🛠️ Tech Stack

- Python 3.x
- Pandas & NumPy (Data Processing & Feature Engineering)
- Scikit-Learn (Model Training & Metrics)
