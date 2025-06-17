# Model Evaluate App 🌟

Model Evaluate App is a **Flask application** designed to streamline the process of **exploring**, **processing**, and **evaluating** your datasets with Machine Learning models — all through a simple UI.

---

## 🔹 Features

✅ **Basic Exploratory Data Analysis (EDA)**  
- Performs automatic data cleaning.
- Displays key statistics and drops missing or unreliable columns.

✅ **Automated Model Selection**  
- Detects whether your problem is **Classification or Regression** based on your data.

✅ **Training Multiple Models**  
For **Classification**, it trains:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

For **Regression**, it trains:
- Linear Regression
- Random Forest Regressor

✅ **Performance Evaluation**  
- Computes Accuracy (for Classification) or R² Score (for Regression).
- Displays a clear comparison of all models.

✅ **Customizable Threshold**  
- If the highest score exceeds your specified threshold, it reports the accuracy directly.
- If it falls below the threshold, it generates a cleaned and preprocessed CSV for you to **download and perform further analysis or tuning**.

---

## 🔹 How It Works

1️⃣ **Load your CSV file.**

2️⃣ The app performs:
- **Basic EDA:** Handle missing values, drop unreliable columns, and cleans the data.
- **Model Type Detection:** Determines whether it's a regression or a classification problem.

3️⃣ Based on the problem, it:
- Initializes appropriate models.
- Trains and evaluates all of them on your cleaned data.

4️⃣ The best score is displayed.  
If it’s above your defined **threshold**, you’re all set!  
If it’s below, the cleaned dataset is made available for **download**, allowing you to perform additional feature engineering or tuning.

---

## 🔹 Tech Stack

- **Python**
- **Flask**
- **Scikit-Learn**
- **XGBoost**
- **Pandas, NumPy**

---

## 🔹 Installation

```bash
git clone https://github.com/sohanur-cefalo/ml-model-assistant.git
cd ml-model-assistant
docker compose up --build
