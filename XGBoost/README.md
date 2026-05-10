# 🚀 XGBoost Customer Churn Prediction

This project demonstrates how to use **XGBoost (Extreme Gradient Boosting)** to predict customer churn using the Telco Customer Churn dataset.

Customer churn prediction helps businesses identify customers who are likely to leave their service, enabling proactive retention strategies.

---

## 📌 Project Objective

The objectives of this project are to:

- Load and preprocess the Telco Customer Churn dataset.
- Convert categorical features into numerical form.
- Train an XGBoost classification model.
- Evaluate performance using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Visualize feature importance.
- Identify the key factors driving customer churn.

---

## 📂 Dataset

**Dataset Name:** Telco Customer Churn Dataset

**Source:** IBM Sample Dataset / Kaggle

### Download Links

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- IBM GitHub CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

### Target Variable

- `No` → Customer stays
- `Yes` → Customer churns

### Important Features

- `tenure`
- `Contract`
- `MonthlyCharges`
- `TotalCharges`
- `OnlineSecurity`
- `TechSupport`
- `InternetService`
- `PaymentMethod`

---

## 🧠 Why XGBoost?

XGBoost is one of the most powerful algorithms for structured tabular data.

It was chosen because:

- It handles nonlinear relationships.
- It automatically captures feature interactions.
- It includes L1 and L2 regularization.
- It reduces overfitting.
- It provides feature importance.
- It achieves excellent performance on business datasets.

---

## 📁 Project Structure

```text
XGBoost/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   └── xgboost_churn_model.py
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost

---

## 📦 Installation

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn xgboost
```

If `pip` does not work:

```bash
py -m pip install pandas matplotlib scikit-learn xgboost
```

---

## ▶️ How to Run the Project

If you are in the root `ML-Foundations` folder:

```bash
py XGBoost/src/xgboost_churn_model.py
```

If you are inside the `XGBoost` folder:

```bash
py src/xgboost_churn_model.py
```

---

## 📊 Model Performance

### Example Results

- Accuracy: **~80%**

### Confusion Matrix

| Actual \ Predicted | No Churn | Churn |
| ------------------ | -------: | ----: |
| No Churn           |      921 |   112 |
| Churn              |      173 |   201 |

### Interpretation

- Correctly predicted customers who stayed: 921
- Correctly predicted customers who churned: 201
- Missed churners: 173
- False churn alerts: 112

---

## 📈 Feature Importance

Top features identified by XGBoost:

1. `Contract`
2. `OnlineSecurity`
3. `TechSupport`
4. `InternetService`
5. `tenure`

### Business Insight

Customers with month-to-month contracts, short tenure, and no security or technical support are more likely to churn.

---

## 🏢 Business Applications

Customer churn prediction is widely used in:

- Telecom
- Banking
- SaaS
- Subscription businesses
- Insurance

It helps organizations reduce customer loss and improve retention.

---

## 📚 Concepts Learned

- Gradient Boosting
- Sequential learning
- Learning rate
- Tree depth
- Regularization (`reg_alpha`, `reg_lambda`)
- Feature importance
- Classification metrics

---

## 🆚 Random Forest vs XGBoost

| Feature            | Random Forest | XGBoost                        |
| ------------------ | ------------- | ------------------------------ |
| Tree Building      | Parallel      | Sequential                     |
| Learns from Errors | No            | Yes                            |
| Regularization     | Limited       | Built-in                       |
| Accuracy           | High          | Often Higher                   |
| Training Speed     | Fast          | More Computationally Intensive |

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- ROC-AUC Curve
- Precision-Recall Curve
- Cross-validation
- SHAP explanations

---
