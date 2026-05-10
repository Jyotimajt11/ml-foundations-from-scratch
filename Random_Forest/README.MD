# 🌲 Random Forest Classification on Heart Disease Dataset

This project demonstrates how to build a **Random Forest Classifier** using Scikit-learn to predict whether a patient has heart disease based on medical attributes.

Random Forest is an ensemble learning algorithm that combines multiple decision trees and uses majority voting to produce more accurate and robust predictions.

---

## 📌 Project Objective

The objectives of this project are to:

- Load and preprocess the Heart Disease dataset.
- Train a Random Forest model.
- Evaluate performance using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Visualize Feature Importance.
- Compare Random Forest with a single Decision Tree.

---

## 📂 Dataset

The dataset used is `heart.csv`.

### Target Variable

- `0` → No Heart Disease
- `1` → Heart Disease Present

### Features

- `age` — Age of the patient
- `sex` — Gender
- `cp` — Chest pain type
- `trestbps` — Resting blood pressure
- `chol` — Serum cholesterol
- `fbs` — Fasting blood sugar
- `restecg` — Resting ECG results
- `thalach` — Maximum heart rate achieved
- `exang` — Exercise induced angina
- `oldpeak` — ST depression induced by exercise
- `slope` — Slope of peak exercise ST segment
- `ca` — Number of major vessels colored by fluoroscopy
- `thal` — Thalassemia result
- `target` — Output label

---

## 🧠 Why Random Forest?

A single Decision Tree may overfit the training data.

Random Forest improves performance by:

1. Building multiple decision trees.
2. Training each tree on a random subset of the data.
3. Using a random subset of features at each split.
4. Combining all predictions using majority voting.

This results in:

- Higher accuracy
- Reduced overfitting
- Better generalization
- Feature importance scores

---

## 📁 Project Structure

```text
Random_Forest/
│
├── data/
│   └── heart.csv
│
├── src/
│   └── random_forest_model.py
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

---

## 📦 Installation

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

If `pip` does not work:

```bash
py -m pip install pandas matplotlib scikit-learn
```

---

## ▶️ How to Run the Project

Open the terminal inside the `Random_Forest` folder and run:

```bash
py src/random_forest_model.py
```

If `py` does not work:

```bash
python src/random_forest_model.py
```

---

## 📊 Model Performance

### Example Results

- Accuracy: **87.3%**

### Confusion Matrix

| Actual \ Predicted |   0 |   1 |
| ------------------ | --: | --: |
| 0 (No Disease)     |  82 |  20 |
| 1 (Disease)        |   6 |  97 |

### Interpretation

- True Negatives (TN): 82
- False Positives (FP): 20
- False Negatives (FN): 6
- True Positives (TP): 97

The model correctly identified most patients with heart disease and missed only 6 cases.

---

## 📈 Feature Importance

The most important features were:

1. `cp` — Chest pain type
2. `thal`
3. `ca`
4. `oldpeak`
5. `thalach`

These medical indicators were the strongest predictors of heart disease.

---

## 🆚 Decision Tree vs Random Forest

| Metric           | Decision Tree | Random Forest |
| ---------------- | ------------: | ------------: |
| Accuracy         |      Moderate |        Higher |
| Overfitting      |          High |           Low |
| Stability        |     Sensitive |        Robust |
| Interpretability |          Easy |      Moderate |
| Training Speed   |          Fast |        Slower |

---

## 🏥 Real-World Applications

- Heart disease prediction
- Fraud detection
- Credit scoring
- Customer churn prediction
- Manufacturing defect detection

---

## 📚 Concepts Learned

- Ensemble Learning
- Bootstrap Sampling
- Feature Randomness
- Majority Voting
- Feature Importance
- Confusion Matrix
- Precision, Recall, and F1-Score

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation
- ROC Curve and AUC Score
- Comparison with XGBoost and LightGBM

---
