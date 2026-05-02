# Linear Regression From Scratch

## 📌 Overview
This project implements Linear Regression from scratch using Python and NumPy without relying on machine learning libraries.

The goal is to understand how models learn using Gradient Descent.

---

## 🎯 Problem Statement
Predict house price based on area.

---

## 🧠 Concepts Covered
- Linear Regression
- Gradient Descent
- Mean Squared Error (MSE)
- Feature Scaling
- Model Training & Convergence

---

## 📊 Dataset
A simple synthetic dataset with two columns:

| Feature | Description |
|--------|------------|
| area   | Area of house |
| price  | Price of house |

---

## ⚙️ Model Formula

y = wx + b

Where:
- w → weight (slope)
- b → bias (intercept)

---

## 🚀 How the Model Learns

1. Start with random weight & bias
2. Predict output
3. Calculate error (loss)
4. Update weight & bias using gradient descent
5. Repeat until loss is minimized

---

## 📉 Results

- Loss decreases rapidly and converges to near zero
- Model perfectly fits data due to clean linear dataset
- Strong linear relationship observed between area and price

---

## 📷 Outputs

### Regression Line
![Regression](outputs/regression_line.png)

### Loss Curve
![Loss](outputs/loss_curve.png)

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
py src/linear_regression_day1.py
