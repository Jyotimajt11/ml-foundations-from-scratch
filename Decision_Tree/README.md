# Decision Tree Classification - Heart Disease Prediction

## Overview

This project uses a Decision Tree Classifier to predict whether a patient has heart disease or not based on medical features.

Decision Tree is a supervised machine learning algorithm that works by asking a series of questions and splitting the data based on feature values.

## Dataset

The dataset used is `heart.csv`.

Target column:

- `0` = No Heart Disease
- `1` = Heart Disease

## Features Used

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

## Model Used

DecisionTreeClassifier from Scikit-Learn.

Important parameters:

```python
criterion="gini"
max_depth=4
random_state=42
```
