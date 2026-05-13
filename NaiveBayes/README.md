# 🧠 Naive Bayes Projects

This folder contains practical implementations of the Naive Bayes family of algorithms using real-world datasets.

Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It assumes that all features are conditionally independent given the class label.

Despite this simplifying assumption, Naive Bayes performs exceptionally well on many real-world problems, especially text classification.

---

## 📚 Projects Included

### 1. Gaussian Naive Bayes — Breast Cancer Detection

- **Algorithm:** GaussianNB
- **Dataset:** Breast Cancer Wisconsin Dataset
- **Problem Type:** Binary Classification
- **Use Case:** Predict whether a tumor is malignant or benign
- **Expected Accuracy:** 93–97%

### 2. Multinomial Naive Bayes — News Classification

- **Algorithm:** MultinomialNB
- **Dataset:** 20 Newsgroups Dataset
- **Problem Type:** Multi-Class Text Classification
- **Use Case:** Categorize news articles into topics
- **Expected Accuracy:** 80–90%

### 3. Bernoulli Naive Bayes — SMS Spam Classification

- **Algorithm:** BernoulliNB
- **Dataset:** SMS Spam Collection Dataset
- **Use Case:** Detect spam messages
- **Expected Accuracy:** 96–98%

---

## 📂 Folder Structure

Naive_Bayes/
├── GaussianNB_Breast_Cancer/
│ ├── src/
│ └── outputs/
│
├── MultinomialNB_News/
│ ├── src/
│ └── outputs/
│
├── BernoulliNB_Spam/
│ ├── data/
│ ├── src/
│ └── outputs/
