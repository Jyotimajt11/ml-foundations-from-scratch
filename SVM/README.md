# Support Vector Machine (SVM) - Handwritten Digit Classification

This project implements a Support Vector Machine model to classify handwritten digits from 0 to 9 using Scikit-learn's built-in Digits dataset.

## Project Overview

Support Vector Machine is a supervised machine learning algorithm used mainly for classification problems. In this project, SVM is used to recognize handwritten digits based on pixel values.

Each digit image is an 8×8 grayscale image. The image is converted into 64 numerical features, and the SVM model learns patterns from these features to classify the correct digit.

## Dataset Used

Dataset: Scikit-learn Digits Dataset

The dataset contains:

- 1,797 digit images
- 10 classes: digits 0 to 9
- Each image size: 8×8 pixels
- Total features per image: 64

## Why SVM Works Well Here

SVM works well on the Digits dataset because:

- The dataset is small to medium-sized
- Each image has many numerical features
- SVM performs well in high-dimensional spaces
- RBF kernel can handle non-linear boundaries
- Feature scaling improves model performance

## Project Structure

```text
SVM/
├── src/
│   └── svm_digitclassification_model.py
├── outputs/
│   ├── sample_digits.png
│   └── confusion_matrix.png
└── README.md
```
