# K-Nearest Neighbors Classification Model

## Overview

This project implements the K-Nearest Neighbors algorithm as part of my Machine Learning Foundations journey.

KNN is a supervised machine learning algorithm used mainly for classification problems. It predicts the class of a new data point by checking the classes of its nearest neighbors.

## Model Type

Supervised Learning - Classification

## Concept

KNN works on the idea that similar data points are close to each other.

For a new input, the model:

1. Calculates distance from existing data points
2. Finds the nearest K neighbors
3. Checks the majority class among those neighbors
4. Assigns that class to the new data point

Example:

```text
If K = 5

Among 5 nearest points:
3 belong to Class 1
2 belong to Class 0

Prediction = Class 1
```
