# Customer Segmentation using K-Means Clustering

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to the **Customer Personality Analysis** dataset to discover meaningful customer segments based on demographics, spending behavior, and purchasing patterns.

The project demonstrates the complete clustering workflow, including data preprocessing, feature engineering, the Elbow Method, Silhouette Score evaluation, PCA-based visualization, and model serialization.

---

## Project Objective

The objective of this project is to group customers into distinct clusters so businesses can:

- Identify high-value customers.
- Detect inactive customers.
- Personalize marketing campaigns.
- Improve customer retention.
- Optimize promotional strategies.

---

## Dataset

- **Dataset:** Customer Personality Analysis
- **Source:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
- **File Used:** `marketing_campaign.csv`
- **Records:** 2,240 customers
- **Features:** 29 columns

---

## Problem Statement

Given customer demographic and transactional data, the goal is to identify natural groups of customers without using any predefined labels.

---

## Machine Learning Concepts Covered

- Unsupervised Learning
- Clustering
- K-Means Algorithm
- Euclidean Distance
- Centroids
- Inertia (WCSS)
- Elbow Method
- Silhouette Score
- Principal Component Analysis (PCA)
- Feature Engineering
- Model Serialization using Joblib

---

## Feature Engineering

Several new features are created to better represent customer behavior.

### Age

```python
Age = 2026 - Year_Birth
```

### Total Spending

Total_Spending = (
MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
)

### Total purchases

Total_Purchases = (
NumWebPurchases + NumCatalogPurchases + NumStorePurchases
)

### Children

Children = Kidhome + Teenhome

### Features Used for Clustering

Age
Income
Recency
Total_Spending
Total_Purchases
NumWebVisitsMonth
Children

### Why Feature Scaling is Required

K-Means uses Euclidean distance, which is sensitive to feature magnitudes.

For example:

Income may be in tens of thousands.
Children ranges from 0 to 3.

Without scaling, larger-value features dominate the clustering process.

StandardScaler transforms each feature to have:

Mean = 0
Standard Deviation = 1

### Choosing the Optimal Number of Clusters

Elbow Method

The model is trained for values of K from 1 to 10.

For each K, the algorithm computes Inertia (WCSS).

The best K is chosen where the curve forms an elbow, indicating diminishing returns.

In this project, the elbow occurs at:

K = 4

### Silhouette Score

The Silhouette Score measures cluster quality.

Close to 1 → Well-separated clusters
Close to 0 → Overlapping clusters
Negative → Poor clustering

This project uses the score to validate the chosen number of clusters.

### Principal Component Analysis (PCA)

The dataset contains multiple numerical features.

PCA reduces the high-dimensional data to two principal components:

PCA Component 1
PCA Component 2

These components preserve most of the variance and allow clusters to be visualized in a two-dimensional scatter plot.

### Why PCA is Used

Simplifies visualization
Preserves important structure
Makes cluster interpretation easier

### Results

The model successfully identified 4 customer segments.

Possible business interpretations:

VIP Customers
Regular Customers
Budget Shoppers
Inactive Customers

### Output Files

outputs/
├── elbow_method.png
├── customer_segments.png
├── cluster_summary.csv
├── segmented_customers.csv
├── kmeans_model.pkl
├── scaler.pkl
└── pca.pkl

### Project Structure

CustomerPersonality_KMEANS/
├── data/
│ └── marketing_campaign.csv
│
├── src/
│ └── customer_segmentation_kmeans_model.py
│
├── outputs/
│ ├── elbow_method.png
│ ├── customer_segments.png
│ ├── cluster_summary.csv
│ ├── segmented_customers.csv
│ ├── kmeans_model.pkl
│ ├── scaler.pkl
│ └── pca.pkl
│
└── README.md

### Visualizations

--- Elbow Method

The Elbow Method plot is used to determine the optimal number of clusters.

Customer Segments (PCA Projection)

--- The PCA scatter plot visualizes all customers in two dimensions.

Each color represents a cluster.

--- Cluster Summary

The cluster_summary.csv file contains the average feature values for each cluster.

This helps interpret the characteristics of each customer segment.

### Installation

pip install pandas numpy matplotlib scikit-learn joblib

### How to Run

From the root directory of the repository:

py CustomerPersonality_KMEANS/src/customer_segmentation_kmeans_model.py
Real-World Applications
Customer Segmentation
Personalized Marketing
Targeted Promotions
Churn Prevention
Loyalty Programs
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Joblib

### Author

Jyotima Tomar

GitHub: https://github.com/jyotimajt11

### Conclusion

This project demonstrates how unsupervised learning can reveal hidden customer patterns without using labeled data.

By combining feature engineering, K-Means clustering, PCA, and business interpretation, the project provides actionable insights that can directly support data-driven marketing decisions.
