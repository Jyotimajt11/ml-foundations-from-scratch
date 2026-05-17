import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Create folders
import os

# Current file directory: .../CustomerPersonality_KMEANS/src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent project directory: .../CustomerPersonality_KMEANS
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Correct paths
DATA_PATH = os.path.join(PROJECT_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Create outputs folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Data path:", DATA_PATH)
print("Output path:", OUTPUT_DIR)

#Load the dataset
df = pd.read_csv(DATA_PATH, sep="\t")
print("Dataset Loaded Successfully")
print(df.head())

# Fill missing values
df["Income"] = df["Income"].fillna(df["Income"].median())

# Create Age feature
df["Age"] = 2026 - df["Year_Birth"]

# Total spending
df["Total_Spending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)

# Total purchases
df["Total_Purchases"] = (
    df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)

# Number of children
df["Children"] = df["Kidhome"] + df["Teenhome"]


# Select features for clustering
features = [
    "Age",
    "Income",
    "Recency",
    "Total_Spending",
    "Total_Purchases",
    "NumWebVisitsMonth",
    "Children"
]

X = df[features]

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#ELBOW METHOD
inertia_values = []

for k in range(1,11):
  model = KMeans(n_clusters = k, random_state=42, n_init=10)
  model.fit(X_scaled)
  inertia_values.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_values, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia / WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_method.png"))
plt.show()

#Train the final KMeans Model
kmeans = KMeans(n_clusters =4, random_state = 42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# Silhouette score
score = silhouette_score(X_scaled, clusters)
print("Silhoutte Score", score)

#Cluster Summary
cluster_summary = df.groupby("Cluster")[features].mean()
print(cluster_summary)

cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))

#PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]

plt.figure(figsize=(8, 6))
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segments using K-Means")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "customer_segments.png"))
plt.show()


# Save final dataset and model
df.to_csv(os.path.join(OUTPUT_DIR, "segmented_customers.csv"), index=False)
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca.pkl"))

print("K-Means clustering completed successfully!")
print("All files saved in outputs folder.")



