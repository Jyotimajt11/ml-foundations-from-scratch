import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
     accuracy_score,
     confusion_matrix,
     classification_report,
     roc_curve,
     auc)

#load the dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

#print dataset info
print("Dataset Shape:", X.shape)
print("\n First 5 rows:", X.head())
print("\n Target names:", data.target_names)
print("\n Missing Values:", X.isnull().sum())

#split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training of model
model=LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\n Model Accuracy:", accuracy)
print("\n Confusion Matrix:", cm)

classification_report(y_test, y_pred, target_names=data.target_names)
print("\n Classification Report:", classification_report)

#feature importance 
print("\n Feature importance:",)
for feature, weight in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {weight}")

#Graph:1 Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=data.target_names,
    yticklabels=data.target_names

)
plt.title("Confusion Matrix")
plt.title("Predicted Class")
plt.title("Actual Class")
plt.tight_layout()

plt.savefig("Logistic_Regression/outputs/confusion_matrix.png", dpi=300)
plt.show()

#Graph:2 ROC Curve
fpr, tpr, thresholds =roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig("Logistic_Regression/outputs/roc_curve.png", dpi=300)
plt.show()
  
    
#Graph 3: Feature Importance

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
})

feature_importance["Absolute Coefficient"] = np.abs(
    feature_importance["Coefficient"]
)

feature_importance = feature_importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
).head(10)

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=feature_importance
)

plt.title("Top 10 Important Features")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("Logistic_Regression/outputs/feature_importance.png", dpi=300)
plt.show()




