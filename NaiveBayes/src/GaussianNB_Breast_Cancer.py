import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import(accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

#create outputs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

#Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Train model
model = GaussianNB()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=data.target_names,
)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gaussian Naive Bayes - Breast Cancer")
plt.tight_layout()

# Save confusion matrix
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# Show plot
plt.show()

# Save model
joblib.dump(model, os.path.join(OUTPUT_DIR, "gaussian_nb_model.pkl"))

print("All outputs saved successfully.")
