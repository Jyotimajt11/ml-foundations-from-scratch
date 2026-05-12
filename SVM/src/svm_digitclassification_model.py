import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


#output folder
os.makedirs("outputs", exist_ok = True)

#Load the dataset
digits = load_digits()

#for features and targets
X = digits.data
y = digits.target

#Print feature and target shape
print("feature shape:", X.shape)
print("target shape:", y.shape)

# Display first 10 digit images
plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {digits.target[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_digits.png"))
plt.show()

#split data into train and test sets
X_train, X_test, y_train, y_test, = train_test_split( X, y, test_size = 0.2, random_state = 42, stratify = y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#create SVM model
model = SVC(kernel='rbf', C=10, gamma='scale')

#train model
model.fit(X_train_scaled, y_train)

#make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names
)
disp.plot()

plt.title("SVM Digits Classification - Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

            

 