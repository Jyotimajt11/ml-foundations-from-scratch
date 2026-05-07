import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

#outputs folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Load the dataset 
data = load_breast_cancer()
X = data.data
y = data.target

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

#Make predictions
y_pred = knn.predict(X_test_scaled)

#Evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)

print("knn classification model:")
print("----------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("\n Classification Report:")
print(report)

# Save classification report
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

with open(report_path, "w") as file:
    file.write("KNN Classification Model\n")
    file.write("------------------------\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)


# Save confusion matrix graph
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()
plt.title("Confusion Matrix - KNN")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()


# Accuracy vs K graph
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, predictions))

plt.figure()
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Value")
plt.xticks(k_values)
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_k.png"))
plt.show()


print("\nFiles saved in outputs folder:")
print("- classification_report.txt")
print("- confusion_matrix.png")
print("- accuracy_vs_k.png")




