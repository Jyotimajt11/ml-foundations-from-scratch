import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Create the outputs directory
os.makedirs("outputs", exist_ok=True)
#Load the dataset
df = pd.read_csv("data/heart.csv")
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]
#split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)

#Train the model
model.fit(X_train, y_train)
#prediction
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy:.2f}")
print("\n Claassification Report:")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Heart Disease", "Heart Disease"])
disp.plot()
plt.title("Heart Disease Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

#decision tree visualization
plt.figure(figsize=(22,10))

plot_tree(model, filled=True, feature_names=X.columns, class_names=["No heart disease", "heart disease"])
plt.title("Decision Tree for Heart Disease Prediction")
plt.savefig("outputs/decision_tree.png")
plt.show()

#Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
feature_importance = feature_importance.sort_values( by="Importance",
    ascending=False)
print("\n Feature Importance:")
print(feature_importance)

#feature importance graph
plt.figure(figsize=(10,6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Heart Disease Prediction")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

# Custom Prediction Example
sample_patient = [[
    52,  # age
    1,   # sex
    0,   # cp
    125, # trestbps
    212, # chol
    0,   # fbs
    1,   # restecg
    168, # thalach
    0,   # exang
    1.0, # oldpeak
    2,   # slope
    2,   # ca
    3    # thal
]]

prediction = model.predict(sample_patient)

print("\nSample Patient Prediction:")

if prediction[0] == 1:
    print("Heart Disease Risk: YES")
else:
    print("Heart Disease Risk: NO")







