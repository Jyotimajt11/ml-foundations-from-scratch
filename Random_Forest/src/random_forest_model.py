import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
 
#Load the dataset
df = pd.read_csv("data/heart.csv")

#separate features and target
X = df.drop("target", axis = 1)
y = df["target"]

#split data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#create random forest model
model = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state = 42)

#train the model 
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap="Blues")

ax.set_title("Random Forest Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

#Feature Importance 
feature_importance = model.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(X.columns, feature_importance)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.savefig("outputs/feature_importance.png")
plt.show()

