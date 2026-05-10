import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

# Get XGBoost project folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create outputs folder
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load the Dataset 
df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
)

#Drop customerID because it is only an ID, not useful for prediction
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)
# Convert TotalCharges from object to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = 'coerce')
# Remove rows with missing TotalCharges
df = df.dropna()

# Convert target column: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical columns into numeric columns
categorical_columns = df.select_dtypes(include=["object", "string"]).columns

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

#create XGBoost model
model = XGBClassifier(n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    subsample=0.8,  
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    eval_metric="logloss",
    random_state=42
)

# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("XGBoost Customer Churn Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("XGBoost Customer Churn Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()


# Feature Importance
feature_importance = model.feature_importances_

plt.figure(figsize=(10, 8))
plt.barh(X.columns, feature_importance)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Customer Churn Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.show()



