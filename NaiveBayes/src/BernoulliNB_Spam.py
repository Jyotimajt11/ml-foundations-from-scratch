import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import(accuracy_score, classification_report, confusion_matrix)

# Create paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "spam.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load Dataset
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Keep only useful columns
# Kaggle SMS spam dataset usually has columns v1 and v2
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
elif "label" in df.columns and "message" in df.columns:
    df = df[["label", "message"]]
else:
    raise ValueError("Dataset must contain either v1/v2 or label/message columns.")

# Convert labels into numbers
# ham = 0, spam = 1
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove empty rows if any
df = df.dropna()

X_text = df["message"]
y = df["label"]

# Convert text into binary features
# binary=True means:
# word present = 1
# word absent = 0
vectorizer = CountVectorizer(
    stop_words="english",
    binary=True
)

X = vectorizer.fit_transform(X_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train BernoulliNB model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
)

print(f"Accuracy: {accuracy:.4f}")
print(report)

# Save classification report
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

with open(report_path, "w", encoding="utf-8") as file:
    file.write("BernoulliNB Spam Classification Report\n")
    file.write("=" * 45)
    file.write("\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("BernoulliNB - Spam Classification")
plt.tight_layout()

# Save graph
graph_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(graph_path)
plt.show()

# Save model and vectorizer
model_path = os.path.join(OUTPUT_DIR, "bernoulli_nb_spam_model.pkl")
vectorizer_path = os.path.join(OUTPUT_DIR, "bernoulli_count_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nSaved files:")
print(f"Report: {report_path}")
print(f"Confusion Matrix: {graph_path}")
print(f"Model: {model_path}")
print(f"Vectorizer: {vectorizer_path}")
