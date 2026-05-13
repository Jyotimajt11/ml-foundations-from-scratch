import os
import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#create outputs folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load Dataset
categories = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.misc"]

news = fetch_20newsgroups(subset="all", categories = categories, shuffle = True, random_state=42,  data_home="NaiveBayes/data")

X_text = news.data
y = news.target

#Covert text to word counts
vectorizer = CountVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(X_text)

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#train model 
model = MultinomialNB()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#Evaluate 
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=news.target_names,
)

print(f"Accuracy: {accuracy:.4f}")
print(report)

# Save report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# --------------------------------------------------
# Save model and vectorizer
# --------------------------------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "multinomial_nb_model.pkl"))
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "count_vectorizer.pkl"))

print("All outputs saved successfully.")


