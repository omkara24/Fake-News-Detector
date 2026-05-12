import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data/cleaned_news.csv").dropna(subset=["clean_text"])

X = data["clean_text"]
y = data["label"]

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_tfidf = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")
