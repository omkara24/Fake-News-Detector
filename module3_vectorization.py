import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

data = pd.read_csv("data/cleaned_news.csv")
data = data.dropna(subset=["clean_text"])

X = data["clean_text"]

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF Shape:", X_tfidf.shape)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("TF-IDF Vectorizer saved.")
