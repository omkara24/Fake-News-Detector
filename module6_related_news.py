import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load data
data = pd.read_csv("data/cleaned_news.csv").dropna(subset=["clean_text"])

# Load vectorizer and model
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Vectorize all news
tfidf_matrix = vectorizer.transform(data["clean_text"])

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Input news
query = input("Enter News Text: ")
query_clean = clean_text(query)
query_vec = vectorizer.transform([query_clean])

# Compute similarity
similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

# Get top 5 similar news (REAL only)
data["similarity"] = similarity_scores
real_news = data[data["label"] == "REAL"].sort_values(by="similarity", ascending=False)

print("\nTop 5 Related REAL News:\n")
for i, row in real_news.head(5).iterrows():
    print("-", row["title"])
