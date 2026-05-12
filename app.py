from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load cleaned dataset
data = pd.read_csv("data/cleaned_news.csv").dropna(subset=["clean_text"])
tfidf_matrix = vectorizer.transform(data["clean_text"])

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    related_news = []

    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        confidence = round(model.predict_proba(vec).max() * 100, 2)

        similarity = cosine_similarity(vec, tfidf_matrix)[0]
        data["similarity"] = similarity
        real_news = data[data["label"] == "REAL"].sort_values(by="similarity", ascending=False)
        related_news = real_news["title"].head(5).tolist()

    return render_template("index.html", prediction=prediction, confidence=confidence, related_news=related_news)

if __name__ == "__main__":
    app.run(debug=True)
