import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Take input from user
news = input("Enter News Text: ")

cleaned = clean_text(news)
vector = vectorizer.transform([cleaned])

prediction = model.predict(vector)[0]
probability = model.predict_proba(vector).max()

print("\nPrediction:", prediction)
print("Confidence:", round(probability * 100, 2), "%")
