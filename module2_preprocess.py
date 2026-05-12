import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

fake_df = pd.read_csv("data/fake.csv")
true_df = pd.read_csv("data/true.csv")

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Combine title + text
data["combined"] = data["title"] + " " + data["text"]
data["clean_text"] = data["combined"].apply(clean_text)

data.to_csv("data/cleaned_news.csv", index=False)
print("Cleaned data saved with title + text combined.")
