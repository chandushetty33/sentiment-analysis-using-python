import pandas as pd
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

# 🔥 Improved Dataset with More Variety
data = {
    "text": [
        "chinthan is a boy","I love this product!", "Absolutely amazing experience!", "Fantastic service, very happy!",
        "Worst experience ever!", "I hate this!", "This is so disappointing.",
        "It's okay, nothing special.", "The product is average.", "It was decent but not great.",
        "I'm extremely happy with my purchase!", "Terrible, would not recommend.",
        "Great quality and fast shipping!", "Not bad, but could be better.",
        "Horrible experience, never coming back.", "Super happy with this!",
        "Worst decision I made.", "The movie was neither good nor bad.",
        "I'm satisfied with my order.", "The taste was awful!", "Best product ever!",
        "I'm indifferent about this.", "Service was just fine, not great, not terrible.",
        "Highly recommend this!", "I will never buy this again.", "Mediocre experience.",
        "I can't stop using this, it's so good!", "This is the absolute worst thing I've bought.",
        "Pretty average, nothing to complain about.", "Top-notch quality!", "So bad I regret buying this.",
        "I guess it's alright.", "Wonderful experience, very pleased!", "I'm neutral about this.",
    ],
    "sentiment": [
        "Positive","Positive", "Positive", "Positive", "Negative", "Negative", "Negative",
        "Neutral", "Neutral", "Neutral", "Positive", "Negative", "Positive",
        "Neutral", "Negative", "Positive", "Negative", "Neutral", "Neutral",
        "Negative", "Positive", "Neutral", "Neutral", "Positive", "Negative",
        "Neutral", "Positive", "Negative", "Neutral", "Positive", "Negative",
        "Neutral", "Positive", "Neutral"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert labels to numeric values
label_mapping = {"Positive": 1, "Negative": -1, "Neutral": 0}
df["sentiment"] = df["sentiment"].map(label_mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

# 🔹 Train vectorizer on FULL text dataset (train + test)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔥 Train a better Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Save the trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")  

print("✅ Model and Vectorizer saved successfully!")
