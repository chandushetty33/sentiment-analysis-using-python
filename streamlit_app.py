import streamlit as st
import joblib
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained model & vectorizer
model_path = os.path.join(current_dir, "sentiment_model.pkl")
vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError("🚨 Model or Vectorizer file not found!")

# Streamlit UI
st.title("🔍 Sentiment Analysis App")
st.write("Enter a text and see if it's Positive, Negative, or Neutral.")

# Text input
user_input = st.text_area("Enter text:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Transform input text
        input_tfidf = vectorizer.transform([user_input.lower()])  # Convert to lowercase
        sentiment = model.predict(input_tfidf)[0]
        
        sentiment_map = {1: "Positive ✅", -1: "Negative ❌", 0: "Neutral ⚖"}
        st.subheader(f"Sentiment: {sentiment_map[sentiment]}")
    else:
        st.warning("Please enter some text!")
