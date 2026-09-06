import streamlit as st
import joblib
import re

# Saved model এবং vectorizer load করা
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text clean করার একই function (train_model.py থেকে কপি)
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Web App এর UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and find out if it's Positive or Negative!")

user_input = st.text_area("Write your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == 1:
            st.success("✅ This review is **Positive**! 😊")
        else:
            st.error("❌ This review is **Negative**! 😞")