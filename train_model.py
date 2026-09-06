import pandas as pd
import re

# Dataset load করা
data = pd.read_csv('IMDB Dataset.csv')

# Text clean করার function
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)      # HTML tag বাদ দেওয়া (যেমন <br />)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # শুধু letter আর space রাখা, বাকি (number, punctuation) বাদ
    text = text.lower()                      # সব ছোট হাতের অক্ষরে (lowercase) আনা
    return text

# পুরো dataset এ clean_text function apply করা
data['cleaned_review'] = data['review'].apply(clean_text)

# আগে আর পরে তুলনা করে দেখা (প্রথম review)
print("Original review:")
print(data['review'][0][:200])  # প্রথম ২০০ character
print("\nCleaned review:")
print(data['cleaned_review'][0][:200])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Sentiment কে 0/1 এ convert করা (positive=1, negative=0)
data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Train এবং Test data আলাদা করা (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_review'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer বানানো (text কে numbers এ convert করার জন্য)
vectorizer = TfidfVectorizer(max_features=5000)  # সবচেয়ে গুরুত্বপূর্ণ ৫০০০ শব্দ ব্যবহার হবে
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model train করা
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Model দিয়ে prediction করা এবং accuracy check করা
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Model এবং vectorizer save করা (পরে ব্যবহার করার জন্য)
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model saved successfully!")