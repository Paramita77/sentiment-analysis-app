
import pandas as pd
import re

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)      # Remove HTML tags such as <br />
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and spaces
    text = text.lower()                      # Convert all text to lowercase
    return text

# Apply the clean_text function to the entire dataset
data['cleaned_review'] = data['review'].apply(clean_text)

# Compare the original and cleaned text
print("Original review:")
print(data['review'][0][:200])  # Display the first 200 characters

print("\nCleaned review:")
print(data['cleaned_review'][0][:200])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Convert sentiment into 0/1 labels (positive = 1, negative = 0)
data['label'] = data['sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0
)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_review'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Create a TF-IDF Vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions and check the model accuracy
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer for later use
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model saved successfully!")

