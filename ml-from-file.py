import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data from CSV file
df = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Logistic Regression classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'query_classifier.pkl')

# Load the model
model = joblib.load('query_classifier.pkl')

# Example usage
user_query = "SongName by The sony"
prediction = model.predict([user_query])[0]
print(f"Query classified as: {prediction}")

# Evaluate the model (optional)
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
