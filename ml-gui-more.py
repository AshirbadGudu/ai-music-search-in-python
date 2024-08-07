import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import tkinter as tk
from tkinter import messagebox
import re

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

# Function to classify the query and extract details
def classify_query():
    user_query = entry.get()
    prediction = model.predict([user_query])[0]

    # Extract details using regular expressions
    match = re.match(r"(.*) by (.*)", user_query, re.IGNORECASE)
    if match:
        song, artist_or_label = match.groups()
        song = song.strip()
        artist_or_label = artist_or_label.strip()
    else:
        song = user_query
        artist_or_label = "Unknown"
    
    # Update result label with detailed information
    if prediction == 'artist':
        result_label.config(text=f"Query classified as: Artist\nSong: {song}\nArtist: {artist_or_label}")
    else:
        result_label.config(text=f"Query classified as: Label\nSong: {song}\nLabel: {artist_or_label}")

# Create the GUI
root = tk.Tk()
root.title("Query Classifier")

# Input field
entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=10)

# Search button
search_button = tk.Button(root, text="Search", command=classify_query)
search_button.pack(pady=5)

# Result label
result_label = tk.Label(root, text="", justify=tk.LEFT)
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
