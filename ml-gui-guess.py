import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import tkinter as tk
from tkinter import messagebox

# Load data from CSV file
df = pd.read_csv('queries.csv')

# Prepare the feature and target variables
X = df['query']
y = df[['song', 'artist', 'label']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and MultiOutputClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression()))
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
    
    song, artist, label = prediction
    
    # Update result label with detailed information
    result_label.config(text=f"Song: {song}\nArtist: {artist}\nLabel: {label}")

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
