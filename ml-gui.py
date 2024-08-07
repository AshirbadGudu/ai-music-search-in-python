import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import tkinter as tk
from tkinter import messagebox

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

# Function to classify the query
def classify_query():
    user_query = entry.get()
    prediction = model.predict([user_query])[0]
    result_label.config(text=f"Query classified as: {prediction}")

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
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
