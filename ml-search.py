import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

data = {
    'query': [
        'Shape of You by Ed Sheeran', 
        'Rolling in the Deep by Adele', 
        'Blinding Lights by The Weeknd', 
        'Bohemian Rhapsody by Queen', 
        'Thriller by Michael Jackson',
        'Abbey Road by Apple Records', 
        'Born to Run by Columbia Records',
        'Hotel California by Eagles', 
        'Like a Rolling Stone by Bob Dylan', 
        'Hey Jude by The Beatles', 
        'Purple Rain by Prince', 
        'Back in Black by AC/DC',
        'Random Access Memories by Columbia Records', 
        '21 by XL Recordings', 
        'Dark Side of the Moon by Harvest Records',
        'Bad Guy by Billie Eilish', 
        'Uptown Funk by Mark Ronson', 
        'Old Town Road by Lil Nas X', 
        'Thank U, Next by Ariana Grande', 
        'Shallow by Lady Gaga',
        'Thriller by Sony Music Group', 
        'Stairway to Heaven by Warner Music Group', 
        'Sweet Child O\' Mine by Universal Music Group',
        'Imagine by Universal Music Group', 
        'Billie Jean by Sony Music Group',
        'Smells Like Teen Spirit by Warner Music Group'
    ],
    'label': [
        'artist', 
        'artist', 
        'artist', 
        'artist', 
        'artist',
        'label', 
        'label',
        'artist', 
        'artist', 
        'artist', 
        'artist', 
        'artist',
        'label', 
        'label', 
        'label',
        'artist', 
        'artist', 
        'artist', 
        'artist', 
        'artist',
        'label', 
        'label', 
        'label',
        'label', 
        'label',
        'label'
    ]
}


# Convert to DataFrame
df = pd.DataFrame(data)

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
user_query = "SongName by weekend"
prediction = model.predict([user_query])[0]
print(f"Query classified as: {prediction}")

# Evaluate the model (optional)
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
