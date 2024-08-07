import spacy

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Example lists of artist names and label names
artists = ["John Doe", "Jane Smith", "ArtistName"]
labels = ["XYZ Records", "LabelName"]

def identify_search_context(query):
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Convert artist and label lists to lowercase for case-insensitive matching
    lower_artists = [artist.lower() for artist in artists]
    lower_labels = [label.lower() for label in labels]
    
    found_artist = None
    found_label = None
    
    for artist in lower_artists:
        if artist in query_lower:
            found_artist = artist
            break  # Stop searching once we find a match
    
    for label in lower_labels:
        if label in query_lower:
            found_label = label
            break  # Stop searching once we find a match
    
    if found_artist:
        return f"Searching for songs by artist: {found_artist.title()}"
    elif found_label:
        return f"Searching for songs by label: {found_label.title()}"
    else:
        return "No matching artist or label found."

# Example usage
user_query = "SongName by XYZ"
print(identify_search_context(user_query))
