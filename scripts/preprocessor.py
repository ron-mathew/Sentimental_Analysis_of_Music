import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('/content/kendrick.lyrics.csv')

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Custom negation handling function
def handle_negations(text):
    negation_tokens = ["not", "n't", "never", "no"]
    words = text.split()
    for i, word in enumerate(words):
        if word in negation_tokens and i+1 < len(words):
            words[i+1] = "NOT_" + words[i+1]  # Mark the next word after negation
    return ' '.join(words)

# Function to preprocess text
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Handle negations
    text = handle_negations(' '.join(words))
    words = text.split()

    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Apply preprocessing to lyrics column
df['cleaned_lyrics'] = df['lyrics'].apply(preprocess_text)

# Remove repetitive phrases
def remove_repetitive_phrases(text, min_count=2):
    word_counts = Counter(text.split())
    words = [word for word in text.split() if word_counts[word] < min_count]
    return ' '.join(words)

df['cleaned_lyrics'] = df['cleaned_lyrics'].apply(remove_repetitive_phrases)

# Show sample of the processed data
print(df[['lyrics', 'cleaned_lyrics']].head())

# Save the cleaned data to a new CSV file
df.to_csv('kendrick_cleaned_lyrics.csv', index=False)
