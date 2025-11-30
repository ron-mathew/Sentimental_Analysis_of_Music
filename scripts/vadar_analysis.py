import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER data for nltk 
nltk.download('vader_lexicon')

# Loading Dataset
data_path = "path_to_your_dataset.csv"
df = pd.read_csv(data_path)

# Inspecting dataset
print(df.head())

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to label emotions based on sentiment score
def get_emotion_label(text):
    sentiment_score = sid.polarity_scores(text)['compound']
    if sentiment_score >= 0.5:
        return "happy"
    elif sentiment_score >= 0.1:
        return "calm"
    elif sentiment_score <= -0.5:
        return "sad"
    elif sentiment_score <= -0.1:
        return "angry"
    else:
        return "neutral"

# Apply the function to each lyric entry in the dataset
df['emotion'] = df['lyrics'].apply(get_emotion_label)

# Inspect the labeled dataset
print(df.head())

# Save the new dataset with emotion labels
df.to_csv("path_to_your_labeled_dataset.csv", index=False)
print("Emotion labels added and saved to 'path_to_your_labeled_dataset.csv'")
