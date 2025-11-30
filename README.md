# Sentiment Analysis of Kendrick Lamar’s Lyrics Using ALBERT

## Overview
This project performs sentiment analysis on Kendrick Lamar’s music lyrics by fine-tuning ALBERT (A Lite BERT), a compact and computationally efficient transformer model.  
The goal is to understand the emotional, social, and political themes present in Kendrick Lamar’s songs using a custom dataset of scraped lyrics.

---

## Workflow

### 1. Data Collection
Lyrics and metadata such as song title, album, and release year are scraped from Genius and saved as CSV/JSON.

Tools Used:  
- Python  
- BeautifulSoup or Scrapy  
- Genius API (optional)

### 2. Data Preprocessing
Collected lyrics are cleaned and prepared for modeling through:

- Removing special characters and noise  
- Lemmatization and tokenization  
- Removing repeated hooks/ad-libs  
- Regex-based normalization  

Tools Used:  
- NLTK  
- SpaCy  
- Regex

### 3. Fine-Tuning ALBERT
The cleaned dataset is used to fine-tune ALBERT for sentiment classification.  
Its lightweight architecture ensures efficient training without major loss in accuracy.

Tools Used:  
- Hugging Face Transformers  
- PyTorch or TensorFlow  
- Google Colab / AWS

### 4. Sentiment Analysis
The fine-tuned model predicts the sentiment polarity of Kendrick Lamar’s lyrics, enabling deeper interpretation of emotional and thematic content.

---

## Dataset Structure
Each dataset record contains:

- Song Title  
- Album  
- Release Year  
- Lyrics

All lyrics are preprocessed and tokenized before being fed into ALBERT.

---

# Project Structure

## Directory Tree

## Project Structure Table
| Path | Description |
|------|-------------|
| `data/` | Stores the scraped Kendrick Lamar lyrics dataset |
| `models/` | Contains fine-tuned ALBERT checkpoints |
| `notebooks/` | Jupyter notebooks for experiments and EDA |
| `scripts/` | Scripts for scraping, preprocessing, model training, and evaluation |
| `scripts/scrape_lyrics.py` | Scrapes lyrics and metadata from Genius |
| `scripts/preprocess_data.py` | Cleans and tokenizes raw lyrics |
| `scripts/fine_tune_albert.py` | Fine-tuning pipeline for ALBERT |
| `scripts/evaluate_model.py` | Evaluates the trained model |

---

