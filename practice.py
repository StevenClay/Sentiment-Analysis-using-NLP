# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
dataset = pd.read_csv('Tweets.csv')
ax = dataset.value_counts('sentiment').plot(kind='bar')
ax.set_title("Sentiment Distribution")
plt.show()

# Preprocessing function
def preprocess_text(text):

    if not isinstance(text, str):
        return ""
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the dataset
dataset['preprocessed_text'] = dataset['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['preprocessed_text'], dataset['sentiment'], test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Convert text data to input format for BERT
inputs_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors="pt")
inputs_test = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt")