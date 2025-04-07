import string
from collections import Counter

import nltk
from nltk import bigrams
from nltk.corpus import stopwords


def check_and_download_nltk_resources():
    """Ensure that necessary NLTK resources are downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
        print("NLTK resources are available: tokenizers/punkt")
    except LookupError:
        print("Punkt tokenizer not found. Downloading...")
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
        print("NLTK resources are available: corpora/stopwords")
    except LookupError:
        print("Stopwords not found. Downloading...")
        nltk.download("stopwords")


def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing punctuation and numbers
    - Tokenizing the text into words
    - Removing stopwords
    """
    # Ensure the input is a string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = "".join(
        [char for char in text if char not in string.punctuation and not char.isdigit()]
    )

    # Tokenize the text (split into words)
    words = text.split()

    # Remove stopwords (common words like "the", "and", etc.)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def get_most_common_words(text_series, top_n=20):
    """
    Returns the most common words from a series of cleaned text.

    Args:
        text_series (pd.Series): A pandas Series containing cleaned text.
        top_n (int): Number of top words to return.

    Returns:
        List of tuples: [(word, frequency), ...]
    """
    all_text = " ".join(text_series.dropna())
    word_list = all_text.split()
    word_freq = Counter(word_list)
    return word_freq.most_common(top_n)


def get_top_bigrams(text_series, top_n=20):
    """
    Returns the top N most common bigrams from a series of cleaned text.

    Args:
        text_series (pd.Series): Cleaned text column.
        top_n (int): Number of top bigrams to return.

    Returns:
        List of tuples: [("word1 word2", count), ...]
    """
    all_words = " ".join(text_series.dropna()).split()
    bigram_list = list(bigrams(all_words))
    bigram_freq = Counter([" ".join(bg) for bg in bigram_list])
    return bigram_freq.most_common(top_n)
