"""Agricultural Text Preprocessing Module.

Provides robust text cleaning, normalization, stop-word management with
preservation of essential agricultural domain terminology (crops, symptoms,
nutrients, units), and an sklearn-compatible Transformer for seamless
integration into ML pipelines.
"""

import re
import string
from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Core English stop words to filter out non-informative words
STANDARD_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Explicit agricultural domain terms and keywords that must NEVER be stripped
AGRICULTURAL_KEYWORDS_PRESERVE = {
    # Crops and commodities
    "rice", "paddy", "wheat", "cotton", "maize", "corn", "sugarcane", "tomato",
    "potato", "onion", "chili", "mustard", "soybean", "gram", "pulses", "barley",
    "bajra", "sorghum", "banana", "mango", "citrus", "garlic", "ginger", "cabbage",
    # Nutrients and chemistry
    "npk", "urea", "dap", "potash", "zinc", "sulfur", "boron", "nitrogen", "phosphorus",
    "potassium", "gypsum", "iron", "calcium",
    # Soil and inputs
    "soil", "sand", "clay", "loam", "seed", "seeds", "hybrid", "compost", "water",
    "irrigation", "drip", "sprinkler", "mulch", "manure", "jeevamrut", "panchagavya",
    # Diseases and pests
    "rust", "blight", "rot", "wilt", "mildew", "smut", "spot", "borer", "aphid",
    "weevil", "armyworm", "whitefly", "thrips", "mite", "weed", "herbicides", "pesticides",
    # Practices and metrics
    "msp", "mandi", "price", "scheme", "kisan", "subsidy", "yield", "acre", "hectare",
    "sowing", "harvest", "transplanting", "weather", "rain", "forecast", "temperature"
}

# Remove agricultural tokens from stop words to ensure critical meaning is retained
EFFECTIVE_STOP_WORDS = {w for w in STANDARD_STOP_WORDS if w not in AGRICULTURAL_KEYWORDS_PRESERVE}


def clean_text(text: Union[str, float, None], max_len: int = 1000) -> str:
    """Cleans, normalizes, and filters an agricultural question string.

    Args:
        text: Raw input string or value.
        max_len: Max allowed length to guard against malformed or buffer overflow inputs.

    Returns:
        Cleaned, normalized, lowercased string of tokens.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Strip and handle empty
    text = text.strip()
    if not text:
        return ""

    # Truncate unusually long queries to safeguard processing
    if len(text) > max_len:
        text = text[:max_len]

    # Lowercase normalization
    text = text.lower()

    # Expand common contractions for consistency
    contractions = {
        "can't": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am",
    }
    for cont, expanded in contractions.items():
        text = text.replace(cont, expanded)

    # Remove special punctuation, keeping alphanumeric characters and spaces
    # Preserve hyphens as space
    text = re.sub(r"[-_/]+", " ", text)
    # Remove all remaining punctuation
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)

    # Tokenize by whitespace
    tokens = text.split()

    # Filter out stop words while preserving agricultural keywords
    filtered_tokens = [
        token for token in tokens
        if len(token) > 1 and (token in AGRICULTURAL_KEYWORDS_PRESERVE or token not in EFFECTIVE_STOP_WORDS)
    ]

    # Normalize whitespace
    cleaned = " ".join(filtered_tokens).strip()

    # Fallback to normalized raw alphanumeric string if all tokens were removed
    if not cleaned and tokens:
        cleaned = " ".join(tokens)

    return cleaned


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that applies clean_text to an iterable of texts.

    Ensures 100% consistency between model training and online inference.
    """

    def __init__(self, max_len: int = 1000):
        self.max_len = max_len

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        if isinstance(X, pd.Series):
            return [clean_text(val, self.max_len) for val in X]
        elif isinstance(X, (list, tuple, np.ndarray)):
            return [clean_text(val, self.max_len) for val in X]
        elif isinstance(X, str):
            return [clean_text(X, self.max_len)]
        else:
            return [clean_text(str(X), self.max_len)]
