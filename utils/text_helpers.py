"""
Text processing utilities for the netnography study.
Provides functions for extracting features from tweet text.
"""

import re
from typing import Optional


# --- Regex Patterns ---
HASHTAG_PATTERN = re.compile(r"#(\w+)", re.UNICODE)
MENTION_PATTERN = re.compile(r"@(\w+)", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    re.UNICODE,
)

# --- Sentiment Keywords ---
POSITIVE_WORDS = frozenset({
    "great", "good", "love", "amazing", "best", "thank", "thanks", "support",
    "congratulations", "beautiful", "excellent", "proud", "win", "winning",
    "awesome", "fantastic", "wonderful", "hero", "brave", "truth", "success",
    "strong", "happy", "incredible", "perfect", "brilliant", "blessed",
    "grateful", "outstanding", "remarkable", "tremendous",
})

NEGATIVE_WORDS = frozenset({
    "worst", "terrible", "hate", "stupid", "lie", "lies", "liar", "corrupt",
    "corruption", "disaster", "fail", "failure", "wrong", "awful", "disgusting",
    "shame", "shameful", "pathetic", "criminal", "fraud", "destroy", "weak",
    "ugly", "horrible", "evil", "dangerous", "threat", "sad", "bad", "worse",
    "tragic", "scam", "fake", "racist", "fascist", "dictator",
})

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't",
    "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't", "rt", "amp", "https", "http", "co",
})


def extract_hashtags(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return HASHTAG_PATTERN.findall(text)


def extract_mentions(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return MENTION_PATTERN.findall(text)


def extract_urls(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return URL_PATTERN.findall(text)


def clean_text(text: Optional[str]) -> str:
    """Remove URLs, mentions, and hashtags for word-level analysis."""
    if not text:
        return ""
    cleaned = URL_PATTERN.sub("", text)
    cleaned = MENTION_PATTERN.sub("", cleaned)
    cleaned = HASHTAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def count_words(text: Optional[str]) -> int:
    if not text:
        return 0
    cleaned = clean_text(text)
    return len(cleaned.split()) if cleaned else 0


def caps_ratio(text: Optional[str]) -> float:
    """Ratio of uppercase letters to total letters."""
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def has_emoji(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(EMOJI_PATTERN.search(text))


def simple_sentiment(text: Optional[str]) -> str:
    """Keyword-based sentiment classification."""
    if not text:
        return "neutral"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def get_stopwords() -> frozenset:
    return STOPWORDS


def get_word_frequencies(texts: list[str], top_n: int = 30) -> list[tuple[str, int]]:
    """Get top N word frequencies from a list of texts, excluding stopwords."""
    word_counts: dict[str, int] = {}
    for text in texts:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        for word in cleaned.lower().split():
            word = re.sub(r"[^\w]", "", word)
            if word and word not in STOPWORDS and len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]
