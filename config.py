"""
Shared configuration for the Trump vs AOC Netnography Study.
Centralizes constants, paths, and settings used across all scripts and notebooks.
"""

from pathlib import Path

# --- Target Users ---
TARGET_USERS = {
    "trump": {
        "screen_name": "realDonaldTrump",
        "user_id": "25073877",
    },
    "aoc": {
        "screen_name": "AOC",
        "user_id": "138203134",
    },
}

# --- Collection Settings ---
TWEETS_PER_USER = 100
MAX_REPLIES_PER_TWEET = None  # None = collect all available replies
RATE_LIMIT_PAUSE_TWEETS = 3  # seconds between tweet pagination calls
RATE_LIMIT_PAUSE_REPLIES = 2  # seconds between reply pagination calls
RATE_LIMIT_PAUSE_IMAGES = 0.5  # seconds between image downloads
RATE_LIMIT_BACKOFF_BASE = 60  # base seconds for exponential backoff on 429
RATE_LIMIT_MAX_RETRIES = 3

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SEED_DIR = DATA_DIR / "seed"
DATA_IMAGES_DIR = DATA_DIR / "images"
SEED_IMAGES_DIR = DATA_SEED_DIR / "images"
COOKIES_PATH = PROJECT_ROOT / "cookies.json"

# --- Visualization ---
COLOR_PALETTE = {
    "trump": "#E63946",
    "aoc": "#457B9D",
}

LABELS = {
    "en": {
        "trump": "Trump (@realDonaldTrump)",
        "aoc": "AOC (@AOC)",
        "likes": "Likes",
        "retweets": "Retweets",
        "replies": "Replies",
        "quotes": "Quotes",
        "bookmarks": "Bookmarks",
        "views": "Views",
        "engagement_rate": "Engagement Rate",
        "total_engagement": "Total Engagement",
        "tweet_count": "Tweet Count",
        "hour": "Hour of Day",
        "day_of_week": "Day of Week",
        "text_length": "Text Length (chars)",
        "word_count": "Word Count",
        "media_type": "Media Type",
        "sentiment": "Sentiment",
        "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    },
    "pt-br": {
        "trump": "Trump (@realDonaldTrump)",
        "aoc": "AOC (@AOC)",
        "likes": "Curtidas",
        "retweets": "Repostagens",
        "replies": "Respostas",
        "quotes": "Citações",
        "bookmarks": "Salvos",
        "views": "Visualizações",
        "engagement_rate": "Taxa de Engajamento",
        "total_engagement": "Engajamento Total",
        "tweet_count": "Quantidade de Tweets",
        "hour": "Hora do Dia",
        "day_of_week": "Dia da Semana",
        "text_length": "Tamanho do Texto (caracteres)",
        "word_count": "Contagem de Palavras",
        "media_type": "Tipo de Mídia",
        "sentiment": "Sentimento",
        "day_names": ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"],
    },
}

FIGURE_SIZE = (12, 6)
FIGURE_SIZE_SMALL = (8, 5)
FIGURE_SIZE_LARGE = (14, 8)

# --- Reproducibility ---
RANDOM_SEED = 42
