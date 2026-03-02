"""
Data loading utilities for the netnography study.
Provides functions to load processed or seed data for notebook analysis.
"""

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED_DIR, DATA_SEED_DIR, DATA_IMAGES_DIR, SEED_IMAGES_DIR


def _resolve_source(source: str) -> Path:
    """Determine which data directory to load from."""
    if source == "processed":
        return DATA_PROCESSED_DIR
    elif source == "seed":
        return DATA_SEED_DIR
    else:  # auto
        if (DATA_PROCESSED_DIR / "tweets.csv").exists():
            return DATA_PROCESSED_DIR
        elif (DATA_SEED_DIR / "tweets.csv").exists():
            return DATA_SEED_DIR
        else:
            raise FileNotFoundError(
                "No data found. Run 02_enrich_data.py first, or ensure seed data exists in data/seed/"
            )


def load_tweets(source: str = "auto") -> pd.DataFrame:
    """Load the enriched tweets DataFrame.

    Args:
        source: "auto" (try processed, then seed), "processed", or "seed".

    Returns:
        DataFrame with all enriched tweet columns.
    """
    data_dir = _resolve_source(source)
    df = pd.read_csv(data_dir / "tweets.csv")

    # Parse datetime
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Categoricals
    if "user" in df.columns:
        df["user"] = df["user"].astype("category")
    if "media_type" in df.columns:
        df["media_type"] = df["media_type"].astype("category")
    if "day_name" in df.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["day_name"] = pd.Categorical(df["day_name"], categories=day_order, ordered=True)

    print(f"Loaded {len(df)} tweets from {data_dir.name}/")
    return df


def load_replies(source: str = "auto") -> pd.DataFrame:
    """Load the enriched replies DataFrame.

    Args:
        source: "auto" (try processed, then seed), "processed", or "seed".

    Returns:
        DataFrame with all enriched reply columns.
    """
    data_dir = _resolve_source(source)
    df = pd.read_csv(data_dir / "replies.csv")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if "parent_user" in df.columns:
        df["parent_user"] = df["parent_user"].astype("category")
    if "sentiment_simple" in df.columns:
        df["sentiment_simple"] = df["sentiment_simple"].astype("category")

    print(f"Loaded {len(df)} replies from {data_dir.name}/")
    return df


def load_raw_tweets(source: str = "auto") -> dict[str, list[dict]]:
    """Load raw tweet JSON files. Returns dict keyed by user label."""
    if source == "auto":
        if (DATA_PROCESSED_DIR.parent / "raw").exists() and any((DATA_PROCESSED_DIR.parent / "raw").glob("*_tweets.json")):
            raw_dir = DATA_PROCESSED_DIR.parent / "raw"
        elif DATA_SEED_DIR.exists():
            raw_dir = DATA_SEED_DIR
        else:
            raise FileNotFoundError("No raw data found.")
    elif source == "seed":
        raw_dir = DATA_SEED_DIR
    else:
        raw_dir = DATA_PROCESSED_DIR.parent / "raw"

    result = {}
    for user_label in ["trump", "aoc"]:
        path = raw_dir / f"{user_label}_tweets.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                result[user_label] = json.load(f)
    return result


def load_user_profiles(source: str = "auto") -> pd.DataFrame:
    """Load the user profiles DataFrame.

    Args:
        source: "auto" (try processed, then seed), "processed", or "seed".

    Returns:
        DataFrame with one row per unique replier.
    """
    data_dir = _resolve_source(source)
    path = data_dir / "user_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(
            "user_profiles.csv not found. Run 03_enrich_community.py first."
        )
    df = pd.read_csv(path)

    for col in ["first_reply_at", "last_reply_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    if "archetype" in df.columns:
        df["archetype"] = df["archetype"].astype("category")
    if "dominant_sentiment" in df.columns:
        df["dominant_sentiment"] = df["dominant_sentiment"].astype("category")

    print(f"Loaded {len(df)} user profiles from {data_dir.name}/")
    return df


def load_mention_network(source: str = "auto") -> pd.DataFrame:
    """Load the mention network edge list.

    Args:
        source: "auto" (try processed, then seed), "processed", or "seed".

    Returns:
        DataFrame with weighted mention edges.
    """
    data_dir = _resolve_source(source)
    path = data_dir / "mention_network.csv"
    if not path.exists():
        raise FileNotFoundError(
            "mention_network.csv not found. Run 03_enrich_community.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} mention edges from {data_dir.name}/")
    return df


def get_images_dir(source: str = "auto") -> Path:
    """Get the appropriate images directory based on data source."""
    if source == "processed" or source == "auto":
        if DATA_IMAGES_DIR.exists() and any(DATA_IMAGES_DIR.rglob("*.jpg")):
            return DATA_IMAGES_DIR
    if SEED_IMAGES_DIR.exists() and any(SEED_IMAGES_DIR.rglob("*.jpg")):
        return SEED_IMAGES_DIR
    if DATA_IMAGES_DIR.exists():
        return DATA_IMAGES_DIR
    return SEED_IMAGES_DIR
