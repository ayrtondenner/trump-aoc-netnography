"""
02_enrich_data.py - Data Warehouse Script
==========================================
Reads raw JSON data from data/raw/ (or data/seed/), computes derived
features, and produces analysis-ready CSV files in data/processed/.

Usage:
    python 02_enrich_data.py                # uses data/raw/ by default
    python 02_enrich_data.py --source seed  # uses data/seed/

All transformations are deterministic (no external API calls, no randomness),
ensuring reproducibility when run against seed data.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SEED_DIR,
    TARGET_USERS,
)
from utils.text_helpers import (
    extract_hashtags,
    extract_mentions,
    extract_urls,
    count_words,
    caps_ratio,
    has_emoji,
    simple_sentiment,
    clean_text,
)


def load_raw_data(source_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load raw tweet and reply JSON files from the given directory."""
    all_tweets = []
    all_replies = []

    for user_label in TARGET_USERS:
        tweets_path = source_dir / f"{user_label}_tweets.json"
        replies_path = source_dir / f"{user_label}_replies.json"

        if tweets_path.exists():
            with open(tweets_path, "r", encoding="utf-8") as f:
                tweets = json.load(f)
                all_tweets.extend(tweets)
                print(f"  Loaded {len(tweets)} tweets for {user_label}")
        else:
            print(f"  Warning: {tweets_path} not found")

        if replies_path.exists():
            with open(replies_path, "r", encoding="utf-8") as f:
                replies = json.load(f)
                all_replies.extend(replies)
                print(f"  Loaded {len(replies)} replies for {user_label}")
        else:
            print(f"  Warning: {replies_path} not found")

    return all_tweets, all_replies


def classify_media_type(media_list) -> str:
    """Classify tweet media as none/photo/video/mixed."""
    if not media_list:
        return "none"
    types = {m.get("type", "unknown") for m in media_list}
    if len(types) > 1:
        return "mixed"
    media_type = types.pop()
    if media_type in ("photo",):
        return "photo"
    if media_type in ("video", "animated_gif"):
        return "video"
    return media_type


def enrich_tweets(raw_tweets: list[dict]) -> pd.DataFrame:
    """Transform raw tweet dicts into an enriched DataFrame."""
    if not raw_tweets:
        return pd.DataFrame()

    df = pd.DataFrame(raw_tweets)

    # Rename id to tweet_id
    df = df.rename(columns={"id": "tweet_id"})

    # User label
    if "user_label" not in df.columns:
        # Infer from user_name
        for user_label, info in TARGET_USERS.items():
            mask = df["user_name"].str.lower() == info["screen_name"].lower()
            df.loc[mask, "user_label"] = user_label
    df = df.rename(columns={"user_label": "user"})

    # Parse datetime
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")

    # --- Temporal Features ---
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = df["created_at"].dt.day_name()
    df["date"] = df["created_at"].dt.date
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # --- Text Features ---
    text_col = df["text"].fillna("")
    df["text_length"] = text_col.str.len()
    df["word_count"] = text_col.apply(count_words)
    df["hashtag_count"] = text_col.apply(lambda t: len(extract_hashtags(t)))
    df["hashtags_extracted"] = text_col.apply(lambda t: ",".join(extract_hashtags(t)))
    df["url_count"] = text_col.apply(lambda t: len(extract_urls(t)))
    df["mention_count"] = text_col.apply(lambda t: len(extract_mentions(t)))
    df["has_emoji"] = text_col.apply(has_emoji)
    df["exclamation_count"] = text_col.str.count("!")
    df["question_count"] = text_col.str.count(r"\?")
    df["caps_ratio"] = text_col.apply(caps_ratio)

    # --- Media Features ---
    df["media_type"] = df["media"].apply(classify_media_type)
    df["media_count"] = df["media"].apply(lambda m: len(m) if m else 0)
    df["has_card"] = df["has_card"].fillna(False).astype(bool)
    df["image_count"] = df["media"].apply(
        lambda m: sum(1 for x in m if x.get("type") == "photo") if m else 0
    )

    # Store image paths if available
    if "local_image_paths" in df.columns:
        df["image_paths"] = df["local_image_paths"].apply(
            lambda p: ",".join(p) if isinstance(p, list) else ""
        )
    else:
        df["image_paths"] = ""

    # --- Tweet Type ---
    df["is_retweet"] = df["retweeted_tweet_id"].notna()
    df["is_quote"] = df["is_quote_status"].fillna(False).astype(bool)
    df["is_reply"] = df["in_reply_to"].notna()

    # --- Engagement Features ---
    for col in ["reply_count", "favorite_count", "retweet_count", "quote_count", "bookmark_count", "view_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["total_engagement"] = (
        df["favorite_count"] + df["retweet_count"] + df["quote_count"] +
        df["bookmark_count"] + df["reply_count"]
    )

    df["engagement_rate"] = np.where(
        df["view_count"] > 0,
        df["total_engagement"] / df["view_count"],
        0.0
    )
    df["like_rate"] = np.where(df["view_count"] > 0, df["favorite_count"] / df["view_count"], 0.0)
    df["retweet_rate"] = np.where(df["view_count"] > 0, df["retweet_count"] / df["view_count"], 0.0)
    df["reply_rate"] = np.where(df["view_count"] > 0, df["reply_count"] / df["view_count"], 0.0)
    df["virality_score"] = np.where(
        df["view_count"] > 0,
        (df["retweet_count"] + df["quote_count"]) / df["view_count"],
        0.0
    )
    df["save_rate"] = np.where(df["view_count"] > 0, df["bookmark_count"] / df["view_count"], 0.0)

    # Select and order columns for output
    output_columns = [
        "tweet_id", "user", "user_name", "created_at", "text",
        "lang", "is_retweet", "is_quote", "is_reply",
        # Engagement
        "reply_count", "favorite_count", "retweet_count", "quote_count",
        "bookmark_count", "view_count",
        "total_engagement", "engagement_rate", "like_rate", "retweet_rate",
        "reply_rate", "virality_score", "save_rate",
        # Temporal
        "hour", "day_of_week", "day_name", "date", "is_weekend",
        # Text
        "text_length", "word_count", "hashtag_count", "hashtags_extracted",
        "url_count", "mention_count", "has_emoji", "exclamation_count",
        "question_count", "caps_ratio",
        # Media
        "media_type", "media_count", "image_count", "has_card", "image_paths",
    ]

    # Only include columns that exist
    output_columns = [c for c in output_columns if c in df.columns]
    return df[output_columns]


def enrich_replies(raw_replies: list[dict], tweets_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw reply dicts into an enriched DataFrame."""
    if not raw_replies:
        return pd.DataFrame()

    df = pd.DataFrame(raw_replies)

    # Rename id to reply_id
    df = df.rename(columns={"id": "reply_id"})

    # Parent user label
    if "parent_user_label" in df.columns:
        df = df.rename(columns={"parent_user_label": "parent_user"})
    elif "parent_tweet_id" in df.columns and not tweets_df.empty:
        tweet_user_map = tweets_df.set_index("tweet_id")["user"].to_dict()
        df["parent_user"] = df["parent_tweet_id"].map(tweet_user_map)

    # Parse datetime
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")

    # --- Text Features ---
    text_col = df["text"].fillna("")
    df["text_length"] = text_col.str.len()
    df["word_count"] = text_col.apply(count_words)
    df["hashtag_count"] = text_col.apply(lambda t: len(extract_hashtags(t)))

    # --- Reply Delay ---
    if "parent_tweet_id" in df.columns and not tweets_df.empty:
        tweet_times = tweets_df.set_index("tweet_id")["created_at"].to_dict()
        df["parent_created_at"] = df["parent_tweet_id"].map(tweet_times)
        df["parent_created_at"] = pd.to_datetime(df["parent_created_at"], utc=True)
        df["reply_delay_minutes"] = (
            (df["created_at"] - df["parent_created_at"]).dt.total_seconds() / 60.0
        )
        # Clamp negative delays (shouldn't happen but protect against clock skew)
        df["reply_delay_minutes"] = df["reply_delay_minutes"].clip(lower=0)
        df = df.drop(columns=["parent_created_at"])
    else:
        df["reply_delay_minutes"] = np.nan

    # --- Sentiment ---
    df["sentiment_simple"] = text_col.apply(simple_sentiment)

    # --- Mention of parent ---
    if "parent_user" in df.columns:
        parent_user_names = {}
        for user_label, info in TARGET_USERS.items():
            parent_user_names[user_label] = info["screen_name"].lower()

        def _mentions_parent(row):
            parent = row.get("parent_user")
            text = row.get("text", "")
            if not parent or not text:
                return False
            target_name = parent_user_names.get(parent, "")
            return target_name.lower() in text.lower()

        df["has_mention_of_parent"] = df.apply(_mentions_parent, axis=1)
    else:
        df["has_mention_of_parent"] = False

    # Numeric columns
    for col in ["reply_count", "favorite_count", "retweet_count", "quote_count", "bookmark_count", "view_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["user_followers_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Select output columns
    output_columns = [
        "reply_id", "parent_tweet_id", "parent_user",
        "user_name", "user_followers_count", "user_is_verified", "user_is_blue_verified",
        "created_at", "text", "lang",
        "favorite_count", "retweet_count", "view_count",
        # Computed
        "text_length", "word_count", "hashtag_count",
        "reply_delay_minutes", "sentiment_simple", "has_mention_of_parent",
    ]
    output_columns = [c for c in output_columns if c in df.columns]
    return df[output_columns]


def validate_tweets(df: pd.DataFrame):
    """Run validation checks on the enriched tweets DataFrame."""
    print("\n  Validation checks:")
    n = len(df)
    print(f"    Total tweets: {n}")

    users = df["user"].unique()
    print(f"    Users: {list(users)}")
    for u in users:
        count = len(df[df["user"] == u])
        print(f"      {u}: {count} tweets")

    # Engagement rate should be between 0 and 1 (with some tolerance for outliers)
    er = df["engagement_rate"]
    print(f"    Engagement rate: min={er.min():.6f}, max={er.max():.6f}, mean={er.mean():.6f}")

    # Hour range
    assert df["hour"].between(0, 23).all(), "Hour values out of range!"
    print(f"    Hour range: {df['hour'].min()} - {df['hour'].max()} (OK)")

    # Day of week range
    assert df["day_of_week"].between(0, 6).all(), "Day of week values out of range!"
    print(f"    Day of week range: {df['day_of_week'].min()} - {df['day_of_week'].max()} (OK)")

    # Non-null check on critical columns
    critical = ["tweet_id", "user", "created_at", "text"]
    for col in critical:
        nulls = df[col].isna().sum()
        if nulls > 0:
            print(f"    Warning: {col} has {nulls} null values")
        else:
            print(f"    {col}: no nulls (OK)")

    print("    Validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Enrich raw tweet data into analysis-ready CSVs")
    parser.add_argument(
        "--source", choices=["raw", "seed"], default="raw",
        help="Data source: 'raw' (data/raw/) or 'seed' (data/seed/)"
    )
    args = parser.parse_args()

    source_dir = DATA_RAW_DIR if args.source == "raw" else DATA_SEED_DIR

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        if args.source == "raw":
            print("Run 01_collect_data.py first to collect data.")
        sys.exit(1)

    print(f"Loading raw data from {source_dir}...")
    raw_tweets, raw_replies = load_raw_data(source_dir)

    if not raw_tweets:
        print("Error: No tweet data found.")
        sys.exit(1)

    print(f"\nEnriching {len(raw_tweets)} tweets...")
    tweets_df = enrich_tweets(raw_tweets)

    print(f"Enriching {len(raw_replies)} replies...")
    replies_df = enrich_replies(raw_replies, tweets_df)

    # Validate
    validate_tweets(tweets_df)

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tweets_path = DATA_PROCESSED_DIR / "tweets.csv"
    replies_path = DATA_PROCESSED_DIR / "replies.csv"

    tweets_df.to_csv(tweets_path, index=False, encoding="utf-8")
    replies_df.to_csv(replies_path, index=False, encoding="utf-8")

    # Save enrichment metadata
    metadata = {
        "enrichment_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "source_dir": str(source_dir),
        "tweets_count": len(tweets_df),
        "replies_count": len(replies_df),
        "tweets_columns": list(tweets_df.columns),
        "replies_columns": list(replies_df.columns),
        "tweets_date_range": {
            "min": str(tweets_df["date"].min()) if not tweets_df.empty else None,
            "max": str(tweets_df["date"].max()) if not tweets_df.empty else None,
        },
    }
    metadata_path = DATA_PROCESSED_DIR / "enrichment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nOutput saved to {DATA_PROCESSED_DIR}:")
    print(f"  tweets.csv: {len(tweets_df)} rows x {len(tweets_df.columns)} columns")
    print(f"  replies.csv: {len(replies_df)} rows x {len(replies_df.columns)} columns")
    print(f"  enrichment_metadata.json")

    # Print summary stats
    print(f"\n{'='*60}")
    print("ENRICHMENT SUMMARY")
    print(f"{'='*60}")
    print(tweets_df.groupby("user")[["favorite_count", "retweet_count", "view_count", "engagement_rate"]].mean().round(2).to_string())
    print(f"\nMedia type distribution:")
    print(tweets_df.groupby(["user", "media_type"]).size().unstack(fill_value=0).to_string())
    if not replies_df.empty:
        print(f"\nReply sentiment distribution:")
        print(replies_df.groupby(["parent_user", "sentiment_simple"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
