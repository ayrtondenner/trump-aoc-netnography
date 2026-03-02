"""
03_enrich_community.py - Community Feature Engineering
======================================================
Reads processed CSV data and raw JSON, computes user-level profiles,
mention networks, archetypes, and behavioral clusters.

Usage:
    python 03_enrich_community.py                # uses data/raw/ by default
    python 03_enrich_community.py --source seed  # uses data/seed/

Produces:
    data/processed/user_profiles.csv
    data/processed/mention_network.csv
    data/processed/community_metadata.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SEED_DIR,
    TARGET_USERS,
    RANDOM_SEED,
)
from utils.text_helpers import extract_mentions, caps_ratio


# Screen names to exclude from mention network (the politicians themselves)
PARENT_SCREEN_NAMES = {
    info["screen_name"].lower() for info in TARGET_USERS.values()
}


def load_data(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load processed CSVs and raw reply JSON."""
    tweets_path = DATA_PROCESSED_DIR / "tweets.csv"
    replies_path = DATA_PROCESSED_DIR / "replies.csv"

    if not tweets_path.exists() or not replies_path.exists():
        print("Error: Processed data not found. Run 02_enrich_data.py first.")
        sys.exit(1)

    tweets_df = pd.read_csv(tweets_path)
    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], utc=True)

    replies_df = pd.read_csv(replies_path)
    replies_df["created_at"] = pd.to_datetime(replies_df["created_at"], utc=True)

    # Load raw replies for fields not in processed CSV
    raw_replies = []
    for user_label in TARGET_USERS:
        path = source_dir / f"{user_label}_replies.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw_replies.extend(json.load(f))
            print(f"  Loaded raw replies for {user_label}")

    print(f"  Processed: {len(tweets_df)} tweets, {len(replies_df)} replies")
    print(f"  Raw replies: {len(raw_replies)}")
    return tweets_df, replies_df, raw_replies


def build_raw_lookup(raw_replies: list[dict]) -> dict:
    """Build lookup from reply_id to raw fields we need."""
    lookup = {}
    for r in raw_replies:
        rid = str(r.get("id", ""))
        lookup[rid] = {
            "user_id": str(r.get("user_id", "")),
            "reply_count": int(r.get("reply_count", 0) or 0),
            "quote_count": int(r.get("quote_count", 0) or 0),
            "bookmark_count": int(r.get("bookmark_count", 0) or 0),
        }
    return lookup


def build_user_profiles(
    replies_df: pd.DataFrame, tweets_df: pd.DataFrame, raw_lookup: dict
) -> pd.DataFrame:
    """Aggregate reply-level data into user-level profiles."""
    # Attach raw fields to replies_df
    replies_df = replies_df.copy()
    replies_df["reply_id_str"] = replies_df["reply_id"].astype(str)
    replies_df["raw_user_id"] = replies_df["reply_id_str"].map(
        lambda rid: raw_lookup.get(rid, {}).get("user_id", "")
    )
    replies_df["raw_reply_count"] = replies_df["reply_id_str"].map(
        lambda rid: raw_lookup.get(rid, {}).get("reply_count", 0)
    )

    # Compute caps_ratio per reply (not in processed CSV)
    replies_df["caps_ratio"] = replies_df["text"].fillna("").apply(caps_ratio)

    # Extract mentions per reply (excluding parent politician)
    def _extract_non_parent_mentions(row):
        mentions = extract_mentions(row.get("text", ""))
        return [m for m in mentions if m.lower() not in PARENT_SCREEN_NAMES]

    replies_df["other_mentions"] = replies_df.apply(_extract_non_parent_mentions, axis=1)
    replies_df["other_mention_count"] = replies_df["other_mentions"].apply(len)

    # Group by user
    grouped = replies_df.groupby("user_name")

    # Identity
    profiles = pd.DataFrame()
    profiles["user_name"] = grouped["user_name"].first().values
    profiles.index = profiles["user_name"]

    # user_id: take first non-empty
    user_ids = grouped["raw_user_id"].first()
    profiles["user_id"] = user_ids

    # Followers: max observed
    profiles["user_followers_count"] = grouped["user_followers_count"].max()

    # Verification: any True
    if "user_is_blue_verified" in replies_df.columns:
        profiles["user_is_blue_verified"] = grouped["user_is_blue_verified"].any()
    else:
        profiles["user_is_blue_verified"] = False

    # Activity volume
    profiles["total_replies"] = grouped.size()

    trump_counts = (
        replies_df[replies_df["parent_user"] == "trump"]
        .groupby("user_name")
        .size()
    )
    aoc_counts = (
        replies_df[replies_df["parent_user"] == "aoc"]
        .groupby("user_name")
        .size()
    )
    profiles["trump_replies"] = trump_counts.reindex(profiles.index, fill_value=0)
    profiles["aoc_replies"] = aoc_counts.reindex(profiles.index, fill_value=0)
    profiles["unique_tweets_replied"] = grouped["parent_tweet_id"].nunique()
    profiles["unique_politicians"] = (
        (profiles["trump_replies"] > 0).astype(int)
        + (profiles["aoc_replies"] > 0).astype(int)
    )
    profiles["is_cross_profile"] = profiles["unique_politicians"] == 2

    # Temporal behavior
    profiles["first_reply_at"] = grouped["created_at"].min()
    profiles["last_reply_at"] = grouped["created_at"].max()
    profiles["active_span_days"] = (
        (profiles["last_reply_at"] - profiles["first_reply_at"]).dt.total_seconds()
        / 86400.0
    )
    profiles["avg_reply_delay_minutes"] = grouped["reply_delay_minutes"].mean()
    profiles["median_reply_delay_minutes"] = grouped["reply_delay_minutes"].median()
    profiles["min_reply_delay_minutes"] = grouped["reply_delay_minutes"].min()

    # Content behavior
    profiles["avg_text_length"] = grouped["text_length"].mean()
    profiles["avg_word_count"] = grouped["word_count"].mean()
    profiles["text_length_std"] = grouped["text_length"].std().fillna(0)
    profiles["avg_hashtag_count"] = grouped["hashtag_count"].mean()
    profiles["mentions_parent_pct"] = grouped["has_mention_of_parent"].mean()
    profiles["avg_caps_ratio"] = grouped["caps_ratio"].mean()

    # Sentiment
    sentiment_counts = replies_df.groupby(["user_name", "sentiment_simple"]).size().unstack(fill_value=0)
    total_per_user = sentiment_counts.sum(axis=1)
    for sent in ["positive", "negative", "neutral"]:
        col = f"{sent}_pct"
        if sent in sentiment_counts.columns:
            profiles[col] = (sentiment_counts[sent] / total_per_user).reindex(profiles.index, fill_value=0)
        else:
            profiles[col] = 0.0

    profiles["dominant_sentiment"] = profiles[["positive_pct", "negative_pct", "neutral_pct"]].idxmax(axis=1).str.replace("_pct", "")
    profiles["sentiment_consistency"] = profiles[["positive_pct", "negative_pct", "neutral_pct"]].max(axis=1)

    # Engagement received
    profiles["total_favorites_received"] = grouped["favorite_count"].sum()
    profiles["avg_favorites_received"] = grouped["favorite_count"].mean()
    profiles["max_favorites_received"] = grouped["favorite_count"].max()
    profiles["total_views_received"] = grouped["view_count"].sum()
    profiles["avg_views_received"] = grouped["view_count"].mean()

    # Network (mentions)
    profiles["mentions_given"] = grouped["other_mention_count"].sum()
    profiles["unique_users_mentioned"] = replies_df.groupby("user_name")["other_mentions"].apply(
        lambda x: len(set(m for mentions in x for m in mentions))
    )

    # Conversation depth (from raw reply_count)
    profiles["replies_with_subreplies"] = grouped["raw_reply_count"].apply(lambda x: (x > 0).sum())
    profiles["total_subreply_count"] = grouped["raw_reply_count"].sum()
    profiles["avg_subreply_count"] = grouped["raw_reply_count"].mean()

    # Reset index
    profiles = profiles.reset_index(drop=True)

    return profiles, replies_df


def build_mention_network(replies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract @mention edges between repliers."""
    all_replier_names = set(replies_df["user_name"].str.lower().unique())

    edges = []
    for _, row in replies_df.iterrows():
        source = row["user_name"]
        mentions = row.get("other_mentions", [])
        parent = row.get("parent_user", "")
        if not mentions:
            continue
        for target in mentions:
            target_is_replier = target.lower() in all_replier_names
            edges.append({
                "source": source,
                "target": target,
                "parent_user": parent,
                "source_is_replier": True,
                "target_is_replier": target_is_replier,
            })

    if not edges:
        return pd.DataFrame(columns=["source", "target", "weight", "parent_user",
                                      "source_is_replier", "target_is_replier"])

    edges_df = pd.DataFrame(edges)

    # Aggregate to weighted edges
    weighted = (
        edges_df.groupby(["source", "target"])
        .agg(
            weight=("source", "size"),
            parent_user=("parent_user", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
            source_is_replier=("source_is_replier", "first"),
            target_is_replier=("target_is_replier", "first"),
        )
        .reset_index()
    )

    return weighted


def classify_archetypes(profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based archetype classification."""
    profiles = profiles.copy()

    def _classify(row):
        # 1. Bot suspect: high volume + low text variance + near-zero engagement
        if (row["total_replies"] >= 20
                and row["text_length_std"] < 15
                and row["avg_favorites_received"] < 0.1):
            return "bot_suspect"

        # 2. Influencer: high followers + gets engagement on replies
        if (row["user_followers_count"] >= 10000
                and row["avg_favorites_received"] >= 2):
            return "influencer"

        # 3. Troll: negative + mentions parent + high caps
        if (row["negative_pct"] >= 0.5
                and row["mentions_parent_pct"] >= 0.8
                and row["avg_caps_ratio"] >= 0.3):
            return "troll"

        # 4. Supporter: predominantly positive
        if row["positive_pct"] >= 0.5:
            return "supporter"

        # 5. Critic: predominantly negative
        if row["negative_pct"] >= 0.5:
            return "critic"

        # 6. Default
        return "casual"

    profiles["archetype"] = profiles.apply(_classify, axis=1)
    return profiles


def cluster_behaviors(profiles: pd.DataFrame) -> pd.DataFrame:
    """Run KMeans clustering on behavioral features."""
    profiles = profiles.copy()

    feature_cols = [
        "total_replies", "avg_text_length", "avg_favorites_received",
        "user_followers_count", "negative_pct", "positive_pct",
        "mentions_parent_pct", "avg_reply_delay_minutes",
    ]

    X = profiles[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
    profiles["behavior_cluster"] = kmeans.fit_predict(X_scaled)

    return profiles


def compute_mentions_received(profiles: pd.DataFrame, network_df: pd.DataFrame) -> pd.DataFrame:
    """Add mentions_received and unique_users_mentioned_by from network data."""
    profiles = profiles.copy()

    if network_df.empty:
        profiles["mentions_received"] = 0
        profiles["unique_users_mentioned_by"] = 0
        return profiles

    # Mentions received: sum of weights where user is target
    received = network_df.groupby("target")["weight"].sum()
    unique_by = network_df.groupby("target")["source"].nunique()

    profiles["mentions_received"] = profiles["user_name"].map(received).fillna(0).astype(int)
    profiles["unique_users_mentioned_by"] = profiles["user_name"].map(unique_by).fillna(0).astype(int)

    return profiles


def validate_outputs(profiles: pd.DataFrame, network_df: pd.DataFrame):
    """Run validation checks on output data."""
    print("\n  Validation checks:")
    print(f"    User profiles: {len(profiles)} rows x {len(profiles.columns)} columns")
    print(f"    Mention network: {len(network_df)} edges")

    # Archetype distribution
    print(f"\n    Archetype distribution:")
    for arch, count in profiles["archetype"].value_counts().items():
        pct = count / len(profiles) * 100
        print(f"      {arch}: {count} ({pct:.1f}%)")

    # Cluster distribution
    print(f"\n    Behavior cluster distribution:")
    for cluster, count in profiles["behavior_cluster"].value_counts().sort_index().items():
        print(f"      Cluster {cluster}: {count}")

    # Cross-profile users
    cross = profiles[profiles["is_cross_profile"]].shape[0]
    print(f"\n    Cross-profile users: {cross}")

    # Sanity checks
    assert profiles["user_name"].is_unique, "Duplicate user_names!"
    assert profiles["total_replies"].min() >= 1, "Users with 0 replies!"
    assert set(profiles["archetype"].unique()).issubset(
        {"supporter", "critic", "troll", "bot_suspect", "casual", "influencer"}
    ), "Unknown archetype!"

    print("    Validation passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Build community profiles and mention networks from reply data"
    )
    parser.add_argument(
        "--source", choices=["raw", "seed"], default="raw",
        help="Raw data source: 'raw' (data/raw/) or 'seed' (data/seed/)"
    )
    args = parser.parse_args()

    source_dir = DATA_RAW_DIR if args.source == "raw" else DATA_SEED_DIR

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        sys.exit(1)

    print(f"Loading data (raw from {source_dir})...")
    tweets_df, replies_df, raw_replies = load_data(source_dir)

    print("\nBuilding raw lookup...")
    raw_lookup = build_raw_lookup(raw_replies)
    print(f"  {len(raw_lookup)} raw reply records indexed")

    print("\nBuilding user profiles...")
    profiles, enriched_replies = build_user_profiles(replies_df, tweets_df, raw_lookup)
    print(f"  {len(profiles)} unique users profiled")

    print("\nBuilding mention network...")
    network_df = build_mention_network(enriched_replies)
    print(f"  {len(network_df)} weighted edges")

    print("\nComputing mentions received...")
    profiles = compute_mentions_received(profiles, network_df)

    print("\nClassifying archetypes...")
    profiles = classify_archetypes(profiles)

    print("\nClustering behaviors...")
    profiles = cluster_behaviors(profiles)

    # Validate
    validate_outputs(profiles, network_df)

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    profiles_path = DATA_PROCESSED_DIR / "user_profiles.csv"
    network_path = DATA_PROCESSED_DIR / "mention_network.csv"

    profiles.to_csv(profiles_path, index=False, encoding="utf-8")
    network_df.to_csv(network_path, index=False, encoding="utf-8")

    # Save metadata
    metadata = {
        "enrichment_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "total_users": len(profiles),
        "total_edges": len(network_df),
        "cross_profile_users": int(profiles["is_cross_profile"].sum()),
        "archetype_distribution": profiles["archetype"].value_counts().to_dict(),
        "cluster_distribution": profiles["behavior_cluster"].value_counts().to_dict(),
        "profiles_columns": list(profiles.columns),
        "network_columns": list(network_df.columns),
    }
    metadata_path = DATA_PROCESSED_DIR / "community_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nOutput saved to {DATA_PROCESSED_DIR}:")
    print(f"  user_profiles.csv: {len(profiles)} rows x {len(profiles.columns)} columns")
    print(f"  mention_network.csv: {len(network_df)} edges")
    print(f"  community_metadata.json")


if __name__ == "__main__":
    main()
