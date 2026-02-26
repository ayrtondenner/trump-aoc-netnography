"""
01_collect_data.py - Data Lake Script
=====================================
Connects to Twitter/X via twikit, downloads the latest 100 tweets from
Trump and AOC, fetches all reply threads, downloads tweet images,
and saves everything as raw JSON to data/raw/.

Usage:
    python 01_collect_data.py

Requires a .env file with:
    TWITTER_USERNAME=...
    TWITTER_EMAIL=...
    TWITTER_PASSWORD=...
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    TARGET_USERS,
    TWEETS_PER_USER,
    MAX_REPLIES_PER_TWEET,
    RATE_LIMIT_PAUSE_TWEETS,
    RATE_LIMIT_PAUSE_REPLIES,
    RATE_LIMIT_PAUSE_IMAGES,
    RATE_LIMIT_BACKOFF_BASE,
    RATE_LIMIT_MAX_RETRIES,
    DATA_RAW_DIR,
    DATA_IMAGES_DIR,
    COOKIES_PATH,
)

load_dotenv()

TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_EMAIL = os.getenv("TWITTER_EMAIL")
TWITTER_PASSWORD = os.getenv("TWITTER_PASSWORD")


def serialize_media(media_list) -> list[dict]:
    """Convert twikit media objects to plain dicts."""
    if not media_list:
        return []
    result = []
    for m in media_list:
        entry = {
            "type": getattr(m, "type", None),
            "url": getattr(m, "media_url", None) or getattr(m, "url", None),
            "media_key": getattr(m, "media_key", None) or getattr(m, "id", None),
        }
        # For videos, try to get the highest quality stream URL
        if entry["type"] in ("video", "animated_gif"):
            streams = getattr(m, "streams", None)
            if streams:
                # streams is typically a list of dicts with 'url' and 'bitrate'
                best = max(streams, key=lambda s: s.get("bitrate", 0) if isinstance(s, dict) else 0)
                entry["video_url"] = best.get("url") if isinstance(best, dict) else getattr(best, "url", None)
        result.append(entry)
    return result


def serialize_tweet(tweet, user_label: str) -> dict:
    """Convert a twikit Tweet object to a plain dict for JSON storage."""
    return {
        "id": tweet.id,
        "created_at": tweet.created_at,
        "text": tweet.text or getattr(tweet, "full_text", None),
        "lang": getattr(tweet, "lang", None),
        "user_id": tweet.user.id if tweet.user else None,
        "user_name": tweet.user.screen_name if tweet.user else None,
        "user_display_name": tweet.user.name if tweet.user else None,
        "user_label": user_label,
        "in_reply_to": getattr(tweet, "in_reply_to", None),
        "is_quote_status": getattr(tweet, "is_quote_status", False),
        "quoted_tweet_id": tweet.quote.id if getattr(tweet, "quote", None) else None,
        "retweeted_tweet_id": tweet.retweeted_tweet.id if getattr(tweet, "retweeted_tweet", None) else None,
        "reply_count": getattr(tweet, "reply_count", None),
        "favorite_count": getattr(tweet, "favorite_count", None),
        "retweet_count": getattr(tweet, "retweet_count", None),
        "quote_count": getattr(tweet, "quote_count", None),
        "bookmark_count": getattr(tweet, "bookmark_count", None),
        "view_count": getattr(tweet, "view_count", None),
        "media": serialize_media(getattr(tweet, "media", None)),
        "has_card": getattr(tweet, "has_card", False),
        "card_title": getattr(tweet, "thumbnail_title", None),
        "card_url": getattr(tweet, "thumbnail_url", None),
        "hashtags": getattr(tweet, "hashtags", None),
        "urls": getattr(tweet, "urls", None),
        "possibly_sensitive": getattr(tweet, "possibly_sensitive", False),
    }


def serialize_reply(reply, parent_tweet_id: str, parent_user_label: str) -> dict:
    """Convert a reply Tweet to dict, adding parent reference."""
    return {
        "id": reply.id,
        "parent_tweet_id": parent_tweet_id,
        "parent_user_label": parent_user_label,
        "created_at": reply.created_at,
        "text": reply.text or getattr(reply, "full_text", None),
        "lang": getattr(reply, "lang", None),
        "user_id": reply.user.id if reply.user else None,
        "user_name": reply.user.screen_name if reply.user else None,
        "user_display_name": reply.user.name if reply.user else None,
        "user_followers_count": reply.user.followers_count if reply.user else None,
        "user_is_verified": getattr(reply.user, "verified", None) if reply.user else None,
        "user_is_blue_verified": getattr(reply.user, "is_blue_verified", None) if reply.user else None,
        "reply_count": getattr(reply, "reply_count", None),
        "favorite_count": getattr(reply, "favorite_count", None),
        "retweet_count": getattr(reply, "retweet_count", None),
        "quote_count": getattr(reply, "quote_count", None),
        "bookmark_count": getattr(reply, "bookmark_count", None),
        "view_count": getattr(reply, "view_count", None),
        "hashtags": getattr(reply, "hashtags", None),
        "urls": getattr(reply, "urls", None),
        "media": serialize_media(getattr(reply, "media", None)),
    }


async def download_images(tweet_dict: dict, user_label: str, http_client: httpx.AsyncClient):
    """Download photo media from a tweet to data/images/{user}/{tweet_id}_{index}.jpg"""
    image_paths = []
    media_list = tweet_dict.get("media", [])
    if not media_list:
        return image_paths

    user_img_dir = DATA_IMAGES_DIR / user_label
    user_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, media in enumerate(media_list):
        if media.get("type") != "photo":
            continue
        url = media.get("url")
        if not url:
            continue

        filename = f"{tweet_dict['id']}_{idx}.jpg"
        filepath = user_img_dir / filename

        if filepath.exists():
            image_paths.append(str(filepath.relative_to(DATA_IMAGES_DIR.parent.parent)))
            continue

        try:
            response = await http_client.get(url, timeout=30.0)
            response.raise_for_status()
            filepath.write_bytes(response.content)
            image_paths.append(str(filepath.relative_to(DATA_IMAGES_DIR.parent.parent)))
            await asyncio.sleep(RATE_LIMIT_PAUSE_IMAGES)
        except Exception as e:
            print(f"  Warning: Failed to download image {url}: {e}")

    return image_paths


async def fetch_replies_for_tweet(client, tweet_id: str, user_label: str) -> list[dict]:
    """Fetch replies for a tweet using search (conversation_id query).

    Uses Twitter search with conversation_id filter instead of
    get_tweet_by_id().replies, which is broken due to API changes.
    """
    from twikit import TooManyRequests

    all_replies = []
    retries = 0
    query = f"conversation_id:{tweet_id}"

    try:
        search_result = await client.search_tweet(query, "Latest", count=20)

        for tweet in search_result:
            # Skip the original tweet itself
            if tweet.id == tweet_id:
                continue
            all_replies.append(serialize_reply(tweet, tweet_id, user_label))

        # Paginate
        page = 1
        while True:
            if MAX_REPLIES_PER_TWEET and len(all_replies) >= MAX_REPLIES_PER_TWEET:
                break
            try:
                await asyncio.sleep(RATE_LIMIT_PAUSE_REPLIES)
                more = await search_result.next()
                if not more:
                    break
                for tweet in more:
                    if tweet.id == tweet_id:
                        continue
                    all_replies.append(serialize_reply(tweet, tweet_id, user_label))
                search_result = more
                page += 1
            except TooManyRequests as e:
                retry_after = getattr(e, "retry_after", RATE_LIMIT_BACKOFF_BASE)
                wait_time = max(retry_after, RATE_LIMIT_BACKOFF_BASE)
                print(f"    Rate limited on replies page {page}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                retries += 1
                if retries >= RATE_LIMIT_MAX_RETRIES:
                    print(f"    Max retries reached for tweet {tweet_id}. Moving on.")
                    break
            except StopIteration:
                break
            except Exception:
                break

    except TooManyRequests as e:
        retry_after = getattr(e, "retry_after", RATE_LIMIT_BACKOFF_BASE)
        wait_time = max(retry_after, RATE_LIMIT_BACKOFF_BASE)
        print(f"    Rate limited on search. Waiting {wait_time}s...")
        await asyncio.sleep(wait_time)
    except Exception as e:
        print(f"    Error fetching replies for tweet {tweet_id}: {e}")

    return all_replies


def save_partial_data(tweets: list[dict], replies: list[dict], user_label: str):
    """Save data collected so far (for graceful interruption)."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    tweets_path = DATA_RAW_DIR / f"{user_label}_tweets.json"
    replies_path = DATA_RAW_DIR / f"{user_label}_replies.json"

    tweets_path.write_text(
        json.dumps(tweets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replies_path.write_text(
        json.dumps(replies, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Saved {len(tweets)} tweets and {len(replies)} replies for {user_label}")


async def collect_user_data(client, user_label: str, user_info: dict) -> dict:
    """Collect tweets, replies, and images for a single user."""
    from twikit import TooManyRequests

    screen_name = user_info["screen_name"]
    print(f"\n{'='*60}")
    print(f"Collecting data for {screen_name} ({user_label})")
    print(f"{'='*60}")

    # Resolve user
    user = await client.get_user_by_screen_name(screen_name)
    user_id = user.id
    print(f"  User ID: {user_id}, Followers: {user.followers_count:,}, Tweets: {user.statuses_count:,}")

    # Collect tweets
    all_tweets = []
    all_replies = []
    start_time = time.time()

    print(f"\n  Fetching {TWEETS_PER_USER} tweets...")
    tweets_result = await client.get_user_tweets(user_id, "Tweets", count=40)

    for tweet in tweets_result:
        all_tweets.append(serialize_tweet(tweet, user_label))

    tweet_pbar = tqdm(total=TWEETS_PER_USER, desc=f"  Tweets ({user_label})", initial=len(all_tweets))

    retries = 0
    while len(all_tweets) < TWEETS_PER_USER:
        try:
            await asyncio.sleep(RATE_LIMIT_PAUSE_TWEETS)
            more_tweets = await tweets_result.next()
            if not more_tweets:
                print(f"\n  No more tweets available. Collected {len(all_tweets)} total.")
                break
            for tweet in more_tweets:
                if len(all_tweets) >= TWEETS_PER_USER:
                    break
                all_tweets.append(serialize_tweet(tweet, user_label))
            tweet_pbar.update(len(all_tweets) - tweet_pbar.n)
            tweets_result = more_tweets
            retries = 0
        except TooManyRequests as e:
            retry_after = getattr(e, "retry_after", RATE_LIMIT_BACKOFF_BASE)
            wait_time = max(retry_after, RATE_LIMIT_BACKOFF_BASE)
            print(f"\n  Rate limited. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            retries += 1
            if retries >= RATE_LIMIT_MAX_RETRIES:
                print(f"  Max retries reached. Collected {len(all_tweets)} tweets.")
                break
        except StopIteration:
            break

    tweet_pbar.close()
    # Trim to exact count
    all_tweets = all_tweets[:TWEETS_PER_USER]

    print(f"\n  Collected {len(all_tweets)} tweets. Now fetching replies and images...")

    # Download images and fetch replies for each tweet
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        for i, tweet_dict in enumerate(tqdm(all_tweets, desc=f"  Replies+Images ({user_label})")):
            tweet_id = tweet_dict["id"]

            # Download images
            image_paths = await download_images(tweet_dict, user_label, http_client)
            tweet_dict["local_image_paths"] = image_paths

            # Fetch replies
            replies = await fetch_replies_for_tweet(client, tweet_id, user_label)
            all_replies.extend(replies)

            # Periodic partial save every 10 tweets
            if (i + 1) % 10 == 0:
                save_partial_data(all_tweets, all_replies, user_label)

    elapsed = time.time() - start_time

    # Final save
    save_partial_data(all_tweets, all_replies, user_label)

    stats = {
        "user_id": user_id,
        "screen_name": screen_name,
        "followers_count": user.followers_count,
        "tweets_requested": TWEETS_PER_USER,
        "tweets_collected": len(all_tweets),
        "replies_collected": len(all_replies),
        "images_downloaded": sum(len(t.get("local_image_paths", [])) for t in all_tweets),
        "collection_duration_seconds": round(elapsed, 1),
    }

    print(f"\n  Summary for {screen_name}:")
    print(f"    Tweets: {stats['tweets_collected']}")
    print(f"    Replies: {stats['replies_collected']}")
    print(f"    Images: {stats['images_downloaded']}")
    print(f"    Duration: {stats['collection_duration_seconds']}s")

    return stats


async def main():
    from twikit import Client

    # Initialize client
    client = Client("en-US")

    # Load cookies (required - login() no longer works due to Twitter API changes)
    if COOKIES_PATH.exists():
        print("Loading saved cookies...")
        client.load_cookies(str(COOKIES_PATH))
        print("Cookies loaded successfully.")
    else:
        print("Error: cookies.json not found.")
        print()
        print("Twitter/X has deprecated the login API endpoint that twikit uses.")
        print("You need to export cookies from your browser instead.")
        print()
        print("Run this first:  python export_cookies.py")
        print()
        print("This will extract your Twitter session cookies from your browser")
        print("(you must be logged into x.com in Chrome, Firefox, or Edge).")
        sys.exit(1)

    # Create output directories
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect data for each user
    metadata = {
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "twikit",
        "tweets_per_user": TWEETS_PER_USER,
        "max_replies_per_tweet": MAX_REPLIES_PER_TWEET,
        "targets": {},
    }

    try:
        for user_label, user_info in TARGET_USERS.items():
            stats = await collect_user_data(client, user_label, user_info)
            metadata["targets"][user_label] = stats
            # Pause between users
            await asyncio.sleep(RATE_LIMIT_PAUSE_TWEETS * 2)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial data has been saved.")
    finally:
        # Save metadata
        metadata_path = DATA_RAW_DIR / "collection_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nCollection metadata saved to {metadata_path}")

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    for user_label, stats in metadata.get("targets", {}).items():
        print(f"  {stats['screen_name']}: {stats['tweets_collected']} tweets, "
              f"{stats['replies_collected']} replies, {stats['images_downloaded']} images")
    print(f"\nRaw data saved to: {DATA_RAW_DIR}")
    print(f"Images saved to: {DATA_IMAGES_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
