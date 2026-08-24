import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------
# Project paths
# ---------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "steam_reviews"
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------
# Steam Review API
# ---------------------------------------

STEAM_REVIEW_URL = "https://store.steampowered.com/appreviews/{appid}"


def hash_steamid(steamid):
    """
    Processed dataset에서는 Steam ID 원문을 사용하지 않고
    SHA-256 hash 값으로 변환한다.
    """

    if not steamid:
        return None

    return hashlib.sha256(
        steamid.encode("utf-8")
    ).hexdigest()


def unix_to_datetime(timestamp):
    """
    Unix timestamp를 UTC datetime으로 변환한다.
    """

    if not timestamp:
        return None

    return pd.to_datetime(
        timestamp,
        unit="s",
        utc=True,
    )


def flatten_review(appid, review):
    """
    Steam Review JSON 하나를
    pandas에서 사용하기 쉬운 1개 row로 변환한다.
    """

    author = review.get("author", {})

    playtime_forever = author.get("playtime_forever")
    playtime_at_review = author.get("playtime_at_review")

    review_text = review.get("review", "")

    return {
        "recommendation_id": str(
            review.get("recommendationid")
        ),

        "appid": appid,

        "reviewer_hash": hash_steamid(
            author.get("steamid")
        ),

        "language": review.get("language"),

        "num_games_owned": author.get(
            "num_games_owned"
        ),

        "num_reviews": author.get(
            "num_reviews"
        ),

        "playtime_forever_min": playtime_forever,

        "playtime_at_review_min": playtime_at_review,

        "playtime_forever_hours": (
            playtime_forever / 60
            if playtime_forever is not None
            else None
        ),

        "playtime_at_review_hours": (
            playtime_at_review / 60
            if playtime_at_review is not None
            else None
        ),

        "voted_up": review.get(
            "voted_up"
        ),

        "votes_up": review.get(
            "votes_up"
        ),

          "weighted_vote_score": (
            float(review.get("weighted_vote_score"))
            if review.get("weighted_vote_score") not in (None, "")
            else None
    ),

        "steam_purchase": review.get(
            "steam_purchase"
        ),

        "received_for_free": review.get(
            "received_for_free"
        ),

        "early_access_review": review.get(
            "written_during_early_access"
        ),

        "created_at": unix_to_datetime(
            review.get("timestamp_created")
        ),

        "updated_at": unix_to_datetime(
            review.get("timestamp_updated")
        ),

        "review_text": review_text,

        "review_length_chars": len(
            review_text
        ),

        "review_length_words": len(
            review_text.split()
        ),

        "collected_at": pd.Timestamp.now(
            tz="UTC"
        ),
    }


def collect_reviews(appid, target_reviews=100):

    cursor = "*"

    collected_reviews = []

    page = 1

    while len(collected_reviews) < target_reviews:

        params = {
            "json": 1,
            "filter": "recent",
            "language": "english",
            "cursor": cursor,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 100,
        }

        response = requests.get(
            STEAM_REVIEW_URL.format(
                appid=appid
            ),
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("success") != 1:
            raise RuntimeError(
                f"Steam API failed: appid={appid}"
            )

        # -----------------------------
        # Raw JSON 저장
        # -----------------------------

        game_raw_dir = RAW_DIR / str(appid)

        game_raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path = (
            game_raw_dir
            / f"page_{page:03d}.json"
        )

        with raw_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # -----------------------------
        # Review 추출
        # -----------------------------

        reviews = payload.get(
            "reviews",
            []
        )

        if not reviews:
            break

        for review in reviews:

            row = flatten_review(
                appid,
                review,
            )

            collected_reviews.append(
                row
            )

        print(
            f"[Steam] "
            f"page={page} "
            f"received={len(reviews)} "
            f"total={len(collected_reviews)}"
        )
        time.sleep(0.25)
        # -----------------------------
        # 다음 page cursor
        # -----------------------------

        next_cursor = payload.get(
            "cursor"
        )

        if not next_cursor:
            break

        if next_cursor == cursor:
            break

        cursor = next_cursor

        page += 1

    # 원하는 개수까지만 사용
    collected_reviews = (
        collected_reviews[
            :target_reviews
        ]
    )

    df = pd.DataFrame(
        collected_reviews
    )

    if not df.empty:

        df = (
            df
            .drop_duplicates(
                subset=[
                    "recommendation_id"
                ]
            )
            .reset_index(drop=True)
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DIR
        / f"reviews_{appid}.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"Total reviews: {len(df)}"
    )

    if len(df) > 0:

        positive_rate = (
            df["voted_up"]
            .mean()
        )

        print(
            f"Positive rate: "
            f"{positive_rate:.3f}"
        )

    return df


if __name__ == "__main__":

    # Meg's Monster
    APPID = 1783360

    collect_reviews(
        appid=APPID,
        target_reviews=100,
    )
