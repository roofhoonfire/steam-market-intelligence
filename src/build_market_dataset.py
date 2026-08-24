import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT
    / "config"
    / "games_master.csv"
)

STEAMSPY_RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "steamspy"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


def parse_owner_range(value):
    """
    SteamSpy owner range:
    '20,000 .. 50,000'
    ->
    low=20000, high=50000
    """

    if not value:
        return None, None

    try:
        low, high = value.split("..")

        low = int(
            low.strip().replace(",", "")
        )

        high = int(
            high.strip().replace(",", "")
        )

        return low, high

    except (ValueError, AttributeError):
        return None, None


def parse_price_usd(value):
    """
    SteamSpy price:
    1499 또는 "1499"
    ->
    14.99 USD

    None, 빈 문자열, 변환 불가능한 값은 None 처리.
    """

    if value in (None, ""):
        return None

    try:
        return float(value) / 100.0

    except (TypeError, ValueError):
        return None


def parse_int(value):
    """
    SteamSpy가 숫자를 문자열 형태로 반환하는 경우까지 고려하여
    안전하게 int로 변환한다.
    """

    if value in (None, ""):
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def load_steamspy(appid):
    """
    해당 AppID의 SteamSpy Raw JSON을 읽는다.

    파일이 아직 없다면 빈 dict를 반환한다.
    """

    path = (
        STEAMSPY_RAW_DIR
        / f"{appid}.json"
    )

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main():

    games = pd.read_csv(
        MASTER_PATH
    )

    rows = []

    for _, game in games.iterrows():

        appid = int(
            game["appid"]
        )

        steamspy = load_steamspy(
            appid
        )

        # -----------------------------------
        # Owner range
        # -----------------------------------

        owners_low, owners_high = (
            parse_owner_range(
                steamspy.get("owners")
            )
        )

        # -----------------------------------
        # Price
        # -----------------------------------

        price_usd = parse_price_usd(
            steamspy.get("price")
        )

        initial_price_usd = parse_price_usd(
            steamspy.get("initialprice")
        )

        # -----------------------------------
        # Reviews
        # -----------------------------------

        positive = parse_int(
            steamspy.get("positive")
        )

        negative = parse_int(
            steamspy.get("negative")
        )

        total_reviews = None
        positive_rate = None

        if (
            positive is not None
            and negative is not None
        ):
            total_reviews = (
                positive + negative
            )

            if total_reviews > 0:
                positive_rate = (
                    positive
                    / total_reviews
                )

        # -----------------------------------
        # Other numeric fields
        # -----------------------------------

        discount_pct = parse_int(
            steamspy.get("discount")
        )

        ccu = parse_int(
            steamspy.get("ccu")
        )

        average_forever_raw = parse_int(
            steamspy.get(
                "average_forever"
            )
        )

        median_forever_raw = parse_int(
            steamspy.get(
                "median_forever"
            )
        )

        # -----------------------------------
        # Tags
        # -----------------------------------

        tags = steamspy.get(
            "tags",
            {}
        )

        if not isinstance(
            tags,
            dict,
        ):
            tags = {}

        # -----------------------------------
        # Processed row
        # -----------------------------------

        rows.append(
            {
                "appid":
                    appid,

                "title":
                    game["title"],

                "group":
                    game["group"],

                "selection_reason":
                    game[
                        "selection_reason"
                    ],

                # Owner estimate
                "steamspy_owners_raw":
                    steamspy.get(
                        "owners"
                    ),

                "steamspy_owners_low":
                    owners_low,

                "steamspy_owners_high":
                    owners_high,

                # Price
                "steamspy_price_usd":
                    price_usd,

                "steamspy_initial_price_usd":
                    initial_price_usd,

                "steamspy_discount_pct":
                    discount_pct,

                # Review aggregate
                "steamspy_positive":
                    positive,

                "steamspy_negative":
                    negative,

                "steamspy_total_reviews":
                    total_reviews,

                "steamspy_positive_rate":
                    positive_rate,

                # Concurrent users
                "steamspy_ccu":
                    ccu,

                # Playtime은 SteamSpy 품질 문제 때문에
                # raw 참고값으로만 보존한다.
                "steamspy_average_forever_raw":
                    average_forever_raw,

                "steamspy_median_forever_raw":
                    median_forever_raw,

                # Tags
                "steamspy_tags":
                    list(
                        tags.keys()
                    ),

                "steamspy_top_tag":
                    (
                        next(
                            iter(tags),
                            None,
                        )
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        PROCESSED_DIR
        / "games.parquet"
    )

    df.to_parquet(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    print()

    print(
        df[
            [
                "appid",
                "title",
                "group",
                "steamspy_owners_raw",
                "steamspy_price_usd",
                "steamspy_positive_rate",
                "steamspy_ccu",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
