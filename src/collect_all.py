import argparse
import time
from pathlib import Path

import pandas as pd

from collect_steam_review import collect_reviews
from collect_steamspy import collect_game as collect_steamspy_game
from build_market_dataset import main as build_market_dataset


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT
    / "config"
    / "games_master.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


def merge_review_files(games):
    """
    게임별 reviews_<appid>.parquet 파일을 합쳐
    reviews.parquet 하나로 만든다.
    """

    frames = []

    for _, game in games.iterrows():

        appid = int(
            game["appid"]
        )

        path = (
            PROCESSED_DIR
            / f"reviews_{appid}.parquet"
        )

        if not path.exists():

            print(
                f"[Merge] missing: {path.name}"
            )

            continue

        df = pd.read_parquet(
            path
        )

        # 분석 편의를 위해 게임 정보 추가
        df["title"] = game["title"]
        df["group"] = game["group"]

        frames.append(
            df
        )

    if not frames:

        print(
            "[Merge] No review files found."
        )

        return

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    # recommendation ID 중복 안전 제거
    merged = (
        merged
        .drop_duplicates(
            subset=[
                "appid",
                "recommendation_id",
            ]
        )
        .reset_index(drop=True)
    )

    output_path = (
        PROCESSED_DIR
        / "reviews.parquet"
    )

    merged.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        f"[Merge] Saved: {output_path}"
    )

    print(
        f"[Merge] Total reviews: "
        f"{len(merged)}"
    )

    print()

    print(
        merged[
            [
                "appid",
                "title",
                "group",
            ]
        ]
        .value_counts()
        .sort_index()
    )


def collect_steam(
    games,
    target_reviews,
):
    """
    선택된 게임들에 대해
    Steam Review를 수집한다.
    """

    failures = []

    print()
    print(
        "================================"
    )
    print(
        "   STEAM REVIEW COLLECTION"
    )
    print(
        "================================"
    )

    for index, (_, game) in enumerate(
        games.iterrows(),
        start=1,
    ):

        appid = int(
            game["appid"]
        )

        title = game["title"]

        print()
        print(
            f"[{index}/{len(games)}] "
            f"{title} ({appid})"
        )

        try:

            df = collect_reviews(
                appid=appid,
                target_reviews=target_reviews,
            )

            print(
                f"[OK] {title}: "
                f"{len(df)} reviews"
            )

        except Exception as e:

            failures.append(
                (
                    appid,
                    title,
                    str(e),
                )
            )

            print(
                f"[ERROR] {title}: {e}"
            )

        time.sleep(
            0.5
        )

    print()
    print(
        "=== Steam collection finished ==="
    )

    if failures:

        print()
        print(
            "Failures:"
        )

        for (
            appid,
            title,
            error,
        ) in failures:

            print(
                f" - {title} "
                f"({appid}): "
                f"{error}"
            )

    else:

        print(
            "No Steam failures."
        )


def collect_steamspy(games):
    """
    선택된 게임들에 대해
    SteamSpy appdetails를 수집한다.
    """

    failures = []

    print()
    print(
        "================================"
    )
    print(
        "   STEAMSPY COLLECTION"
    )
    print(
        "================================"
    )

    for index, (_, game) in enumerate(
        games.iterrows(),
        start=1,
    ):

        appid = int(
            game["appid"]
        )

        title = game["title"]

        print()
        print(
            f"[{index}/{len(games)}] "
            f"{title} ({appid})"
        )

        try:

            collect_steamspy_game(
                appid=appid
            )

        except Exception as e:

            failures.append(
                (
                    appid,
                    title,
                    str(e),
                )
            )

            print(
                f"[ERROR] {title}: {e}"
            )

        # SteamSpy는 조금 천천히 호출
        time.sleep(
            1.2
        )

    print()
    print(
        "=== SteamSpy collection finished ==="
    )

    if failures:

        print()
        print(
            "Failures:"
        )

        for (
            appid,
            title,
            error,
        ) in failures:

            print(
                f" - {title} "
                f"({appid}): "
                f"{error}"
            )

    else:

        print(
            "No SteamSpy failures."
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=[
            "steam",
            "steamspy",
            "all",
        ],
        required=True,
    )

    parser.add_argument(
        "--reviews",
        type=int,
        default=750,
        help=(
            "Target number of Steam reviews "
            "per game."
        ),
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help=(
            "Collect only one group from "
            "games_master.csv, e.g. contrast. "
            "If omitted, collect all games."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 전체 master는 항상 보존
    # --------------------------------------------------

    all_games = pd.read_csv(
        MASTER_PATH
    )

    if all_games[
        "appid"
    ].duplicated().any():

        raise ValueError(
            "Duplicate AppID found "
            "in games_master.csv"
        )

    print(
        f"Games in master: "
        f"{len(all_games)}"
    )

    # --------------------------------------------------
    # 실제 수집 대상만 filtering
    # --------------------------------------------------

    selected_games = (
        all_games.copy()
    )

    if args.group is not None:

        available_groups = set(
            all_games[
                "group"
            ].unique()
        )

        if (
            args.group
            not in available_groups
        ):

            raise ValueError(
                f"Unknown group: "
                f"{args.group}. "
                f"Available groups: "
                f"{sorted(available_groups)}"
            )

        selected_games = (
            all_games[
                all_games["group"]
                == args.group
            ]
            .copy()
            .reset_index(drop=True)
        )

    print(
        f"Games selected: "
        f"{len(selected_games)}"
    )

    if args.group is not None:

        print(
            f"Group filter: "
            f"{args.group}"
        )

    print()

    print(
        selected_games[
            [
                "appid",
                "title",
                "group",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # SteamSpy
    # --------------------------------------------------

    if args.source in {
        "steamspy",
        "all",
    }:

        collect_steamspy(
            selected_games
        )

        print()
        print(
            "Rebuilding games.parquet..."
        )

        # build_market_dataset은
        # 전체 games_master.csv 기준으로 rebuild
        build_market_dataset()

    # --------------------------------------------------
    # Steam Reviews
    # --------------------------------------------------

    if args.source in {
        "steam",
        "all",
    }:

        collect_steam(
            selected_games,
            target_reviews=args.reviews,
        )

        print()
        print(
            "Rebuilding merged reviews.parquet "
            "with ALL master games..."
        )

        # 중요:
        # 수집은 selected_games만 했지만
        # 최종 dataset은 기존 + 신규 전체를 합친다.
        merge_review_files(
            all_games
        )


if __name__ == "__main__":
    main()
