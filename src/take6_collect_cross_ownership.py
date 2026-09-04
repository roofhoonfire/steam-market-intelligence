import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "config" / "games_master.csv"
REVIEWS_PATH = ROOT / "data" / "processed" / "reviews.parquet"
RAW_REVIEW_DIR = ROOT / "data" / "raw" / "steam_reviews"
OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "user_owned_selected_games.parquet"
)

STEAM_OWNED_GAMES_URL = (
    "https://api.steampowered.com/"
    "IPlayerService/GetOwnedGames/v1/"
)


def hash_steamid(steamid):
    if not steamid:
        return None

    return hashlib.sha256(
        str(steamid).encode("utf-8")
    ).hexdigest()


def recover_raw_steamids(needed_hashes):
    """
    기존 raw Steam Review JSON에서
    reviewer_hash <-> raw steamid 매핑을 복구한다.

    raw steamid는 API 호출용으로만 메모리에서 사용하고
    processed output에는 저장하지 않는다.
    """

    mapping = {}

    files = sorted(
        RAW_REVIEW_DIR.glob("*/*.json")
    )

    if not files:
        raise FileNotFoundError(
            f"No raw Steam review JSON found under: "
            f"{RAW_REVIEW_DIR}"
        )

    print(
        f"[Raw mapping] JSON files: {len(files)}"
    )

    for path in files:

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(f)

        except Exception as exc:
            print(
                f"[WARN] Could not read {path}: {exc}"
            )
            continue

        for review in payload.get(
            "reviews",
            [],
        ):
            author = review.get(
                "author",
                {},
            )

            steamid = author.get(
                "steamid"
            )

            reviewer_hash = hash_steamid(
                steamid
            )

            if (
                reviewer_hash
                in needed_hashes
            ):

                old = mapping.get(
                    reviewer_hash
                )

                if (
                    old is not None
                    and old != str(steamid)
                ):
                    raise RuntimeError(
                        "Hash collision / inconsistent "
                        "SteamID mapping detected."
                    )

                mapping[
                    reviewer_hash
                ] = str(steamid)

    return mapping


def fetch_selected_owned_games(
    steamid,
    selected_appids,
    api_key,
    timeout=20,
    retries=4,
):
    """
    Steam GetOwnedGames API를 사용해
    프로젝트에서 선택한 게임들의 ownership만 조회한다.

    response에 game_count가 존재하면 observable,
    response가 비어 있으면 privacy/unavailable로 취급한다.
    """

    input_payload = {
        "steamid":
            str(steamid),

        "include_appinfo":
            False,

        "include_played_free_games":
            False,

        "appids_filter":
            [
                int(x)
                for x in selected_appids
            ],

        "skip_unvetted_apps":
            False,
    }

    headers = {
        "x-webapi-key":
            api_key,
    }

    params = {
        "input_json":
            json.dumps(
                input_payload,
                separators=(",", ":"),
            )
    }

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            response = requests.get(
                STEAM_OWNED_GAMES_URL,
                headers=headers,
                params=params,
                timeout=timeout,
            )

            if response.status_code in {
                401,
                403,
            }:
                raise RuntimeError(
                    "Steam Web API authentication failed "
                    f"(HTTP {response.status_code}). "
                    "Check STEAM_WEB_API_KEY."
                )

            if (
                response.status_code == 429
                or response.status_code >= 500
            ):
                raise requests.HTTPError(
                    "Temporary Steam API error: "
                    f"HTTP {response.status_code}"
                )

            response.raise_for_status()

            payload = response.json()

            body = payload.get(
                "response",
                {},
            )

            # Privacy / unavailable
            if (
                not isinstance(body, dict)
                or "game_count" not in body
            ):
                return {
                    "api_observable":
                        False,

                    "api_status":
                        "private_or_unavailable",

                    "selected_owned_count":
                        None,

                    "owned_appids":
                        [],
                }

            games = body.get(
                "games",
                [],
            )

            owned_appids = sorted({
                int(game["appid"])
                for game in games
                if "appid" in game
            })

            return {
                "api_observable":
                    True,

                "api_status":
                    "ok",

                "selected_owned_count":
                    int(
                        body.get(
                            "game_count",
                            len(
                                owned_appids
                            ),
                        )
                    ),

                "owned_appids":
                    owned_appids,
            }

        except RuntimeError:
            raise

        except Exception as exc:

            last_error = exc

            if attempt < retries:
                time.sleep(
                    0.8
                    * (
                        2
                        ** (
                            attempt - 1
                        )
                    )
                )

    return {
        "api_observable":
            None,

        "api_status":
            (
                "request_error:"
                f"{type(last_error).__name__}"
            ),

        "selected_owned_count":
            None,

        "owned_appids":
            [],
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Smoke test용 reviewer 수 제한. "
            "예: --limit 20"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Concurrent API workers "
            "(default: 8)."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "기존 output이 있어도 다시 조회."
        ),
    )

    args = parser.parse_args()

    api_key = os.getenv(
        "STEAM_WEB_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "STEAM_WEB_API_KEY is not set."
        )

    games = pd.read_csv(
        MASTER_PATH
    )

    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    if (
        "reviewer_hash"
        not in reviews.columns
    ):
        raise ValueError(
            "reviews.parquet has no "
            "reviewer_hash column."
        )

    selected_appids = (
        games[
            "appid"
        ]
        .astype(int)
        .tolist()
    )

    needed_hashes = set(
        reviews[
            "reviewer_hash"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"Selected games: "
        f"{len(selected_appids)}"
    )

    print(
        f"Unique reviewers: "
        f"{len(needed_hashes)}"
    )

    # --------------------------------------------------
    # Existing cache
    # --------------------------------------------------

    existing = None

    if (
        OUTPUT_PATH.exists()
        and not args.force
    ):

        existing = pd.read_parquet(
            OUTPUT_PATH
        )

        done_hashes = set(
            existing[
                "reviewer_hash"
            ]
            .astype(str)
        )

        needed_hashes = (
            needed_hashes
            - done_hashes
        )

        print(
            f"Already cached: "
            f"{len(done_hashes)}"
        )

        print(
            f"Remaining: "
            f"{len(needed_hashes)}"
        )

    # --------------------------------------------------
    # Recover raw SteamIDs
    # --------------------------------------------------

    steamid_map = (
        recover_raw_steamids(
            needed_hashes
        )
    )

    missing = (
        needed_hashes
        - set(
            steamid_map.keys()
        )
    )

    print(
        f"Recovered raw SteamIDs: "
        f"{len(steamid_map)}"
    )

    print(
        f"Missing raw SteamIDs: "
        f"{len(missing)}"
    )

    items = list(
        steamid_map.items()
    )

    if args.limit is not None:
        items = items[
            :args.limit
        ]

    print(
        f"API requests planned: "
        f"{len(items)}"
    )

    if not items:
        print(
            "Nothing to collect."
        )
        return

    # --------------------------------------------------
    # Steam API collection
    # --------------------------------------------------

    rows = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_map = {
            executor.submit(
                fetch_selected_owned_games,
                steamid,
                selected_appids,
                api_key,
            ):
            reviewer_hash

            for (
                reviewer_hash,
                steamid,
            ) in items
        }

        total = len(
            future_map
        )

        for index, future in enumerate(
            as_completed(
                future_map
            ),
            start=1,
        ):

            reviewer_hash = (
                future_map[
                    future
                ]
            )

            try:

                result = (
                    future.result()
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Collection aborted: "
                    f"{exc}"
                ) from exc

            owned_appids = (
                result.pop(
                    "owned_appids"
                )
            )

            row = {
                "reviewer_hash":
                    reviewer_hash,

                **result,

                "owned_appids_json":
                    json.dumps(
                        owned_appids
                    ),
            }

            rows.append(
                row
            )

            if (
                index <= 10
                or index % 100 == 0
                or index == total
            ):

                print(
                    f"[{index}/{total}] "
                    f"observable="
                    f"{row['api_observable']} "
                    f"owned_selected="
                    f"{row['selected_owned_count']}"
                )

    new_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------
    # Append cache
    # --------------------------------------------------

    if (
        existing is not None
        and len(existing) > 0
    ):

        result_df = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )

        result_df = (
            result_df
            .drop_duplicates(
                subset=[
                    "reviewer_hash"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    else:

        result_df = new_df

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Rows: "
        f"{len(result_df)}"
    )

    print()

    print(
        "API status:"
    )

    print(
        result_df[
            "api_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    observable = result_df[
        result_df[
            "api_observable"
        ] == True
    ]

    print()

    print(
        "Observable response share: "
        f"{len(observable) / len(result_df):.1%}"
    )

    print()

    print(
        "IMPORTANT: raw SteamID was used "
        "only in memory and is NOT stored "
        "in the processed output."
    )


if __name__ == "__main__":
    main()
