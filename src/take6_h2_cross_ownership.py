import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "config" / "games_master.csv"
REVIEWS_PATH = ROOT / "data" / "processed" / "reviews.parquet"
OWNERSHIP_PATH = ROOT / "data" / "processed" / "user_owned_selected_games.parquet"

OUTPUT_DIR = ROOT / "outputs" / "take6"
REVIEWER_OUTPUT = OUTPUT_DIR / "h2_cross_ownership_reviewer_level.parquet"
SUMMARY_OUTPUT = OUTPUT_DIR / "h2_cross_ownership_summary.csv"

INDIE_GROUPS = {
    "core",
    "positioning",
    "contrast",
}

AAA_GROUPS = {
    "benchmark",
}


def parse_owned_appids(value):
    if isinstance(
        value,
        list,
    ):
        return {
            int(x)
            for x in value
        }

    if (
        value is None
        or (
            isinstance(
                value,
                float,
            )
            and pd.isna(
                value
            )
        )
    ):
        return set()

    try:
        return {
            int(x)
            for x in json.loads(
                value
            )
        }

    except Exception:
        return set()


def pct(numerator, denominator):
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def main():

    games = pd.read_csv(
        MASTER_PATH
    )

    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    ownership = pd.read_parquet(
        OWNERSHIP_PATH
    )

    indie_appids = set(
        games.loc[
            games[
                "group"
            ].isin(
                INDIE_GROUPS
            ),
            "appid",
        ]
        .astype(int)
    )

    aaa_appids = set(
        games.loc[
            games[
                "group"
            ].isin(
                AAA_GROUPS
            ),
            "appid",
        ]
        .astype(int)
    )

    print(
        "Indie AppIDs:",
        len(indie_appids),
    )

    print(
        "AAA AppIDs:",
        len(aaa_appids),
    )

    # --------------------------------------------------
    # One reviewer -> reviewed groups / reviewed appids
    # --------------------------------------------------

    reviewer_groups = (
        reviews
        .groupby(
            "reviewer_hash"
        )["group"]
        .agg(
            lambda x:
                sorted(
                    set(x)
                )
        )
        .rename(
            "reviewed_groups"
        )
    )

    reviewer_reviewed_apps = (
        reviews
        .groupby(
            "reviewer_hash"
        )["appid"]
        .agg(
            lambda x:
                sorted({
                    int(v)
                    for v in x
                })
        )
        .rename(
            "reviewed_appids"
        )
    )

    reviewers = (
        pd.concat(
            [
                reviewer_groups,
                reviewer_reviewed_apps,
            ],
            axis=1,
        )
        .reset_index()
    )

    df = reviewers.merge(
        ownership,
        on="reviewer_hash",
        how="left",
    )

    # --------------------------------------------------
    # Parse ownership
    # --------------------------------------------------

    df[
        "owned_appids"
    ] = df[
        "owned_appids_json"
    ].apply(
        parse_owned_appids
    )

    df[
        "is_indie_reviewer"
    ] = df[
        "reviewed_groups"
    ].apply(
        lambda groups:
            any(
                g in INDIE_GROUPS
                for g in groups
            )
    )

    df[
        "is_aaa_reviewer"
    ] = df[
        "reviewed_groups"
    ].apply(
        lambda groups:
            any(
                g in AAA_GROUPS
                for g in groups
            )
    )

    df[
        "owns_any_indie"
    ] = df[
        "owned_appids"
    ].apply(
        lambda apps:
            len(
                apps
                & indie_appids
            ) > 0
    )

    df[
        "owns_any_aaa"
    ] = df[
        "owned_appids"
    ].apply(
        lambda apps:
            len(
                apps
                & aaa_appids
            ) > 0
    )

    df[
        "owns_both_aaa_indie"
    ] = (
        df[
            "owns_any_indie"
        ]
        &
        df[
            "owns_any_aaa"
        ]
    )

    # Diagnostic:
    # API 결과에 적어도 하나의 '리뷰한 선정 게임'이 보이는가?
    df[
        "reviewed_game_visible"
    ] = df.apply(
        lambda row:
            bool(
                set(
                    row[
                        "reviewed_appids"
                    ]
                )
                & row[
                    "owned_appids"
                ]
            )
            if (
                row[
                    "api_observable"
                ] is True
                or row[
                    "api_observable"
                ] == True
            )
            else False,
        axis=1,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # object(set)은 parquet에 바로 저장하지 않음
    save_df = df.drop(
        columns=[
            "owned_appids"
        ]
    )

    save_df.to_parquet(
        REVIEWER_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # Core H2-a metrics
    # --------------------------------------------------

    observable = df[
        df[
            "api_observable"
        ] == True
    ].copy()

    unique_reviewers = (
        reviews[
            "reviewer_hash"
        ]
        .nunique()
    )

    api_rows = (
        df[
            "api_observable"
        ]
        .notna()
        .sum()
    )

    observable_n = len(
        observable
    )

    print()
    print("=" * 78)
    print("H2-a — AAA / INDIE CROSS-OWNERSHIP")
    print("=" * 78)

    print(
        "Unique reviewers:",
        unique_reviewers,
    )

    print(
        "Reviewers with API collection row:",
        api_rows,
    )

    print(
        "Observable GetOwnedGames responses:",
        observable_n,
        f"({observable_n / unique_reviewers:.1%} of all reviewers)"
        if unique_reviewers
        else "",
    )

    if observable_n == 0:
        raise RuntimeError(
            "No observable ownership responses. "
            "Check API key/privacy/output."
        )

    sanity = (
        observable[
            "reviewed_game_visible"
        ].mean()
    )

    print(
        "Reviewed-selected-game visible "
        "diagnostic:",
        f"{sanity:.1%}",
    )

    # Indie reviewer -> AAA ownership
    indie_reviewers = observable[
        observable[
            "is_indie_reviewer"
        ]
    ]

    indie_with_aaa = (
        indie_reviewers[
            "owns_any_aaa"
        ].sum()
    )

    # AAA reviewer -> Indie ownership
    aaa_reviewers = observable[
        observable[
            "is_aaa_reviewer"
        ]
    ]

    aaa_with_indie = (
        aaa_reviewers[
            "owns_any_indie"
        ].sum()
    )

    # Overall both
    both_count = (
        observable[
            "owns_both_aaa_indie"
        ].sum()
    )

    print()

    print(
        "Indie reviewers with visible AAA ownership:",
        f"{int(indie_with_aaa)} / {len(indie_reviewers)}",
        f"({pct(indie_with_aaa, len(indie_reviewers)):.1%})"
        if len(indie_reviewers)
        else "",
    )

    print(
        "AAA reviewers with visible Indie ownership:",
        f"{int(aaa_with_indie)} / {len(aaa_reviewers)}",
        f"({pct(aaa_with_indie, len(aaa_reviewers)):.1%})"
        if len(aaa_reviewers)
        else "",
    )

    print(
        "Observable reviewers owning both AAA + Indie:",
        f"{int(both_count)} / {observable_n}",
        f"({pct(both_count, observable_n):.1%})",
    )

    # --------------------------------------------------
    # Group-level summary
    # --------------------------------------------------

    rows = []

    for group in sorted(
        set(
            reviews[
                "group"
            ].dropna()
        )
    ):

        group_reviewers = observable[
            observable[
                "reviewed_groups"
            ].apply(
                lambda groups:
                    group in groups
            )
        ]

        if group in INDIE_GROUPS:
            target_col = (
                "owns_any_aaa"
            )
            target_name = (
                "visible_aaa_ownership"
            )

        elif group in AAA_GROUPS:
            target_col = (
                "owns_any_indie"
            )
            target_name = (
                "visible_indie_ownership"
            )

        else:
            continue

        numerator = int(
            group_reviewers[
                target_col
            ].sum()
        )

        denominator = len(
            group_reviewers
        )

        rows.append({
            "reviewer_group":
                group,

            "observable_reviewers":
                denominator,

            "target":
                target_name,

            "cross_owned_reviewers":
                numerator,

            "cross_ownership_rate":
                pct(
                    numerator,
                    denominator,
                ),

            "reviewed_game_visible_rate":
                (
                    group_reviewers[
                        "reviewed_game_visible"
                    ].mean()
                    if denominator
                    else None
                ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("GROUP-LEVEL")
    print("=" * 78)

    if not summary.empty:
        display = summary.copy()

        display[
            "cross_ownership_rate"
        ] = (
            display[
                "cross_ownership_rate"
            ]
            * 100
        ).round(1)

        display[
            "reviewed_game_visible_rate"
        ] = (
            display[
                "reviewed_game_visible_rate"
            ]
            * 100
        ).round(1)

        print(
            display.to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved: {REVIEWER_OUTPUT}"
    )

    print(
        f"Saved: {SUMMARY_OUTPUT}"
    )

    print()
    print(
        "INTERPRETATION BOUNDARY:"
    )

    print(
        "- This tests visible cross-ownership, "
        "not purchase timing, price paid, or wallet allocation."
    )

    print(
        "- Private game details and individually-private games "
        "can create false negatives."
    )

    print(
        "- Therefore H2-a can support coexistence in observed libraries; "
        "Share of Wallet remains unresolved."
    )


if __name__ == "__main__":
    main()

