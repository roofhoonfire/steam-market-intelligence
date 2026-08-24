from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TOPICS_PATH = (
    ROOT
    / "data"
    / "features"
    / "review_topics_initial.parquet"
)

LABELS_PATH = (
    ROOT
    / "outputs"
    / "take8"
    / "cluster_labels.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "take9"
)


NEED_MAP = {
    0: "Emotional Narrative & Payoff",
    12: "Emotional Narrative & Payoff",
    13: "Emotional Narrative & Payoff",
    29: "Emotional Narrative & Payoff",
    32: "Emotional Narrative & Payoff",
    34: "Emotional Narrative & Payoff",
    35: "Emotional Narrative & Payoff",
    37: "Emotional Narrative & Payoff",
    41: "Emotional Narrative & Payoff",

    10: "Distinctive Presentation & Atmosphere",
    39: "Distinctive Presentation & Atmosphere",

    16: "Narrative Pacing & Experience Density",
    22: "Narrative Pacing & Experience Density",
    33: "Narrative Pacing & Experience Density",
    42: "Narrative Pacing & Experience Density",

    17: "Gameplay Depth & Agency",
    38: "Gameplay Depth & Agency",
    43: "Gameplay Depth & Agency",

    23: "Replayability & Returnability",

    9: "Content Longevity & Expansion",

    28: "Technical Accessibility",
}


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    topics = pd.read_parquet(
        TOPICS_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    labels["cluster_id"] = (
        labels["cluster_id"]
        .astype(int)
    )

    labels["higher_order_need"] = (
        labels["cluster_id"]
        .map(NEED_MAP)
    )

    df = topics.merge(
        labels[
            [
                "cluster_id",
                "actionable",
                "higher_order_need",
            ]
        ],
        on="cluster_id",
        how="left",
    )

    # All topic-eligible reviews remain available
    # as the within-game reference pool.
    results = []

    for need in sorted(
        set(NEED_MAP.values())
    ):

        need_clusters = {
            cluster_id
            for cluster_id, mapped_need
            in NEED_MAP.items()
            if mapped_need == need
        }

        need_df = df[
            df["cluster_id"].isin(
                need_clusters
            )
        ].copy()

        games = sorted(
            need_df["title"]
            .unique()
        )

        for title in games:

            exposed = need_df[
                need_df["title"] == title
            ]

            # Same game, but excluding reviews
            # assigned to this higher-order Need.
            reference = df[
                (df["title"] == title)
                &
                (~df["cluster_id"].isin(
                    need_clusters
                ))
            ]

            if (
                len(exposed) < 3
                or len(reference) < 10
            ):
                continue

            need_rate = (
                exposed["voted_up"].mean()
            )

            ref_rate = (
                reference["voted_up"].mean()
            )

            results.append({
                "higher_order_need":
                    need,

                "title":
                    title,

                "group":
                    exposed["group"].iloc[0],

                "need_reviews":
                    len(exposed),

                "reference_reviews":
                    len(reference),

                "need_recommend_rate":
                    need_rate,

                "reference_recommend_rate":
                    ref_rate,

                "delta_pp":
                    (
                        need_rate
                        - ref_rate
                    ) * 100,

                "need_median_playtime_h":
                    exposed[
                        "playtime_at_review_hours"
                    ].median(),

                "reference_median_playtime_h":
                    reference[
                        "playtime_at_review_hours"
                    ].median(),
            })

    game_results = pd.DataFrame(
        results
    )

    summary = (
        game_results
        .groupby(
            "higher_order_need",
            as_index=False,
        )
        .agg(
            games_compared=(
                "title",
                "nunique",
            ),
            median_within_game_delta_pp=(
                "delta_pp",
                "median",
            ),
            mean_within_game_delta_pp=(
                "delta_pp",
                "mean",
            ),
            games_positive_delta=(
                "delta_pp",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            games_negative_delta=(
                "delta_pp",
                lambda x: int(
                    (x < 0).sum()
                ),
            ),
            median_need_playtime_h=(
                "need_median_playtime_h",
                "median",
            ),
            median_reference_playtime_h=(
                "reference_median_playtime_h",
                "median",
            ),
        )
    )

    summary[
        "playtime_ratio"
    ] = (
        summary[
            "median_need_playtime_h"
        ]
        /
        summary[
            "median_reference_playtime_h"
        ]
    )

    numeric_cols = [
        "median_within_game_delta_pp",
        "mean_within_game_delta_pp",
        "median_need_playtime_h",
        "median_reference_playtime_h",
        "playtime_ratio",
    ]

    summary[
        numeric_cols
    ] = summary[
        numeric_cols
    ].round(2)

    game_path = (
        OUTPUT_DIR
        / "within_game_need_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "within_game_need_summary.csv"
    )

    game_results.to_csv(
        game_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 80)
    print("TAKE 9-B — WITHIN-GAME VALIDATION")
    print("=" * 80)

    print()
    print(
        summary.sort_values(
            "median_within_game_delta_pp",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print(f"Saved: {game_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
