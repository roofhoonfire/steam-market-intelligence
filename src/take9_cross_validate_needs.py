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


# ============================================================
# Final higher-order taxonomy from Take 8
# ============================================================

NEED_MAP = {
    # 1. Emotional Narrative & Payoff
    0: "Emotional Narrative & Payoff",
    12: "Emotional Narrative & Payoff",
    13: "Emotional Narrative & Payoff",
    29: "Emotional Narrative & Payoff",
    32: "Emotional Narrative & Payoff",
    34: "Emotional Narrative & Payoff",
    35: "Emotional Narrative & Payoff",
    37: "Emotional Narrative & Payoff",
    41: "Emotional Narrative & Payoff",

    # 2. Distinctive Presentation & Atmosphere
    10: "Distinctive Presentation & Atmosphere",
    39: "Distinctive Presentation & Atmosphere",

    # 3. Narrative Pacing & Experience Density
    16: "Narrative Pacing & Experience Density",
    22: "Narrative Pacing & Experience Density",
    33: "Narrative Pacing & Experience Density",
    42: "Narrative Pacing & Experience Density",

    # 4. Gameplay Depth & Agency
    17: "Gameplay Depth & Agency",
    38: "Gameplay Depth & Agency",
    43: "Gameplay Depth & Agency",

    # 5. Replayability & Returnability
    23: "Replayability & Returnability",

    # 6. Content Longevity & Expansion
    9: "Content Longevity & Expansion",

    # 7. Technical Accessibility
    28: "Technical Accessibility",
}


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    topics = pd.read_parquet(
        TOPICS_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    print("=" * 72)
    print("TAKE 9 — CUSTOMER NEED CROSS-VALIDATION")
    print("=" * 72)

    print(
        f"Topic reviews loaded: {len(topics)}"
    )

    print(
        f"LLM labels loaded: {len(labels)}"
    )

    # --------------------------------------------------------
    # Validate taxonomy
    # --------------------------------------------------------

    actionable = labels[
        labels["actionable"] == True
    ].copy()

    actionable_ids = set(
        actionable["cluster_id"].astype(int)
    )

    mapped_ids = set(
        NEED_MAP.keys()
    )

    missing = (
        actionable_ids
        - mapped_ids
    )

    extra = (
        mapped_ids
        - actionable_ids
    )

    if missing:
        raise ValueError(
            "Actionable clusters missing "
            f"from NEED_MAP: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            "NEED_MAP contains clusters "
            f"that are not actionable: {sorted(extra)}"
        )

    print(
        f"Actionable clusters: "
        f"{len(actionable_ids)}"
    )

    print(
        "Taxonomy mapping: OK"
    )

    # --------------------------------------------------------
    # Attach LLM labels + higher-order need
    # --------------------------------------------------------

    labels = labels.copy()

    labels[
        "cluster_id"
    ] = labels[
        "cluster_id"
    ].astype(int)

    labels[
        "higher_order_need"
    ] = labels[
        "cluster_id"
    ].map(
        NEED_MAP
    )

    df = topics.merge(
        labels[
            [
                "cluster_id",
                "label",
                "theme_type",
                "need_category",
                "actionable",
                "coherence",
                "evidence_strength",
                "cross_game_relevance",
                "higher_order_need",
            ]
        ],
        on="cluster_id",
        how="left",
    )

    # Only actionable semantic clusters
    analysis = df[
        (df["cluster_id"] >= 0)
        & (df["actionable"] == True)
        & (df["higher_order_need"].notna())
    ].copy()

    print(
        f"Actionable clustered reviews: "
        f"{len(analysis)}"
    )

    # --------------------------------------------------------
    # Overall reference
    # --------------------------------------------------------

    overall_recommend = (
        analysis["voted_up"].mean()
    )

    overall_playtime = (
        analysis[
            "playtime_at_review_hours"
        ].median()
    )

    print(
        f"Overall recommend rate: "
        f"{overall_recommend:.3f}"
    )

    print(
        f"Overall median playtime: "
        f"{overall_playtime:.2f} h"
    )

    # --------------------------------------------------------
    # Cluster-level metrics
    # --------------------------------------------------------

    cluster_rows = []

    for cluster_id, g in analysis.groupby(
        "cluster_id"
    ):

        group_share = (
            g["group"]
            .value_counts(
                normalize=True
            )
        )

        cluster_rows.append({
            "cluster_id":
                int(cluster_id),

            "label":
                g["label"].iloc[0],

            "higher_order_need":
                g[
                    "higher_order_need"
                ].iloc[0],

            "theme_type":
                g["theme_type"].iloc[0],

            "reviews":
                len(g),

            "games_present":
                g["title"].nunique(),

            "groups_present":
                g["group"].nunique(),

            "recommend_rate":
                g["voted_up"].mean(),

            "recommend_delta_pp":
                (
                    g["voted_up"].mean()
                    - overall_recommend
                ) * 100,

            "median_playtime_h":
                g[
                    "playtime_at_review_hours"
                ].median(),

            "contrast_share":
                group_share.get(
                    "contrast",
                    0.0,
                ),

            "core_share":
                group_share.get(
                    "core",
                    0.0,
                ),

            "positioning_share":
                group_share.get(
                    "positioning",
                    0.0,
                ),

            "benchmark_share":
                group_share.get(
                    "benchmark",
                    0.0,
                ),
        })

    cluster_metrics = pd.DataFrame(
        cluster_rows
    ).sort_values(
        [
            "higher_order_need",
            "reviews",
        ],
        ascending=[
            True,
            False,
        ],
    )

    # --------------------------------------------------------
    # Game-level intermediate table
    #
    # Used to calculate game-equal recommendation rates,
    # so one game contributing many reviews does not
    # completely determine the higher-order need result.
    # --------------------------------------------------------

    game_need = (
        analysis
        .groupby(
            [
                "higher_order_need",
                "title",
                "group",
            ],
            as_index=False,
        )
        .agg(
            reviews=(
                "recommendation_id",
                "count",
            ),
            recommend_rate=(
                "voted_up",
                "mean",
            ),
            median_playtime_h=(
                "playtime_at_review_hours",
                "median",
            ),
        )
    )

    # --------------------------------------------------------
    # Higher-order Need metrics
    # --------------------------------------------------------

    need_rows = []

    total_actionable_reviews = len(
        analysis
    )

    for need, g in analysis.groupby(
        "higher_order_need"
    ):

        cluster_ids = sorted(
            g["cluster_id"]
            .unique()
            .astype(int)
            .tolist()
        )

        group_share = (
            g["group"]
            .value_counts(
                normalize=True
            )
        )

        game_slice = game_need[
            game_need[
                "higher_order_need"
            ] == need
        ]

        weighted_recommend = (
            g["voted_up"].mean()
        )

        game_equal_recommend = (
            game_slice[
                "recommend_rate"
            ].mean()
        )

        need_rows.append({
            "higher_order_need":
                need,

            "clusters":
                ",".join(
                    map(
                        str,
                        cluster_ids,
                    )
                ),

            "cluster_count":
                len(cluster_ids),

            "reviews":
                len(g),

            "actionable_review_share":
                (
                    len(g)
                    / total_actionable_reviews
                ),

            "games_present":
                g["title"].nunique(),

            "groups_present":
                g["group"].nunique(),

            "review_weighted_recommend_rate":
                weighted_recommend,

            "game_equal_recommend_rate":
                game_equal_recommend,

            "recommend_delta_pp":
                (
                    weighted_recommend
                    - overall_recommend
                ) * 100,

            "median_playtime_h":
                g[
                    "playtime_at_review_hours"
                ].median(),

            "contrast_share":
                group_share.get(
                    "contrast",
                    0.0,
                ),

            "core_share":
                group_share.get(
                    "core",
                    0.0,
                ),

            "positioning_share":
                group_share.get(
                    "positioning",
                    0.0,
                ),

            "benchmark_share":
                group_share.get(
                    "benchmark",
                    0.0,
                ),
        })

    need_metrics = (
        pd.DataFrame(
            need_rows
        )
        .sort_values(
            "reviews",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cluster_path = (
        OUTPUT_DIR
        / "cluster_cross_validation.csv"
    )

    need_path = (
        OUTPUT_DIR
        / "need_cross_validation.csv"
    )

    game_path = (
        OUTPUT_DIR
        / "game_need_metrics.csv"
    )

    cluster_metrics.to_csv(
        cluster_path,
        index=False,
    )

    need_metrics.to_csv(
        need_path,
        index=False,
    )

    game_need.to_csv(
        game_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    display = need_metrics.copy()

    pct_cols = [
        "actionable_review_share",
        "review_weighted_recommend_rate",
        "game_equal_recommend_rate",
        "contrast_share",
        "core_share",
        "positioning_share",
        "benchmark_share",
    ]

    for col in pct_cols:
        display[col] = (
            display[col]
            * 100
        ).round(1)

    display[
        "median_playtime_h"
    ] = display[
        "median_playtime_h"
    ].round(2)

    display[
        "recommend_delta_pp"
    ] = display[
        "recommend_delta_pp"
    ].round(1)

    print()
    print("=" * 72)
    print("HIGHER-ORDER NEED SUMMARY")
    print("=" * 72)

    print(
        display[
            [
                "higher_order_need",
                "reviews",
                "actionable_review_share",
                "games_present",
                "review_weighted_recommend_rate",
                "game_equal_recommend_rate",
                "recommend_delta_pp",
                "median_playtime_h",
                "contrast_share",
                "core_share",
                "positioning_share",
                "benchmark_share",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {need_path}"
    )

    print(
        f"Saved: {cluster_path}"
    )

    print(
        f"Saved: {game_path}"
    )


if __name__ == "__main__":
    main()
