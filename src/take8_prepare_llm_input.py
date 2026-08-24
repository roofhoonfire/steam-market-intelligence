from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TOPICS_PATH = (
    ROOT
    / "data"
    / "features"
    / "review_topics_initial.parquet"
)

REPS_PATH = (
    ROOT
    / "outputs"
    / "take7"
    / "diverse_representative_reviews.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "take8"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "llm_cluster_input.json"
)


def build_cluster_quality(df):

    rows = []

    clustered = df[
        df["cluster_id"] >= 0
    ]

    for cluster_id, group in clustered.groupby(
        "cluster_id"
    ):

        game_counts = (
            group["title"]
            .value_counts()
        )

        dominant_game = (
            game_counts.index[0]
        )

        dominant_game_share = (
            game_counts.iloc[0]
            / len(group)
        )

        rows.append(
            {
                "cluster_id":
                    int(cluster_id),

                "reviews":
                    int(len(group)),

                "games_present":
                    int(
                        group[
                            "title"
                        ].nunique()
                    ),

                "dominant_game":
                    dominant_game,

                "dominant_game_share":
                    float(
                        dominant_game_share
                    ),

                "recommend_rate":
                    float(
                        group[
                            "voted_up"
                        ].mean()
                    ),

                "median_playtime_h":
                    float(
                        group[
                            "playtime_at_review_hours"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():

    topics = pd.read_parquet(
        TOPICS_PATH
    )

    reps = pd.read_csv(
        REPS_PATH
    )

    quality = (
        build_cluster_quality(
            topics
        )
    )

    # --------------------------------------------
    # Cross-game candidate operational rule
    # --------------------------------------------

    candidates = quality[
        (
            quality["games_present"]
            >= 3
        )
        &
        (
            quality[
                "dominant_game_share"
            ]
            <= 0.60
        )
    ].copy()

    
    payload = []

    for _, meta in (
        candidates
        .sort_values(
            "reviews",
            ascending=False
        )
        .iterrows()
    ):

        cluster_id = int(
            meta["cluster_id"]
        )

        cluster_reps = (
            reps[
                reps["cluster_id"]
                == cluster_id
            ]
            .copy()
        )

        reviews = []

        for _, row in (
            cluster_reps.iterrows()
        ):

            reviews.append(
                {
                    "title":
                        row["title"],

                    "voted_up":
                        bool(
                            row[
                                "voted_up"
                            ]
                        ),

                    "playtime_h":
                        (
                            None
                            if pd.isna(
                                row[
                                    "playtime_at_review_hours"
                                ]
                            )
                            else round(
                                float(
                                    row[
                                        "playtime_at_review_hours"
                                    ]
                                ),
                                2,
                            )
                        ),

                    "text":
                        row[
                            "analysis_text"
                        ],
                }
            )

        payload.append(
            {
                "cluster_id":
                    cluster_id,

                "cluster_stats":
                    {
                        "reviews":
                            int(
                                meta[
                                    "reviews"
                                ]
                            ),

                        "games_present":
                            int(
                                meta[
                                    "games_present"
                                ]
                            ),

                        "recommend_rate":
                            round(
                                float(
                                    meta[
                                        "recommend_rate"
                                    ]
                                ),
                                4,
                            ),

                        "median_playtime_h":
                            round(
                                float(
                                    meta[
                                        "median_playtime_h"
                                    ]
                                ),
                                2,
                            ),

                        "dominant_game":
                            meta[
                                "dominant_game"
                            ],

                        "dominant_game_share":
                            round(
                                float(
                                    meta[
                                        "dominant_game_share"
                                    ]
                                ),
                                4,
                            ),
                    },

                "representative_reviews":
                    reviews,
            }
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Cross-game clusters:",
        len(payload)
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )

    print()

    print(
        "Clusters:"
    )

    for cluster in payload:

        stats = (
            cluster[
                "cluster_stats"
            ]
        )

        print(
            f"  {cluster['cluster_id']:2d}"
            f" | reviews={stats['reviews']:3d}"
            f" | games={stats['games_present']:2d}"
            f" | recommend={stats['recommend_rate']:.3f}"
            f" | reps={len(cluster['representative_reviews'])}"
        )


if __name__ == "__main__":
    main()
