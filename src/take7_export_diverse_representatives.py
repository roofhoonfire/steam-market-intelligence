from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "features"
    / "review_topics_initial.parquet"
)

LANG_INPUT = (
    ROOT
    / "data"
    / "features"
    / "topic_corpus_language_checked.parquet"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "take7"
    / "diverse_representative_reviews.csv"
)


def main():

    # ==========================================================
    # 1. Load clustered review dataset
    # ==========================================================

    df = pd.read_parquet(INPUT)

    # ==========================================================
    # 2. Load detected-language result
    # ==========================================================

    language = pd.read_parquet(
        LANG_INPUT,
        columns=[
            "recommendation_id",
            "detected_language",
        ],
    )

    # ==========================================================
    # 3. Merge language metadata
    # ==========================================================

    df = df.merge(
        language,
        on="recommendation_id",
        how="left",
    )

    # ==========================================================
    # 4. Keep only clustered + detected-English reviews
    # ==========================================================

    df = df[
        (df["cluster_id"] >= 0)
        &
        (df["detected_language"] == "en")
    ].copy()

    print(
        "English clustered reviews:",
        len(df)
    )

    rows = []

    # ==========================================================
    # 5. Build diversified representatives per cluster
    # ==========================================================

    for cluster_id, cluster in df.groupby(
        "cluster_id"
    ):

        # ------------------------------------------------------
        # A. Avoid one-game domination
        #
        # 각 게임에서 cluster probability가 높은
        # 리뷰를 최대 2개까지만 확보
        # ------------------------------------------------------

        per_game = (
            cluster
            .sort_values(
                "cluster_probability",
                ascending=False,
            )
            .groupby(
                "title",
                group_keys=False,
            )
            .head(2)
        )

        # ------------------------------------------------------
        # B. Main representative set
        #
        # 게임별 대표 후보 중 확률 높은 순으로
        # 최대 12개 선택
        # ------------------------------------------------------

        selected = (
            per_game
            .sort_values(
                "cluster_probability",
                ascending=False,
            )
            .head(12)
        )

        # ------------------------------------------------------
        # C. Add negative reviews
        #
        # 긍정 리뷰만 대표되는 것을 막기 위해
        # 해당 cluster에 부정 리뷰가 존재하면
        # probability 높은 부정 리뷰 최대 3개 추가
        # ------------------------------------------------------

        negative = (
            cluster[
                cluster["voted_up"] == False
            ]
            .sort_values(
                "cluster_probability",
                ascending=False,
            )
            .head(3)
        )

        selected = pd.concat(
            [
                selected,
                negative,
            ]
        )

        # ------------------------------------------------------
        # D. Remove duplicate reviews
        # ------------------------------------------------------

        selected = (
            selected
            .drop_duplicates(
                subset=[
                    "recommendation_id"
                ]
            )
            .head(15)
        )

        rows.append(
            selected
        )

    # ==========================================================
    # 6. Merge all representative reviews
    # ==========================================================

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    columns = [
        "cluster_id",
        "cluster_probability",
        "title",
        "group",
        "voted_up",
        "playtime_at_review_hours",
        "detected_language",
        "analysis_text",
    ]

    # ==========================================================
    # 7. Save result
    # ==========================================================

    result[
        columns
    ].to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        "Saved:",
        OUTPUT
    )

    print(
        "Representative reviews:",
        len(result)
    )

    print()

    # ==========================================================
    # 8. Quality summary
    # ==========================================================

    summary = (
        result
        .groupby("cluster_id")
        .agg(
            representative_reviews=(
                "recommendation_id",
                "count",
            ),
            games_represented=(
                "title",
                "nunique",
            ),
            negative_reviews=(
                "voted_up",
                lambda x:
                    (~x).sum(),
            ),
        )
    )

    print(
        summary.to_string()
    )


if __name__ == "__main__":
    main()
