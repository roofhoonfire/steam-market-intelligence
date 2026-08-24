from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]

REVIEWS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "reviews.parquet"
)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def rank_biserial(x, y):
    """
    Mann-Whitney U 기반 rank-biserial effect size.

    + : x가 더 큰 방향
    - : y가 더 큰 방향
    """

    result = mannwhitneyu(
        x,
        y,
        alternative="two-sided",
    )

    n1 = len(x)
    n2 = len(y)

    effect = (
        2 * result.statistic
        / (n1 * n2)
        - 1
    )

    return (
        result.statistic,
        result.pvalue,
        effect,
    )


def main():

    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    # ==========================================================
    # 1. Zero audit
    # ==========================================================

    section(
        "1. ZERO-VALUE AUDIT"
    )

    zero_group = (
        reviews
        .assign(
            zero=
            reviews["num_games_owned"].eq(0)
        )
        .groupby("group")["zero"]
        .agg(
            count="sum",
            share="mean",
        )
    )

    print(
        zero_group.to_string()
    )

    print()

    zero_game = (
        reviews
        .assign(
            zero=
            reviews["num_games_owned"].eq(0)
        )
        .groupby(
            [
                "group",
                "title",
            ]
        )["zero"]
        .mean()
        .sort_values()
    )

    print(
        "Zero share by game:"
    )

    print(
        zero_game
        .round(3)
        .to_string()
    )

    # ==========================================================
    # 2. Correct group-level dedup
    # ==========================================================

    section(
        "2. WITHIN-GROUP UNIQUE REVIEWERS"
    )

    library = reviews[
        reviews["num_games_owned"] > 0
    ].copy()

    # 같은 그룹 안에서는 한 reviewer를 한 번만 사용.
    # 다른 그룹에 등장하면 양쪽 그룹 모두 유지.
    library = (
        library
        .sort_values(
            "num_games_owned",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "group",
                "reviewer_hash",
            ]
        )
        .reset_index(drop=True)
    )

    summary = (
        library
        .groupby("group")[
            "num_games_owned"
        ]
        .agg(
            reviewers="count",
            mean="mean",
            median="median",
            q25=lambda x:
                x.quantile(0.25),
            q75=lambda x:
                x.quantile(0.75),
            q90=lambda x:
                x.quantile(0.90),
        )
        .round(1)
    )

    print(
        summary.to_string()
    )

    # ==========================================================
    # 3. Explorer sensitivity
    # ==========================================================

    section(
        "3. EXPLORER SHARE"
    )

    # 전체 non-zero unique reviewer 기준 Q75
    global_unique = (
        reviews[
            reviews["num_games_owned"] > 0
        ]
        .sort_values(
            "num_games_owned",
            ascending=False,
        )
        .drop_duplicates(
            subset=["reviewer_hash"]
        )
    )

    threshold = (
        global_unique[
            "num_games_owned"
        ]
        .quantile(0.75)
    )

    print(
        "Explorer threshold:",
        round(threshold, 1),
    )

    library["explorer"] = (
        library["num_games_owned"]
        >= threshold
    )

    explorer = (
        library
        .groupby("group")[
            "explorer"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        explorer
        .round(3)
        .to_string()
    )

    # ==========================================================
    # 4. Effect sizes
    # ==========================================================

    section(
        "4. PAIRWISE EFFECT SIZE"
    )

    values = {
        group:
        df["num_games_owned"].to_numpy()

        for group, df
        in library.groupby("group")
    }

    for a, b in combinations(
        values.keys(),
        2,
    ):

        U, p, effect = (
            rank_biserial(
                values[a],
                values[b],
            )
        )

        print(
            f"{a:12} vs {b:12}"
            f" | U={U:.1f}"
            f" | p={p:.3e}"
            f" | rank-biserial={effect:.3f}"
        )

    print()
    print(
        "Effect-size rough guide:"
    )
    print(
        "|r| ~0.1 small, "
        "~0.3 medium, "
        "~0.5+ large"
    )

    # ==========================================================
    # 5. Secondary behavioral proxy: num_reviews
    # ==========================================================

    section(
        "5. REVIEW ACTIVITY AS SECONDARY PROXY"
    )

    activity = (
        reviews
        .sort_values(
            "num_reviews",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "group",
                "reviewer_hash",
            ]
        )
    )

    activity_summary = (
        activity
        .groupby("group")[
            "num_reviews"
        ]
        .agg(
            reviewers="count",
            median="median",
            q75=lambda x:
                x.quantile(0.75),
            q90=lambda x:
                x.quantile(0.90),
        )
        .round(1)
    )

    print(
        activity_summary.to_string()
    )

    # ==========================================================
    # 6. Observed cross-title reviewer overlap
    # ==========================================================

    section(
        "6. OBSERVED REVIEWER OVERLAP"
    )

    reviewer_games = (
        reviews
        .groupby("reviewer_hash")[
            "appid"
        ]
        .nunique()
    )

    print(
        "Unique reviewers:",
        len(reviewer_games)
    )

    for n in [
        2,
        3,
        4,
    ]:

        count = (
            reviewer_games >= n
        ).sum()

        print(
            f"Reviewed >= {n} selected games: "
            f"{count} "
            f"({count / len(reviewer_games):.1%})"
        )

    reviewer_groups = (
        reviews
        .groupby("reviewer_hash")[
            "group"
        ]
        .agg(set)
    )

    pairs = [
        ("core", "positioning"),
        ("core", "benchmark"),
        ("positioning", "benchmark"),
    ]

    print()

    for a, b in pairs:

        count = reviewer_groups.apply(
            lambda groups:
                a in groups
                and b in groups
        ).sum()

        print(
            f"{a:12} <-> {b:12}: "
            f"{count} observed reviewers"
        )

    print()
    print(
        "주의: 이것은 ownership overlap이 아니라"
    )
    print(
        "이번에 수집한 11개 게임의 "
        "review-author overlap이다."
    )


if __name__ == "__main__":
    main()
