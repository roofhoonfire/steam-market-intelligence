from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]

REVIEWS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "reviews.parquet"
)

GAMES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "games.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "take6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def basic_audit(reviews):
    print_section(
        "1. DATA QUALITY AUDIT"
    )

    print(
        "reviews:",
        len(reviews)
    )

    print(
        "unique reviewers:",
        reviews["reviewer_hash"].nunique()
    )

    duplicated_reviewers = (
        reviews["reviewer_hash"]
        .duplicated(keep=False)
        .sum()
    )

    print(
        "rows from reviewers appearing multiple times:",
        duplicated_reviewers
    )

    zero_count = (
        reviews["num_games_owned"]
        .eq(0)
        .sum()
    )

    print(
        "num_games_owned == 0:",
        zero_count,
        f"({zero_count / len(reviews):.1%})"
    )

    print()

    print(
        "zero share by group:"
    )

    zero_by_group = (
        reviews
        .assign(
            games_owned_zero=
            reviews["num_games_owned"].eq(0)
        )
        .groupby("group")[
            "games_owned_zero"
        ]
        .mean()
        .sort_values()
    )

    print(
        zero_by_group
        .round(3)
        .to_string()
    )

    return zero_by_group


def build_unique_reviewer_dataset(
    reviews
):
    """
    H1 분석용 데이터.

    1) games_owned == 0 제외
    2) 동일 reviewer가 여러 리뷰에 등장하면
       reviewer당 하나만 남김

    같은 reviewer의 num_games_owned가
    여러 행에 존재할 경우 최대값을 사용한다.
    """

    clean = reviews[
        reviews["num_games_owned"] > 0
    ].copy()

    clean = (
        clean
        .sort_values(
            "num_games_owned",
            ascending=False
        )
        .drop_duplicates(
            subset=["reviewer_hash"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return clean


def overall_distribution(clean):
    print_section(
        "2. UNIQUE REVIEWER LIBRARY BREADTH"
    )

    x = clean[
        "num_games_owned"
    ]

    summary = {
        "reviewers":
            len(x),

        "mean":
            x.mean(),

        "median":
            x.median(),

        "q25":
            x.quantile(0.25),

        "q75":
            x.quantile(0.75),

        "q90":
            x.quantile(0.90),

        "q95":
            x.quantile(0.95),

        "max":
            x.max(),
    }

    for key, value in summary.items():

        if key == "reviewers":
            print(
                f"{key:12}: {value}"
            )

        else:
            print(
                f"{key:12}: {value:.1f}"
            )

    print()
    print(
        "Practical breadth thresholds:"
    )

    for threshold in [
        100,
        500,
        1000,
    ]:

        share = (
            x.ge(threshold)
            .mean()
        )

        print(
            f">= {threshold:4} games: "
            f"{share:.1%}"
        )


def group_summary(clean):
    print_section(
        "3. LIBRARY BREADTH BY GROUP"
    )

    summary = (
        clean
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

    summary.to_csv(
        OUTPUT_DIR
        / "library_breadth_by_group.csv"
    )

    return summary


def game_summary(clean):
    print_section(
        "4. LIBRARY BREADTH BY GAME"
    )

    summary = (
        clean
        .groupby(
            [
                "group",
                "title",
            ]
        )[
            "num_games_owned"
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
        .sort_values(
            "median",
            ascending=False,
        )
    )

    print(
        summary.to_string()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "library_breadth_by_game.csv"
    )


def statistical_tests(clean):
    print_section(
        "5. GROUP DISTRIBUTION TESTS"
    )

    groups = {}

    for group, df in clean.groupby(
        "group"
    ):

        groups[group] = (
            df["num_games_owned"]
            .to_numpy()
        )

    if len(groups) >= 2:

        result = kruskal(
            *groups.values()
        )

        print(
            "Kruskal-Wallis"
        )

        print(
            f"H = {result.statistic:.3f}"
        )

        print(
            f"p = {result.pvalue:.6g}"
        )

    print()
    print(
        "Pairwise Mann-Whitney U"
    )

    print(
        "(Bonferroni-adjusted p)"
    )

    pairs = list(
        combinations(
            groups.keys(),
            2,
        )
    )

    n_tests = len(pairs)

    for a, b in pairs:

        result = mannwhitneyu(
            groups[a],
            groups[b],
            alternative="two-sided",
        )

        adjusted_p = min(
            result.pvalue
            * n_tests,
            1.0,
        )

        print(
            f"{a:12} vs "
            f"{b:12}: "
            f"U={result.statistic:.1f}, "
            f"p_adj={adjusted_p:.6g}"
        )

    print()
    print(
        "주의: 유의성은 분포 차이를 뜻할 뿐,"
    )

    print(
        "가격이나 게임 유형이 소비행동을"
        " '원인'으로 만들었다는 의미는 아니다."
    )


def define_explorer_segment(clean):
    print_section(
        "6. DATA-DRIVEN EXPLORER SEGMENT"
    )

    threshold = (
        clean[
            "num_games_owned"
        ]
        .quantile(0.75)
    )

    print(
        "Explorer threshold "
        "(overall Q75):",
        round(threshold, 1),
        "games"
    )

    clean = clean.copy()

    clean["explorer"] = (
        clean["num_games_owned"]
        >= threshold
    )

    explorer_by_group = (
        clean
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
        "Explorer share by group:"
    )

    print(
        explorer_by_group
        .round(3)
        .to_string()
    )

    explorer_by_game = (
        clean
        .groupby(
            [
                "group",
                "title",
            ]
        )[
            "explorer"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Explorer share by game:"
    )

    print(
        explorer_by_game
        .round(3)
        .to_string()
    )

    explorer_by_group.to_csv(
        OUTPUT_DIR
        / "explorer_share_by_group.csv"
    )


def plot_library_boxplot(clean):
    groups = [
        "core",
        "positioning",
        "benchmark",
    ]

    values = []

    labels = []

    for group in groups:

        subset = clean[
            clean["group"] == group
        ][
            "num_games_owned"
        ]

        if len(subset) > 0:

            values.append(
                np.log10(
                    subset + 1
                )
            )

            labels.append(
                group
            )

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        values,
        labels=labels,
        showfliers=False,
    )

    plt.ylabel(
        "log10(Games owned + 1)"
    )

    plt.xlabel(
        "Comparable group"
    )

    plt.title(
        "Reviewer Library Breadth by Group"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "library_breadth_boxplot.png"
    )

    plt.savefig(
        output,
        dpi=160,
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def plot_ecdf(clean):
    plt.figure(
        figsize=(8, 6)
    )

    for group in [
        "core",
        "positioning",
        "benchmark",
    ]:

        values = (
            clean.loc[
                clean["group"] == group,
                "num_games_owned",
            ]
            .sort_values()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        y = (
            np.arange(
                1,
                len(values) + 1
            )
            / len(values)
        )

        plt.plot(
            values,
            y,
            label=group,
        )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "Games owned (log scale)"
    )

    plt.ylabel(
        "Cumulative share of reviewers"
    )

    plt.title(
        "ECDF of Reviewer Library Breadth"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "library_breadth_ecdf.png"
    )

    plt.savefig(
        output,
        dpi=160,
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def market_snapshot(games):
    print_section(
        "7. MARKET SNAPSHOT"
    )

    columns = [
        "title",
        "group",
        "steamspy_price_usd",
        "steamspy_owners_raw",
        "steamspy_positive_rate",
    ]

    print(
        games[
            columns
        ]
        .sort_values(
            [
                "group",
                "steamspy_price_usd",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "SteamSpy owner range는 추정값이며,"
    )

    print(
        "판매량/매출의 실제 관측값으로 "
        "해석하지 않는다."
    )


def main():

    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    games = pd.read_parquet(
        GAMES_PATH
    )

    basic_audit(
        reviews
    )

    clean = (
        build_unique_reviewer_dataset(
            reviews
        )
    )

    overall_distribution(
        clean
    )

    group_summary(
        clean
    )

    game_summary(
        clean
    )

    statistical_tests(
        clean
    )

    define_explorer_segment(
        clean
    )

    print_section(
        "8. PLOTS"
    )

    plot_library_boxplot(
        clean
    )

    plot_ecdf(
        clean
    )

    market_snapshot(
        games
    )

    print_section(
        "TAKE 6 H1 ANALYSIS COMPLETE"
    )

    print(
        "Outputs:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
