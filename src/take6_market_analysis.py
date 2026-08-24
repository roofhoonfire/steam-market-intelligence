from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def price_band(price):
    if pd.isna(price):
        return "unknown"

    if price < 5:
        return "<$5"

    if price < 10:
        return "$5-10"

    if price < 15:
        return "$10-15"

    if price < 25:
        return "$15-25"

    return "$25+"


def build_game_level_dataset(
    reviews,
    games,
):
    rows = []

    for title, df in reviews.groupby(
        "title"
    ):
        purchased = df[
            (df["steam_purchase"] == True)
            &
            (df["received_for_free"] == False)
        ]

        rows.append(
            {
                "title":
                    title,

                "sample_reviews":
                    len(df),

                "sample_recommend_rate":
                    df["voted_up"].mean(),

                "purchase_subset_reviews":
                    len(purchased),

                "purchase_subset_recommend_rate":
                    (
                        purchased["voted_up"].mean()
                        if len(purchased) > 0
                        else None
                    ),

                "median_playtime_at_review_h":
                    df[
                        "playtime_at_review_hours"
                    ].median(),

                "q25_playtime_at_review_h":
                    df[
                        "playtime_at_review_hours"
                    ].quantile(0.25),

                "q75_playtime_at_review_h":
                    df[
                        "playtime_at_review_hours"
                    ].quantile(0.75),

                "median_num_reviews":
                    df[
                        "num_reviews"
                    ].median(),

                "received_for_free_share":
                    df[
                        "received_for_free"
                    ].mean(),

                "steam_purchase_share":
                    df[
                        "steam_purchase"
                    ].mean(),
            }
        )

    review_metrics = pd.DataFrame(
        rows
    )

    result = games.merge(
        review_metrics,
        on="title",
        how="left",
    )

    result["price_band"] = (
        result[
            "steamspy_price_usd"
        ]
        .apply(price_band)
    )

    return result


def print_game_table(df):
    section(
        "1. GAME-LEVEL MARKET / BEHAVIOR TABLE"
    )

    columns = [
        "title",
        "group",
        "steamspy_price_usd",
        "price_band",
        "steamspy_owners_raw",
        "sample_reviews",
        "median_playtime_at_review_h",
        "sample_recommend_rate",
        "purchase_subset_recommend_rate",
    ]

    print(
        df[
            columns
        ]
        .sort_values(
            "steamspy_price_usd"
        )
        .round(
            {
                "median_playtime_at_review_h": 2,
                "sample_recommend_rate": 3,
                "purchase_subset_recommend_rate": 3,
            }
        )
        .to_string(
            index=False
        )
    )


def price_band_summary(df):
    section(
        "2. PRICE-BAND DESCRIPTIVE SUMMARY"
    )

    summary = (
        df
        .groupby(
            "price_band",
            observed=True,
        )
        .agg(
            games=(
                "appid",
                "count",
            ),
            mean_recommend_rate=(
                "sample_recommend_rate",
                "mean",
            ),
            median_game_playtime_h=(
                "median_playtime_at_review_h",
                "median",
            ),
        )
        .round(3)
    )

    print(
        summary.to_string()
    )

    print()
    print(
        "주의: 게임 수가 작고 각 price band의 "
        "장르/브랜드/출시시점이 다르므로 "
        "가격 효과의 인과추정으로 해석하지 않는다."
    )

    summary.to_csv(
        OUTPUT_DIR
        / "price_band_summary.csv"
    )


def short_experience_snapshot(df):
    section(
        "3. PLAYTIME vs SATISFACTION"
    )

    snapshot = (
        df[
            [
                "title",
                "group",
                "steamspy_price_usd",
                "median_playtime_at_review_h",
                "sample_recommend_rate",
                "steamspy_owners_raw",
            ]
        ]
        .sort_values(
            "median_playtime_at_review_h"
        )
    )

    print(
        snapshot
        .round(
            {
                "median_playtime_at_review_h": 2,
                "sample_recommend_rate": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "해석 포인트:"
    )

    print(
        "짧은 median playtime을 가진 게임에서도 "
        "높은 recommend rate가 유지되는 사례가 존재하는지 본다."
    )


def purchase_sensitivity(df):
    section(
        "4. PURCHASED-USER SENSITIVITY"
    )

    snapshot = (
        df[
            [
                "title",
                "sample_recommend_rate",
                "purchase_subset_recommend_rate",
                "purchase_subset_reviews",
                "steam_purchase_share",
                "received_for_free_share",
            ]
        ]
        .copy()
    )

    snapshot[
        "recommend_gap_purchase_minus_all"
    ] = (
        snapshot[
            "purchase_subset_recommend_rate"
        ]
        -
        snapshot[
            "sample_recommend_rate"
        ]
    )

    print(
        snapshot
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "가격/Value 해석에서는 전체 리뷰뿐 아니라 "
        "Steam 구매자 + 무료 제공 아님 subset도 함께 본다."
    )


def plot_price_vs_recommend(df):
    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        df["steamspy_price_usd"],
        df["sample_recommend_rate"],
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["title"],
            (
                row["steamspy_price_usd"],
                row["sample_recommend_rate"],
            ),
            fontsize=8,
        )

    plt.xlabel(
        "SteamSpy current price (USD)"
    )

    plt.ylabel(
        "Recent English sample recommend rate"
    )

    plt.title(
        "Price vs Review Satisfaction"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "price_vs_recommend.png"
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


def plot_playtime_vs_recommend(df):
    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        df["median_playtime_at_review_h"],
        df["sample_recommend_rate"],
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["title"],
            (
                row["median_playtime_at_review_h"],
                row["sample_recommend_rate"],
            ),
            fontsize=8,
        )

    plt.xlabel(
        "Median playtime at review (hours)"
    )

    plt.ylabel(
        "Recent English sample recommend rate"
    )

    plt.title(
        "Experience Length vs Review Satisfaction"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "playtime_vs_recommend.png"
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


def main():
    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    games = pd.read_parquet(
        GAMES_PATH
    )

    dataset = (
        build_game_level_dataset(
            reviews,
            games,
        )
    )

    output = (
        OUTPUT_DIR
        / "game_level_market_metrics.csv"
    )

    dataset.to_csv(
        output,
        index=False,
    )

    print(
        "Saved:",
        output
    )

    print_game_table(
        dataset
    )

    price_band_summary(
        dataset
    )

    short_experience_snapshot(
        dataset
    )

    purchase_sensitivity(
        dataset
    )

    section(
        "5. PLOTS"
    )

    plot_price_vs_recommend(
        dataset
    )

    plot_playtime_vs_recommend(
        dataset
    )

    section(
        "TAKE 6-B COMPLETE"
    )


if __name__ == "__main__":
    main()
