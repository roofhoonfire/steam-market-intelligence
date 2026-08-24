from pathlib import Path

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

CURRENT_LABELS_PATH = (
    ROOT
    / "outputs"
    / "take8"
    / "cluster_labels.csv"
)

BASELINE_LABELS_PATH = (
    ROOT
    / "outputs"
    / "baseline_11games"
    / "take8"
    / "cluster_labels.csv"
)

WITHIN_GAME_PATH = (
    ROOT
    / "outputs"
    / "take9"
    / "within_game_need_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "take10"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "market_intelligence_decision_sheet.md"
)


def pct(value):
    return f"{value * 100:.1f}%"


def pp(value):
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%p"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviews = pd.read_parquet(
        REVIEWS_PATH
    )

    games = pd.read_parquet(
        GAMES_PATH
    )

    current_labels = pd.read_csv(
        CURRENT_LABELS_PATH
    )

    baseline_labels = pd.read_csv(
        BASELINE_LABELS_PATH
    )

    within = pd.read_csv(
        WITHIN_GAME_PATH
    )

    # ========================================================
    # H1 — Exploration-oriented segment
    # ========================================================

    library = reviews[
        reviews["num_games_owned"] > 0
    ].copy()

    library = (
        library
        .sort_values(
            "num_games_owned",
            ascending=False,
        )
        .drop_duplicates(
            [
                "group",
                "reviewer_hash",
            ]
        )
    )

    library_summary = (
        library
        .groupby("group")[
            "num_games_owned"
        ]
        .agg(
            reviewers="count",
            median="median",
            q75=lambda x: x.quantile(.75),
        )
    )

    activity = (
        reviews
        .sort_values(
            "num_reviews",
            ascending=False,
        )
        .drop_duplicates(
            [
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
        .median()
    )

    # ========================================================
    # H3 / H4 — Price and short-experience examples
    # ========================================================

    game_review = (
        reviews
        .groupby(
            [
                "appid",
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

    game_market = games[
        [
            "appid",
            "steamspy_price_usd",
        ]
    ].copy()

    game_metrics = game_review.merge(
        game_market,
        on="appid",
        how="left",
    )

    price_band = game_metrics[
        game_metrics[
            "steamspy_price_usd"
        ].between(
            5.0,
            15.0,
            inclusive="both",
        )
    ].copy()

    short_success = game_metrics[
        (
            game_metrics[
                "median_playtime_h"
            ] <= 8.0
        )
        &
        (
            game_metrics[
                "recommend_rate"
            ] >= 0.92
        )
    ].copy()

    # ========================================================
    # H5 — Robustness: baseline vs expanded
    # ========================================================

    baseline_total = len(
        baseline_labels
    )

    baseline_narrative = (
        baseline_labels[
            "need_category"
        ]
        .eq(
            "Narrative & Emotion"
        )
        .sum()
    )

    current_total = len(
        current_labels
    )

    current_narrative = (
        current_labels[
            "need_category"
        ]
        .eq(
            "Narrative & Emotion"
        )
        .sum()
    )

    baseline_narrative_share = (
        baseline_narrative
        / baseline_total
    )

    current_narrative_share = (
        current_narrative
        / current_total
    )

    # ========================================================
    # Take 9 strongest within-game signals
    # ========================================================

    within_lookup = (
        within
        .set_index(
            "higher_order_need"
        )
    )

    def get_signal(need):

        row = within_lookup.loc[
            need
        ]

        return {
            "games":
                int(
                    row[
                        "games_compared"
                    ]
                ),

            "median_delta":
                float(
                    row[
                        "median_within_game_delta_pp"
                    ]
                ),

            "positive":
                int(
                    row[
                        "games_positive_delta"
                    ]
                ),

            "negative":
                int(
                    row[
                        "games_negative_delta"
                    ]
                ),
        }

    emotional = get_signal(
        "Emotional Narrative & Payoff"
    )

    identity = get_signal(
        "Distinctive Presentation & Atmosphere"
    )

    gameplay = get_signal(
        "Gameplay Depth & Agency"
    )

    replay = get_signal(
        "Replayability & Returnability"
    )

    longevity = get_signal(
        "Content Longevity & Expansion"
    )

    technical = get_signal(
        "Technical Accessibility"
    )

    # ========================================================
    # Markdown
    # ========================================================

    lines = []

    lines.append(
        "# Market Intelligence Decision Sheet"
    )

    lines.append("")

    lines.append(
        "## 1. Business Question"
    )

    lines.append("")

    lines.append(
        "Can a smaller indie game avoid direct content-volume competition "
        "with AAA titles by targeting exploration-oriented customers with "
        "a lower-entry-price product that offers a sharply differentiated "
        "experience?"
    )

    lines.append("")

    lines.append(
        "The analysis does not assume that AAA and indie games use separate "
        "wallet budgets. It tests observable customer and product signals "
        "that are consistent with that commercialization hypothesis."
    )

    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "## 2. Hypothesis Verdicts"
    )

    lines.append("")

    lines.append(
        "| Hypothesis | Verdict | Evidence |"
    )

    lines.append(
        "|---|---|---|"
    )

    core_median = (
        library_summary.loc[
            "core",
            "median",
        ]
    )

    contrast_median = (
        library_summary.loc[
            "contrast",
            "median",
        ]
    )

    benchmark_median = (
        library_summary.loc[
            "benchmark",
            "median",
        ]
    )

    core_activity = (
        activity_summary.loc[
            "core"
        ]
    )

    contrast_activity = (
        activity_summary.loc[
            "contrast"
        ]
    )

    benchmark_activity = (
        activity_summary.loc[
            "benchmark"
        ]
    )

    lines.append(
        "| H1. Exploration-oriented segment exists "
        "| **Supported within observed sample** "
        f"| Non-zero library median: Core {core_median:.1f}, "
        f"Contrast {contrast_median:.1f}, "
        f"AAA benchmark {benchmark_median:.1f}. "
        f"Median review activity: Core {core_activity:.1f}, "
        f"Contrast {contrast_activity:.1f}, "
        f"AAA {benchmark_activity:.1f}. |"
    )

    lines.append(
        "| H2. AAA and indie purchases are complementary "
        "| **Unresolved** "
        "| Review data does not provide reliable ownership overlap "
        "or wallet-spend allocation. Share of Wallet cannot be directly proven. |"
    )

    lines.append(
        "| H3. $5–15 supports a low-friction complementary position "
        "| **Partial / provisional** "
        f"| {len(price_band)} observed games fall in the $5–15 band "
        "and include highly rated commercial examples, but current price "
        "does not establish purchase-time price or causality. |"
    )

    lines.append(
        "| H4. Short experiences can still achieve high satisfaction "
        "| **Supported as feasibility, not causality** "
        f"| {len(short_success)} games in the sample had median review-time "
        "playtime ≤8h while maintaining ≥92% sample recommendation. |"
    )

    lines.append(
        "| H5. Customer needs can be discovered without pre-forced labels "
        "| **Supported** "
        f"| Narrative & Emotion accounted for "
        f"{baseline_narrative}/{baseline_total} "
        f"({pct(baseline_narrative_share)}) labels in the 11-game baseline, "
        f"falling to {current_narrative}/{current_total} "
        f"({pct(current_narrative_share)}) after gameplay-led contrast games "
        "were added. New Gameplay Depth, Replayability, Content Longevity, "
        "and Technical Accessibility themes emerged. |"
    )

    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "## 3. Validated Customer Value Structure"
    )

    lines.append("")

    lines.append(
        "### Selection Drivers"
    )

    lines.append("")

    lines.append(
        f"- **Distinctive Presentation & Atmosphere** — "
        f"median within-game recommendation delta "
        f"{pp(identity['median_delta'])}; "
        f"{identity['positive']}/{identity['games']} games positive."
    )

    lines.append(
        f"- **Emotional Narrative & Payoff** — "
        f"median within-game recommendation delta "
        f"{pp(emotional['median_delta'])}; "
        f"{emotional['positive']}/{emotional['games']} games positive."
    )

    lines.append(
        f"- **Replayability & Returnability** — "
        f"median within-game recommendation delta "
        f"{pp(replay['median_delta'])}; "
        f"{replay['positive']}/{replay['games']} games positive."
    )

    lines.append("")

    lines.append(
        "### Product Guardrails"
    )

    lines.append("")

    lines.append(
        f"- **Gameplay Depth & Agency** — "
        f"median within-game recommendation delta "
        f"{pp(gameplay['median_delta'])}; "
        f"{gameplay['negative']}/{gameplay['games']} games negative."
    )

    lines.append(
        f"- **Technical Accessibility** — "
        f"median within-game recommendation delta "
        f"{pp(technical['median_delta'])}; "
        f"{technical['negative']}/{technical['games']} games negative. "
        "This signal should be interpreted cautiously because the "
        "number of compared games is smaller."
    )

    lines.append("")

    lines.append(
        "### Post-Purchase Expansion Signal"
    )

    lines.append("")

    lines.append(
        f"- **Content Longevity & Expansion** — "
        f"median within-game recommendation delta "
        f"{pp(longevity['median_delta'])}; "
        f"{longevity['positive']}/{longevity['games']} games positive. "
        "Reviews explicitly request DLC, updates, and additional content, "
        "but this is not direct evidence of realized lifetime value."
    )

    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "## 4. Product Planning Interpretation"
    )

    lines.append("")

    lines.append(
        "**Narrative-led products** can compete through emotional payoff, "
        "memorable characters, music, and a distinctive audiovisual identity "
        "rather than through content volume alone."
    )

    lines.append("")

    lines.append(
        "**Gameplay-led products** can compete through immediately legible "
        "core loops, replayability, meaningful variation, mastery, and "
        "continued content demand."
    )

    lines.append("")

    lines.append(
        "Across both types, a differentiated hook is insufficient when "
        "gameplay becomes shallow, repetitive, passive, or technically "
        "difficult to access."
    )

    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "## 5. Commercialization Decision"
    )

    lines.append("")

    lines.append(
        "> Do not position the product as a smaller substitute for AAA. "
        "Target exploration-oriented customers with a low-entry-price, "
        "sharply differentiated experience that creates a clear additional "
        "purchase occasion."
    )

    lines.append("")

    lines.append(
        "The most defensible product proposition is therefore not "
        "\"more content for less money,\" but **a compact experience with "
        "an immediately recognizable reason to choose it**."
    )

    lines.append("")

    lines.append(
        "Candidate reasons-to-choose identified in this analysis are "
        "**emotional payoff, distinctive audiovisual identity, and "
        "replayable gameplay depth**."
    )

    # --------------------------------------------------------

    lines.append("")

    lines.append(
        "## 6. Evidence Boundary"
    )

    lines.append("")

    lines.append(
        "- Steam reviews represent an observed reviewer sample, not all buyers."
    )

    lines.append(
        "- `num_games_owned == 0` is ambiguous and was excluded from "
        "library-breadth comparisons."
    )

    lines.append(
        "- SteamSpy owner counts are third-party estimates, not actual sales."
    )

    lines.append(
        "- Current Steam prices are not guaranteed to equal historical "
        "purchase prices."
    )

    lines.append(
        "- Recommendation and playtime associations do not establish causality."
    )

    lines.append(
        "- Actual AAA-to-indie cross-ownership, wallet allocation, and "
        "Share of Wallet were not directly observed."
    )

    lines.append("")

    lines.append(
        "Accordingly, the final Share-of-Wallet proposition remains a "
        "**commercialization hypothesis supported by behavioral proxies, "
        "not a proven causal fact**."
    )

    # --------------------------------------------------------

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print("TAKE 10 — MARKET INTELLIGENCE DECISION SHEET")
    print("=" * 72)

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("=== H1 ===")
    print(
        f"Library median | "
        f"core={core_median:.1f}, "
        f"contrast={contrast_median:.1f}, "
        f"benchmark={benchmark_median:.1f}"
    )

    print()
    print("=== ROBUSTNESS ===")
    print(
        f"Narrative label share | "
        f"baseline={pct(baseline_narrative_share)} "
        f"-> expanded={pct(current_narrative_share)}"
    )

    print()
    print("=== STRONGEST POSITIVE SIGNALS ===")
    print(
        f"Distinctive Identity : "
        f"{pp(identity['median_delta'])}"
    )
    print(
        f"Replayability        : "
        f"{pp(replay['median_delta'])}"
    )
    print(
        f"Emotional Payoff     : "
        f"{pp(emotional['median_delta'])}"
    )

    print()
    print("=== STRONGEST NEGATIVE SIGNALS ===")
    print(
        f"Gameplay Depth       : "
        f"{pp(gameplay['median_delta'])}"
    )
    print(
        f"Technical Access     : "
        f"{pp(technical['median_delta'])}"
    )

    print()
    print("TAKE 10 COMPLETE")


if __name__ == "__main__":
    main()
