from pathlib import Path
from typing import Literal
import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "take8"
    / "llm_cluster_input.json"
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

RESULT_JSON = (
    OUTPUT_DIR
    / "cluster_labels.json"
)

RESULT_CSV = (
    OUTPUT_DIR
    / "cluster_labels.csv"
)

MODEL = "gpt-5.6-luna"


# ============================================================
# Structured Output Schema
# ============================================================

class ClusterLabel(BaseModel):

    cluster_id: int

    label: str = Field(
        description=(
            "A concise product/customer theme label, "
            "preferably 3-8 words."
        )
    )

    theme_type: Literal[
        "positive_driver",
        "pain_point",
        "mixed_tradeoff",
        "neutral_need",
        "non_actionable",
    ]

    need_category: str = Field(
        description=(
            "A broader customer-need category such as "
            "Narrative & Emotion, Gameplay Depth, "
            "Visual Identity, Value, Pacing, "
            "Usability, Exploration, etc. "
            "Do not force one of these examples."
        )
    )

    summary: str = Field(
        description=(
            "A concise factual summary of the common "
            "theme found across the representative reviews."
        )
    )

    customer_need: str = Field(
        description=(
            "Express the underlying customer need or pain "
            "in product-planning language."
        )
    )

    coherence: Literal[
        "high",
        "medium",
        "low",
    ]

    evidence_strength: Literal[
        "high",
        "medium",
        "low",
    ]

    cross_game_relevance: Literal[
        "high",
        "medium",
        "low",
    ]

    actionable: bool

    evidence_review_indices: list[int] = Field(
        description=(
            "Indices of representative reviews that most "
            "clearly support the assigned label. "
            "Use indices supplied in the prompt."
        )
    )

    caveat: str = Field(
        description=(
            "Any important limitation, conflicting signal, "
            "or reason this cluster should not be overinterpreted."
        )
    )


SYSTEM_PROMPT = """
You are labeling semantic clusters of Steam game reviews
for a product-planning market-intelligence analysis.

The clusters were created BEFORE this step using:
semantic embeddings -> UMAP -> HDBSCAN.

Your role is ONLY to convert an already-discovered cluster
into an interpretable customer-need label.

Rules:

1. Derive the semantic theme ONLY from the review texts.
2. Do not invent a theme merely because a cluster exists.
3. If the reviews are mainly generic praise, memes,
   game-specific references, or semantically incoherent,
   use theme_type="non_actionable".
4. Look for a theme recurring across multiple games,
   not merely one game's lore, character, or proper nouns.
5. Separate:
   - positive value drivers
   - pain points
   - genuine trade-offs
   - neutral customer needs
6. Do NOT use recommend_rate to decide what the cluster means.
   Recommendation statistics are metadata for later analysis.
7. Do NOT infer causality from price, playtime, or recommendation.
8. A short game is not automatically good or bad.
9. Prefer product-planning language:
   customer experience, gameplay depth, pacing,
   emotional attachment, identity, value perception, etc.
10. If evidence is mixed, explicitly say so.
11. evidence_review_indices must refer only to review indices
    present in the supplied cluster.
12. Keep the summary and customer_need concise.
"""


def load_input():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_existing_results():

    if not RESULT_JSON.exists():
        return []

    with open(
        RESULT_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_results(results):

    results = sorted(
        results,
        key=lambda x: x["cluster_id"],
    )

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        results
    ).to_csv(
        RESULT_CSV,
        index=False,
    )


def build_user_prompt(cluster):

    stats = cluster["cluster_stats"]
    reviews = cluster["representative_reviews"]

    lines = []

    lines.append(
        f"Cluster ID: {cluster['cluster_id']}"
    )

    lines.append("")
    lines.append("Cluster metadata:")
    lines.append(
        f"- total clustered reviews: {stats['reviews']}"
    )
    lines.append(
        f"- games present: {stats['games_present']}"
    )
    lines.append(
        f"- recommend rate: {stats['recommend_rate']}"
    )
    lines.append(
        f"- median playtime hours: "
        f"{stats['median_playtime_h']}"
    )
    lines.append(
        f"- dominant game: "
        f"{stats['dominant_game']}"
    )
    lines.append(
        f"- dominant game share: "
        f"{stats['dominant_game_share']}"
    )

    lines.append("")
    lines.append(
        "Representative reviews:"
    )

    for i, review in enumerate(reviews):

        lines.append("")
        lines.append(
            f"[Review {i}]"
        )

        lines.append(
            f"Game: {review['title']}"
        )

        lines.append(
            f"Recommended: "
            f"{review['voted_up']}"
        )

        lines.append(
            f"Playtime hours: "
            f"{review['playtime_h']}"
        )

        lines.append(
            f"Text: {review['text']}"
        )

    lines.append("")
    lines.append(
        "Label this cluster according to the schema."
    )

    return "\n".join(lines)


def label_cluster(
    client,
    cluster,
):

    user_prompt = build_user_prompt(
        cluster
    )

    response = client.responses.parse(
        model=MODEL,

        reasoning={
            "effort": "low",
        },

        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        text_format=ClusterLabel,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError(
            "No parsed structured output returned."
        )

    # Cluster ID는 입력값을 authoritative하게 유지
    result.cluster_id = int(
        cluster["cluster_id"]
    )

    return result.model_dump()


def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--cluster",
        type=int,
        help="Label one cluster only.",
    )

    mode.add_argument(
        "--all",
        action="store_true",
        help="Label all prepared clusters.",
    )

    args = parser.parse_args()

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI()

    clusters = load_input()

    if args.cluster is not None:

        clusters = [
            c
            for c in clusters
            if c["cluster_id"]
            == args.cluster
        ]

        if not clusters:
            raise ValueError(
                f"Cluster {args.cluster} "
                "not found in input."
            )

    existing = (
        load_existing_results()
    )

    existing_by_id = {
        int(x["cluster_id"]): x
        for x in existing
    }

    print(
        "Model:",
        MODEL
    )

    print(
        "Clusters requested:",
        len(clusters)
    )

    print()

    for index, cluster in enumerate(
        clusters,
        start=1,
    ):

        cluster_id = int(
            cluster["cluster_id"]
        )

        if (
            args.all
            and cluster_id
            in existing_by_id
        ):
            print(
                f"[SKIP] Cluster "
                f"{cluster_id}: "
                "already labeled"
            )
            continue

        print(
            f"[{index}/{len(clusters)}] "
            f"Labeling cluster "
            f"{cluster_id}..."
        )

        try:

            result = label_cluster(
                client,
                cluster,
            )

            existing_by_id[
                cluster_id
            ] = result

            save_results(
                list(
                    existing_by_id.values()
                )
            )

            print(
                f"[OK] "
                f"{cluster_id}: "
                f"{result['label']}"
            )

            print(
                f"     type="
                f"{result['theme_type']}"
                f" | coherence="
                f"{result['coherence']}"
                f" | evidence="
                f"{result['evidence_strength']}"
            )

        except Exception as exc:

            print(
                f"[FAIL] Cluster "
                f"{cluster_id}"
            )

            print(
                type(exc).__name__,
                ":",
                exc,
            )

            # 한 cluster 실패해도
            # 기존 결과는 이미 저장됨
            if args.cluster is not None:
                raise

        time.sleep(0.5)

    print()
    print(
        "Saved:",
        RESULT_JSON
    )

    print(
        "Saved:",
        RESULT_CSV
    )


if __name__ == "__main__":
    main()

