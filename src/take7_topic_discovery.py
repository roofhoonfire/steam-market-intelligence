import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]

REVIEWS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "reviews.parquet"
)

FEATURE_DIR = (
    ROOT
    / "data"
    / "features"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "take7"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def clean_text(text):
    """
    Embedding용 최소 cleaning.

    의미를 훼손하지 않고
    whitespace만 정리한다.
    """

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def build_topic_corpus(reviews):
    section(
        "1. BUILD TOPIC CORPUS"
    )

    df = reviews.copy()

    df["analysis_text"] = (
        df["review_text"]
        .apply(clean_text)
    )

    df["analysis_word_count"] = (
        df["analysis_text"]
        .str.split()
        .str.len()
    )

    # 단순 emoticon / "good game" 같은
    # topic 정보가 거의 없는 초단문 제외
    df["topic_eligible"] = (
        (df["analysis_word_count"] >= 5)
        &
        (df["analysis_text"].str.len() >= 20)
    )

    eligible = (
        df[
            df["topic_eligible"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "All reviews:",
        len(df)
    )

    print(
        "Topic eligible:",
        len(eligible),
        f"({len(eligible) / len(df):.1%})"
    )

    duplicate_count = (
        eligible["analysis_text"]
        .duplicated()
        .sum()
    )

    print(
        "Exact duplicate texts:",
        duplicate_count
    )

    long_count = (
        eligible[
            "analysis_word_count"
        ]
        .gt(200)
        .sum()
    )

    print(
        ">200-word reviews:",
        long_count,
        f"({long_count / len(eligible):.1%})"
    )

    corpus_path = (
        FEATURE_DIR
        / "topic_corpus.parquet"
    )

    eligible.to_parquet(
        corpus_path,
        index=False,
    )

    print(
        "Saved:",
        corpus_path
    )

    return eligible


def embed_reviews(df):
    section(
        "2. SENTENCE EMBEDDING"
    )

    print(
        "Model:",
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = (
        df["analysis_text"]
        .tolist()
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    embedding_path = (
        FEATURE_DIR
        / "embeddings.npy"
    )

    np.save(
        embedding_path,
        embeddings,
    )

    index = df[
        [
            "recommendation_id",
            "appid",
            "title",
            "group",
        ]
    ].copy()

    index["embedding_row"] = (
        np.arange(
            len(index)
        )
    )

    index_path = (
        FEATURE_DIR
        / "embedding_index.parquet"
    )

    index.to_parquet(
        index_path,
        index=False,
    )

    print(
        "Saved:",
        embedding_path
    )

    print(
        "Saved:",
        index_path
    )

    return embeddings


def reduce_for_clustering(
    embeddings
):
    section(
        "3. UMAP FOR CLUSTERING"
    )

    reducer = UMAP(
        n_neighbors=30,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    reduced = reducer.fit_transform(
        embeddings
    )

    reduced = reduced.astype(
        np.float32
    )

    path = (
        FEATURE_DIR
        / "umap_cluster_embedding.npy"
    )

    np.save(
        path,
        reduced,
    )

    print(
        "Cluster embedding shape:",
        reduced.shape
    )

    print(
        "Saved:",
        path
    )

    return reduced


def cluster_reviews(
    reduced
):
    section(
        "4. HDBSCAN"
    )

    clusterer = HDBSCAN(
        min_cluster_size=40,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    clusterer.fit(
        reduced
    )

    labels = (
        clusterer.labels_
    )

    probabilities = (
        clusterer.probabilities_
    )

    real_clusters = sorted(
        set(labels)
        - {-1, -2, -3}
    )

    noise_share = (
        np.mean(labels < 0)
    )

    print(
        "Clusters:",
        len(real_clusters)
    )

    print(
        "Noise share:",
        f"{noise_share:.1%}"
    )

    return (
        labels,
        probabilities,
    )


def reduce_for_visualization(
    embeddings
):
    section(
        "5. 2D UMAP"
    )

    reducer = UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    coords = reducer.fit_transform(
        embeddings
    )

    return coords


def build_cluster_dataset(
    df,
    labels,
    probabilities,
    coords,
):
    section(
        "6. BUILD CLUSTER DATASET"
    )

    result = df.copy()

    result["cluster_id"] = (
        labels
    )

    result["cluster_probability"] = (
        probabilities
    )

    result["umap_x"] = (
        coords[:, 0]
    )

    result["umap_y"] = (
        coords[:, 1]
    )

    output = (
        FEATURE_DIR
        / "review_topics_initial.parquet"
    )

    result.to_parquet(
        output,
        index=False,
    )

    print(
        "Saved:",
        output
    )

    return result


def cluster_summary(df):
    section(
        "7. CLUSTER SUMMARY"
    )

    clustered = df[
        df["cluster_id"] >= 0
    ]

    summary = (
        clustered
        .groupby("cluster_id")
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
            mean_probability=(
                "cluster_probability",
                "mean",
            ),
        )
        .sort_values(
            "reviews",
            ascending=False,
        )
    )

    print(
        summary
        .round(3)
        .to_string()
    )

    output = (
        OUTPUT_DIR
        / "cluster_summary.csv"
    )

    summary.to_csv(
        output
    )

    print()
    print(
        "Saved:",
        output
    )

    return summary


def export_representative_reviews(
    df
):
    section(
        "8. REPRESENTATIVE REVIEWS"
    )

    clustered = df[
        df["cluster_id"] >= 0
    ].copy()

    representatives = (
        clustered
        .sort_values(
            [
                "cluster_id",
                "cluster_probability",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "cluster_id"
        )
        .head(8)
    )

    columns = [
        "cluster_id",
        "cluster_probability",
        "title",
        "group",
        "voted_up",
        "playtime_at_review_hours",
        "analysis_text",
    ]

    output = (
        OUTPUT_DIR
        / "representative_reviews.csv"
    )

    representatives[
        columns
    ].to_csv(
        output,
        index=False,
    )

    print(
        "Representative reviews:",
        len(representatives)
    )

    print(
        "Saved:",
        output
    )


def plot_clusters(df):
    section(
        "9. CLUSTER MAP"
    )

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        df["umap_x"],
        df["umap_y"],
        c=df["cluster_id"],
        s=6,
        alpha=0.6,
    )

    plt.xlabel(
        "UMAP 1"
    )

    plt.ylabel(
        "UMAP 2"
    )

    plt.title(
        "Steam Review Semantic Clusters"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "review_cluster_map.png"
    )

    plt.savefig(
        output,
        dpi=180,
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

    corpus = build_topic_corpus(
        reviews
    )

    embeddings = embed_reviews(
        corpus
    )

    reduced = (
        reduce_for_clustering(
            embeddings
        )
    )

    labels, probabilities = (
        cluster_reviews(
            reduced
        )
    )

    coords = (
        reduce_for_visualization(
            embeddings
        )
    )

    result = (
        build_cluster_dataset(
            corpus,
            labels,
            probabilities,
            coords,
        )
    )

    cluster_summary(
        result
    )

    export_representative_reviews(
        result
    )

    plot_clusters(
        result
    )

    section(
        "TAKE 7 INITIAL TOPIC DISCOVERY COMPLETE"
    )


if __name__ == "__main__":
    main()
