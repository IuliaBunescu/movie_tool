import datetime
from collections import Counter

import pandas as pd
import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.metrics.pairwise import euclidean_distances


@st.cache_data(
    ttl=datetime.timedelta(hours=6), show_spinner="K-Prototypes Clustering..."
)
def k_prototypes_clustering(df, categorical_columns, n_clusters=8):
    """
    Applies K-Prototypes clustering algorithm to the data (both numerical and categorical features).

    Parameters:
    - df: The prepared DataFrame with both numerical and categorical features.
    - categorical_columns: List of original categorical feature names.
    - n_clusters: Number of clusters to create.

    Returns:
    - df: DataFrame with assigned cluster labels, including 'tmdb_id' and 'title'.
    """
    print(f"\n{'='*50}")
    print(f"Started K-Prototypes clustering with {n_clusters} clusters.")

    # Step 1: Keep 'tmdb_id' and 'title' columns for reference
    reference_columns = (
        df[["tmdb_id", "title"]]
        if "tmdb_id" in df.columns and "title" in df.columns
        else None
    )

    # Step 2: Store the original order of the rows
    original_index = df.index

    # Step 3: Drop 'tmdb_id' and 'title' columns for clustering
    if "tmdb_id" in df.columns:
        df = df.drop(columns=["tmdb_id"])
        print("- Dropped 'tmdb_id' column")

    if "title" in df.columns:
        df = df.drop(columns=["title"])
        print("- Dropped 'title' column")

    # Step 4: Identify the One-Hot Encoded categorical columns
    one_hot_columns = [
        col for col in df.columns if col.startswith(tuple(categorical_columns))
    ]

    # Step 5: Combine original categorical columns and One-Hot Encoded columns
    all_categorical_columns = one_hot_columns
    categorical_indexes = [df.columns.get_loc(col) for col in all_categorical_columns]

    # Step 6: Fit the K-Prototypes model
    kproto = KPrototypes(n_clusters=n_clusters, init="Cao", n_init=10, verbose=2)
    kproto.fit(df, categorical=categorical_indexes)

    # Step 7: Assign clusters to the dataframe
    df["cluster"] = kproto.labels_
    print(f"Cluster centers (centroids): \n{kproto.cluster_centroids_}")

    # Step 8: If reference columns exist, add them back to the dataframe while preserving the original order
    if reference_columns is not None:
        df = pd.concat([reference_columns, df], axis=1)
        df = df.set_index(original_index)  # Reset to original row order
        print("- Added 'tmdb_id' and 'title' back to the DataFrame.")

    print(f"\n{'='*50}")

    return df


def merge_clusters_with_preprocessed_df(preprocessed_df, clustered_df):
    """
    Merges the cluster labels from the clustered DataFrame into the pre-processed DataFrame,
    and drops rows without a cluster assignment.

    Parameters:
    - preprocessed_df: The original DataFrame with clear features.
    - clustered_df: The DataFrame after clustering with cluster labels.

    Returns:
    - merged_df: The pre-processed DataFrame with an added 'cluster' column and no missing clusters.
    """
    # Ensure both DataFrames have 'tmdb_id' column for merging
    if (
        "tmdb_id" not in preprocessed_df.columns
        or "tmdb_id" not in clustered_df.columns
    ):
        raise ValueError("Both DataFrames must have the 'tmdb_id' column for merging.")

    # Select only tmdb_id and cluster columns from the clustered DataFrame
    cluster_mapping = clustered_df[["tmdb_id", "cluster"]]

    # Merge the cluster labels into the preprocessed DataFrame
    merged_df = preprocessed_df.merge(cluster_mapping, on="tmdb_id", how="left")
    print("\nCluster column successfully merged into the preprocessed DataFrame.")

    # Drop rows without a cluster assignment
    before_drop = merged_df.shape[0]
    merged_df = merged_df.dropna(subset=["cluster"]).reset_index(drop=True)
    after_drop = merged_df.shape[0]
    print(f"Dropped {before_drop - after_drop} rows without a cluster assignment.")

    merged_df["cluster"] = merged_df["cluster"].astype(int)

    # Add release_year column extracted from release_date
    if "release_date" in merged_df.columns:
        merged_df["release_year"] = pd.to_datetime(merged_df["release_date"]).dt.year
        print("Extracted 'release_year' from 'release_date' column.")
    else:
        raise ValueError("'release_date' column not found in the merged DataFrame.")

    print(
        f"Merged DataFrame now has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns."
    )

    return merged_df


def calculate_top_genres(df):
    """
    Find the most common genres for each cluster.

    Parameters:
    - df: DataFrame containing 'genres' and 'cluster' columns.

    Returns:
    - DataFrame with the top genres for each cluster.
    """

    def most_common_genres(group):
        all_genres = ", ".join(group).split(", ")
        return Counter(all_genres).most_common(3)  # Top 3 genres

    top_genres = df.groupby("cluster")["genres"].apply(most_common_genres)

    # Prepare the data for a nice DataFrame
    data = []
    for cluster, genres in top_genres.items():
        for rank, (genre, count) in enumerate(genres, 1):
            data.append(
                {"Cluster": cluster, "Rank": rank, "Genre": genre, "Count": count}
            )

    # Convert to DataFrame
    genre_df = pd.DataFrame(data)

    # Pivot to get a nice table format with top 3 genres per cluster
    genre_df_pivot = genre_df.pivot_table(
        index="Cluster", columns="Rank", values=["Genre", "Count"], aggfunc="first"
    )

    # Flatten the multi-level columns for better readability
    genre_df_pivot.columns = [
        f"{metric} Rank {rank}" for metric, rank in genre_df_pivot.columns
    ]

    # Reset index to make the DataFrame more readable
    genre_df_pivot.reset_index(inplace=True)

    genre_df_pivot = genre_df_pivot[
        [
            "Cluster",
            "Count Rank 1",
            "Genre Rank 1",
            "Count Rank 2",
            "Genre Rank 2",
            "Count Rank 3",
            "Genre Rank 3",
        ]
    ]

    return genre_df_pivot


def recommend_similar_movies(df, df_reference, features, top_n=10):
    """
    Recommend top N movies based on cluster proximity to a reference movie using preprocessed features.

    Args:
        df (pd.DataFrame): DataFrame containing movie data with clusters and features.
        df_reference (pd.DataFrame): The reference dataframe with the reference movie.
        features (list): List of features used for comparison (exclude 'tmdb_id' and 'title').
        top_n (int): Number of top similar movies to recommend.

    Returns:
        pd.DataFrame: DataFrame with top N similar movies and their proximity scores.
    """

    # Step 1: Get the tmdb_id of the reference movie from df_reference
    reference_tmdb_id = df_reference["tmdb_id"].iloc[0]  # or use any specific row

    # Step 2: Get the cluster of the reference movie
    reference_cluster = df[df["tmdb_id"] == reference_tmdb_id]["cluster"].values[0]

    # Step 3: Filter all movies in the same cluster
    df_cluster = df[df["cluster"] == reference_cluster]

    # Step 4: Extract the feature vectors for the reference movie and the other movies in the same cluster
    reference_features = df[df["tmdb_id"] == reference_tmdb_id][features].values

    # Step 5: Calculate Euclidean distance between the reference movie and all movies in the same cluster
    distances = euclidean_distances(reference_features, df_cluster[features]).flatten()

    # Step 6: Add the distance to the dataframe
    df_cluster["distance_to_reference"] = distances

    # Step 7: Sort the movies by proximity (distance) and select top N closest movies
    top_recommendations = df_cluster.sort_values(by="distance_to_reference").head(top_n)

    return top_recommendations[["tmdb_id", "title", "distance_to_reference"]]
