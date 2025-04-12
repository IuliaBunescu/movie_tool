import datetime
from collections import Counter

import pandas as pd
import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances


@st.cache_data(
    ttl=datetime.timedelta(hours=12),
    show_spinner="K-Prototypes Clustering (can take up to 3 minutes)...",
)
def k_prototypes_clustering(df, categorical_columns, n_clusters=8, id_column="tmdb_id"):
    """
    Applies K-Prototypes clustering algorithm to the data (both numerical and categorical features).

    Parameters:
    - df: The prepared DataFrame with both numerical and categorical features.
    - categorical_columns: List of original categorical feature names.
    - n_clusters: Number of clusters to create.
    - id_column: Name of the column to use as the unique identifier (default is "tmdb_id").

    Returns:
    - df: DataFrame with assigned cluster labels, including the ID column and 'title' if present.
    """
    print(f"\n{'='*50}")
    print(f"Started K-Prototypes clustering with {n_clusters} clusters.")

    # Step 1: Keep the ID column and 'title' column for reference
    reference_columns = (
        df[[id_column, "title"]]
        if id_column in df.columns and "title" in df.columns
        else None
    )

    # Step 2: Store the original order of the rows
    original_index = df.index

    # Step 3: Drop the ID column and 'title' column for clustering
    if id_column in df.columns:
        df = df.drop(columns=[id_column])
        print(f"- Dropped '{id_column}' column")

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
    kproto = KPrototypes(n_clusters=n_clusters, init="Cao", n_init=5, verbose=2)
    kproto.fit(df, categorical=categorical_indexes)

    # Step 7: Assign clusters to the dataframe
    df["cluster"] = kproto.labels_
    print(f"Cluster centers (centroids): \n{kproto.cluster_centroids_}")

    # Step 8: If reference columns exist, add them back to the dataframe while preserving the original order
    if reference_columns is not None:
        df = pd.concat([reference_columns, df], axis=1)
        df = df.set_index(original_index)  # Reset to original row order
        print(f"- Added '{id_column}' and 'title' back to the DataFrame.")

    print(f"\n{'='*50}")

    return df


@st.cache_data(
    ttl=datetime.timedelta(hours=12),
    show_spinner="Agglomerative Clustering ...",
)
def agglomerative_clustering(df, n_clusters=8):
    """
    Applies Agglomerative Clustering to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame including features (must be numeric).
    - n_clusters (int): Number of clusters to form.

    Returns:
    - df: Original DataFrame with added 'cluster' column.
    """
    print(f"\n{'='*50}")
    print(f"Started Agglomerative Clustering (n_clusters={n_clusters}).")

    # Step 1: Keep reference columns if available
    reference_columns = (
        df[["tmdb_id", "title"]]
        if "tmdb_id" in df.columns and "title" in df.columns
        else None
    )
    original_index = df.index

    # Step 2: Drop non-feature columns
    df_features = df.drop(columns=["tmdb_id", "title"], errors="ignore")

    print(f"- Shape of feature set: {df_features.shape}")

    # Step 3: Drop columns with NaN values
    df_features_cleaned = df_features.dropna(axis=1)  # Drops columns with NaN values

    print(
        f"- Shape of feature set after dropping columns with NaN values: {df_features_cleaned.shape}"
    )

    # Step 4: Apply Agglomerative Clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(df_features_cleaned)

    # Step 5: Add cluster labels back to DataFrame
    df["cluster"] = pd.Series(cluster_labels).astype(str)
    print(f"- Assigned clusters. Unique clusters found: {len(set(cluster_labels))}")

    # Step 6: Add back reference columns if they were dropped
    if reference_columns is not None:
        df = pd.concat(
            [reference_columns, df.drop(columns=["tmdb_id", "title"], errors="ignore")],
            axis=1,
        )
        df = df.set_index(original_index)

    print(f"{'='*50}\n")
    return df


@st.cache_data(
    ttl=datetime.timedelta(hours=12),
    show_spinner="K-Means Clustering...",
)
def kmeans_clustering(df, n_clusters=8):
    """
    Applies KMeans Clustering to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame including features (must be numeric).
    - n_clusters (int): Number of clusters to form.

    Returns:
    - df: Original DataFrame with added 'cluster' column.
    """
    print(f"\n{'='*50}")
    print(f"Started KMeans Clustering (n_clusters={n_clusters}).")

    # Step 1: Keep reference columns if available
    reference_columns = (
        df[["tmdb_id", "title"]]
        if "tmdb_id" in df.columns and "title" in df.columns
        else None
    )
    original_index = df.index

    # Step 2: Drop non-feature columns
    df_features = df.drop(columns=["tmdb_id", "title"], errors="ignore")

    print(f"- Shape of feature set: {df_features.shape}")

    # Step 3: Apply KMeans Clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000, n_init=5)
    cluster_labels = clusterer.fit_predict(df_features)

    # Step 4: Add cluster labels back to DataFrame
    df["cluster"] = pd.Series(cluster_labels).astype(str)
    print(f"- Assigned clusters. Unique clusters found: {len(set(cluster_labels))}")

    # Step 5: Add back reference columns if they were dropped
    if reference_columns is not None:
        df = pd.concat(
            [reference_columns, df.drop(columns=["tmdb_id", "title"], errors="ignore")],
            axis=1,
        )
        df = df.set_index(original_index)

    print(f"{'='*50}\n")

    return df


def merge_clusters_with_preprocessed_df(original_df, clustered_df, id_column="tmdb_id"):
    """
    Merges the cluster labels from the clustered DataFrame into the original DataFrame,
    drops unnecessary columns, and removes rows without a cluster assignment. Adds an IMDb link.

    Parameters:
    - original_df (pd.DataFrame): The original DataFrame before clustering.
    - clustered_df (pd.DataFrame): The DataFrame containing cluster labels.
    - id_column (str): The column name used to merge both DataFrames (default: 'tmdb_id').

    Returns:
    - merged_df (pd.DataFrame): The cleaned and merged DataFrame with a 'cluster' column and IMDb links.
    """
    # Drop 'cleaned_overview' if present
    if "cleaned_overview" in original_df.columns:
        original_df = original_df.drop(columns=["cleaned_overview"])
        print("- Dropped 'cleaned_overview' column from the original DataFrame.")

    # Ensure both DataFrames have the ID column
    if id_column not in original_df.columns or id_column not in clustered_df.columns:
        raise ValueError(
            f"Both DataFrames must contain the '{id_column}' column for merging."
        )

    # Select only the ID and cluster columns
    cluster_mapping = clustered_df[[id_column, "cluster"]]

    # Merge cluster labels
    merged_df = original_df.merge(cluster_mapping, on=id_column, how="left")
    print(f"- Cluster labels successfully merged using '{id_column}'.")

    # Drop rows without a cluster assignment
    before_drop = merged_df.shape[0]
    merged_df = merged_df.dropna(subset=["cluster"]).reset_index(drop=True)
    after_drop = merged_df.shape[0]
    print(f"- Dropped {before_drop - after_drop} rows without cluster assignment.")

    # Convert cluster to integer type
    merged_df["cluster"] = merged_df["cluster"].astype(int)

    # Drop 'release_date' and 'release_year', keeping only 'year'
    if "release_date" in merged_df.columns:
        merged_df = merged_df.drop(columns=["release_date"])
        print("- Dropped 'release_date' column.")

    if "release_year" in merged_df.columns:
        merged_df = merged_df.drop(columns=["release_year"])
        print("- Dropped 'release_year' column.")

    # Create IMDb links using 'imdb_id' column
    if "imdb_id" in merged_df.columns:
        merged_df["imdb_link"] = merged_df["imdb_id"].apply(
            lambda x: f"https://www.imdb.com/title/{x}/" if pd.notna(x) else None
        )
        print("- Created IMDb links using 'imdb_id'.")

        merged_df = merged_df.drop(columns=["imdb_id"])
        print("- Dropped 'imdb_id' column.")

    # Create TMDb links using 'tmdb_id' column
    if "tmdb_id" in merged_df.columns:
        merged_df["tmdb_link"] = merged_df["tmdb_id"].apply(
            lambda x: f"https://www.themoviedb.org/movie/{x}" if pd.notna(x) else None
        )
        print("- Created TMDb links using 'tmdb_id'.")

    print(
        f"- Final merged DataFrame shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns."
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


def recommend_similar_movies(df, df_reference, features, id_column="tmdb_id", top_n=10):
    """
    Recommend top N movies based on cluster proximity to a reference movie using preprocessed features.

    Args:
        df (pd.DataFrame): DataFrame containing movie data with clusters and features.
        df_reference (pd.DataFrame): The reference dataframe with the reference movie.
        features (list): List of features used for comparison (exclude 'id_column' and 'title').
        id_column (str): The column name that holds the unique identifier (e.g., 'tmdb_id' or 'imdb_id').
        top_n (int): Number of top similar movies to recommend.

    Returns:
        pd.DataFrame: DataFrame with top N similar movies and their proximity scores.
    """

    # Step 1: Get the reference movie's ID from df_reference
    reference_id = df_reference[id_column].iloc[0]  # or use any specific row

    # Step 2: Get the cluster of the reference movie
    reference_cluster = df[df[id_column] == reference_id]["cluster"].values[0]

    # Step 3: Filter all movies in the same cluster
    df_cluster = df[df["cluster"] == reference_cluster]

    # Step 4: Extract the feature vectors for the reference movie and the other movies in the same cluster
    reference_features = df[df[id_column] == reference_id][features].values

    # Step 5: Calculate Euclidean distance between the reference movie and all movies in the same cluster
    distances = euclidean_distances(reference_features, df_cluster[features]).flatten()

    # Step 6: Add the distance to the dataframe
    df_cluster["distance_to_reference"] = distances

    # Step 7: Sort the movies by proximity (distance) and select top N closest movies
    top_recommendations = df_cluster.sort_values(by="distance_to_reference").head(
        top_n + 1
    )

    return top_recommendations[[id_column, "title", "distance_to_reference"]]
