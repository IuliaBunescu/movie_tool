import datetime
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from kmodes.kprototypes import KPrototypes
from matplotlib import cm
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances


@st.cache_data(
    ttl=datetime.timedelta(hours=12),
    show_spinner="K-Prototypes Clustering (can take up to 3 minutes)...",
)
def k_prototypes_clustering(
    input_df, categorical_columns, n_clusters=8, id_column="tmdb_id"
):
    """
    Applies K-Prototypes clustering algorithm to the data (both numerical and categorical features).

    Parameters:
    - input_df: The prepared DataFrame with both numerical and categorical features.
    - categorical_columns: List of original categorical feature names.
    - n_clusters: Number of clusters to create.
    - id_column: Name of the column to use as the unique identifier (default is "tmdb_id").

    Returns:
    - df: DataFrame with assigned cluster labels, including the ID column and 'title' if present.
    """
    df = input_df.copy()

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

    categorical_indexes = [df.columns.get_loc(col) for col in categorical_columns]

    # Step 4: Fit the K-Prototypes model
    kproto = KPrototypes(n_clusters=n_clusters, init="Cao", n_init=10, verbose=2)
    kproto.fit(df, categorical=categorical_indexes)

    # Step 5: Assign clusters to the dataframe
    df["cluster"] = kproto.labels_
    print(f"Cluster centers (centroids): \n{kproto.cluster_centroids_}")

    # Step 6: If reference columns exist, add them back to the dataframe while preserving the original order
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
def agglomerative_clustering(input_df, n_clusters=8):
    """
    Applies Agglomerative Clustering to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame including features (must be numeric).
    - n_clusters (int): Number of clusters to form.

    Returns:
    - df: Original DataFrame with added 'cluster' column.
    """
    df = input_df.copy()

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

    # Step 3: Apply Agglomerative Clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(df_features)

    # Step 4: Add cluster labels back to DataFrame
    df["cluster"] = cluster_labels
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


@st.cache_data(
    ttl=datetime.timedelta(hours=12),
    show_spinner="K-Means Clustering...",
)
def kmeans_clustering(input_df, n_clusters=8):
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
    df = input_df.copy()

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
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
    cluster_labels = clusterer.fit_predict(df_features)

    # Step 4: Add cluster labels back to DataFrame
    df["cluster"] = cluster_labels
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


def merge_with_preprocessed_df(
    original_df, preprocessed_df, id_column="tmdb_id", expect_cluster=True
):
    """
    Merges data (cluster labels or similarity scores) from a preprocessed DataFrame into the original DataFrame.

    Parameters:
    - original_df (pd.DataFrame): The original DataFrame before clustering or scoring.
    - preprocessed_df (pd.DataFrame): The DataFrame containing either cluster labels or similarity scores.
    - id_column (str): The column name used to merge both DataFrames (default: 'tmdb_id').
    - expect_cluster (bool): Whether to expect and process 'cluster' labels (default: True).
                              If False, assumes it's for similarity only.

    Returns:
    - merged_df (pd.DataFrame): The cleaned and merged DataFrame.
    """
    # Drop 'cleaned_overview' if present
    if "cleaned_overview" in original_df.columns:
        original_df = original_df.drop(columns=["cleaned_overview"])
        print("- Dropped 'cleaned_overview' column from the original DataFrame.")

    # Ensure both DataFrames have the ID column
    if id_column not in original_df.columns or id_column not in preprocessed_df.columns:
        raise ValueError(
            f"Both DataFrames must contain the '{id_column}' column for merging."
        )

    # Prepare columns to merge
    columns_to_merge = [id_column]

    if expect_cluster and "cluster" in preprocessed_df.columns:
        columns_to_merge.append("cluster")
    if "similarity_to_reference" in preprocessed_df.columns:
        columns_to_merge.append("similarity_to_reference")

    # Select only the required columns
    preprocessed_mapping = preprocessed_df[columns_to_merge]

    # Merge
    merged_df = original_df.merge(preprocessed_mapping, on=id_column, how="left")
    print(f"- Merged preprocessed data using '{id_column}'.")

    # Drop rows without a cluster assignment (only if cluster expected)
    if expect_cluster and "cluster" in merged_df.columns:
        before_drop = merged_df.shape[0]
        merged_df = merged_df.dropna(subset=["cluster"]).reset_index(drop=True)
        after_drop = merged_df.shape[0]
        print(f"- Dropped {before_drop - after_drop} rows without cluster assignment.")

        # Convert cluster to integer
        merged_df["cluster"] = merged_df["cluster"].astype(int)

    # Drop 'release_date' and 'release_year' if present
    for col in ["release_date", "release_year"]:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])
            print(f"- Dropped '{col}' column.")

    # Create IMDb links
    if "imdb_id" in merged_df.columns:
        merged_df["imdb_link"] = merged_df["imdb_id"].apply(
            lambda x: f"https://www.imdb.com/title/{x}/" if pd.notna(x) else None
        )
        print("- Created IMDb links.")

        merged_df = merged_df.drop(columns=["imdb_id"])
        print("- Dropped 'imdb_id' column.")

    # Create TMDb links
    if "tmdb_id" in merged_df.columns:
        merged_df["tmdb_link"] = merged_df["tmdb_id"].apply(
            lambda x: f"https://www.themoviedb.org/movie/{x}" if pd.notna(x) else None
        )
        print("- Created TMDb links.")

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


def optimal_analysis_kmeans(df, k_range=range(2, 11), random_state=42):
    """
    Determines the optimal number of clusters (k) using silhouette score,
    plots silhouette scores over k_range (Plotly), and generates a silhouette plot
    for the optimal k using Plotly.

    Parameters:
    - df (pd.DataFrame): PCA-reduced DataFrame that includes 'tmdb_id' and 'title'.
    - k_range (iterable): Range of k values to test.
    - random_state (int): Seed for reproducibility.
    """
    # Drop non-numeric reference columns
    feature_df = df.drop(columns=["tmdb_id", "title"], errors="ignore")

    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        cluster_labels = kmeans.fit_predict(feature_df)
        score = silhouette_score(feature_df, cluster_labels)
        silhouette_scores.append(score)

    # Plot silhouette scores vs. k using Plotly
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode="lines+markers",
            line=dict(color="royalblue"),
            marker=dict(size=8),
            name="Silhouette Score",
        )
    )
    fig_line.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        template="plotly_white",
    )

    # Determine best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best k: {best_k} with silhouette score: {max(silhouette_scores):.4f}")

    # Generate silhouette plot for best k using Plotly
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(feature_df)

    silhouette_avg = silhouette_score(feature_df, cluster_labels)
    sample_silhouette_values = silhouette_samples(feature_df, cluster_labels)

    colors = px.colors.sequential.Agsunset
    sil_plot_data = []
    y_lower = 0
    for i in range(best_k):
        cluster_vals = sample_silhouette_values[cluster_labels == i]
        cluster_vals.sort()
        size_cluster_i = cluster_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i % len(colors)]  # Cycle if clusters > color palette

        sil_plot_data.append(
            go.Bar(
                x=cluster_vals,
                y=list(range(y_lower, y_upper)),
                orientation="h",
                name=f"Cluster {i}",
                marker_color=color,
                hoverinfo="x+y",
            )
        )

        y_lower = y_upper + 20  # add space between clusters

    fig_silhouette = go.Figure(data=sil_plot_data)
    fig_silhouette.update_layout(
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Sample Index",
        showlegend=True,
        template="plotly_white",
        shapes=[
            dict(
                type="line",
                x0=silhouette_avg,
                x1=silhouette_avg,
                y0=0,
                y1=y_lower,
                line=dict(color="red", dash="dash"),
            )
        ],
    )

    return best_k, fig_line, fig_silhouette


def optimal_agglomerative_clustering(df, k_range=range(2, 11), linkage_method="ward"):
    """
    Determines the optimal number of clusters for Agglomerative Clustering using silhouette score.
    Then plots the silhouette scores and a dendrogram for visual inspection.

    Parameters:
    - df (pd.DataFrame): The data to cluster (excluding non-numeric metadata).
    - k_range (iterable): Range of k values to test.
    - linkage_method (str): Linkage method to use ('ward', 'complete', 'average', 'single').

    Returns:
    - best_k (int): Optimal number of clusters.
    - silhouette_fig (plotly.graph_objects.Figure): Silhouette score plot.
    - dendrogram_fig (plotly.graph_objects.Figure): Colored dendrogram plot.
    """
    # Drop non-numeric columns
    feature_df = df.drop(columns=["tmdb_id", "title"], errors="ignore")

    silhouette_scores = []

    # Compute silhouette scores for each k
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(feature_df)
        score = silhouette_score(feature_df, labels)
        silhouette_scores.append(score)

    # Plot silhouette score vs k using Plotly
    silhouette_fig = go.Figure()
    silhouette_fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode="lines+markers",
            marker=dict(size=8, color="darkorange"),
            name="Silhouette Score",
        )
    )
    silhouette_fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        template="plotly_white",
    )

    # Choose best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best k: {best_k} with silhouette score: {max(silhouette_scores):.4f}")

    # Generate linkage matrix
    linked = linkage(feature_df, method=linkage_method)

    # Parameters for truncation and coloring
    p = 50
    cutoff_dist = linked[-(best_k - 1), 2]

    # Generate truncated dendrogram
    dendro = dendrogram(
        linked,
        truncate_mode="lastp",
        p=p,
        no_plot=True,
        show_contracted=True,
        color_threshold=cutoff_dist,
    )

    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    labels = dendro["ivl"]
    colors = dendro["color_list"]

    # Create consistent color mapping
    unique_colors = list(set(colors))
    colormap = cm.get_cmap("tab20", len(unique_colors))
    color_map_dict = {uc: to_hex(colormap(i)) for i, uc in enumerate(unique_colors)}

    # Build dendrogram traces
    traces = []
    for i in range(len(icoord)):
        x = icoord[i]
        y = dcoord[i]
        color = color_map_dict[colors[i]]
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )

    # Create dendrogram figure
    dendrogram_fig = go.Figure(data=traces)

    dendrogram_fig.add_shape(
        type="line",
        x0=0,
        x1=10 * p,
        y0=cutoff_dist,
        y1=cutoff_dist,
        line=dict(color="red", width=2, dash="dash"),
    )

    dendrogram_fig.update_layout(
        xaxis=dict(
            tickvals=[(i * 10 + 5) for i in range(len(labels))],
            ticktext=labels,
            title="Cluster (size or index)",
        ),
        yaxis_title="Distance",
        width=1200,
        height=600,
    )

    return best_k, silhouette_fig, dendrogram_fig


def evaluate_clustering(df, feature_cols, label_col="cluster"):
    """
    Evaluates clustering quality using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and cluster labels.
        feature_cols (list): List of column names to use as features.
        label_col (str): Column name containing the cluster labels.

    Returns:
        dict: Dictionary with the three metric scores.
    """
    X = df[feature_cols].values
    labels = df[label_col].values

    scores = {}

    if len(set(labels)) > 1 and len(set(labels)) < len(
        X
    ):  # Must have at least 2 clusters, not all unique
        scores["silhouette_score"] = silhouette_score(X, labels)
        scores["davies_bouldin_index"] = davies_bouldin_score(X, labels)
        scores["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
    else:
        scores["silhouette_score"] = None
        scores["davies_bouldin_index"] = None
        scores["calinski_harabasz_score"] = None
        print("Not enough clusters for evaluation metrics.")

    return scores
