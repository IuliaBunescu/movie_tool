import streamlit as st
from src.clustering import (
    agglomerative_clustering,
    calculate_top_genres,
    k_prototypes_clustering,
    kmeans_clustering,
    merge_clusters_with_preprocessed_df,
    recommend_similar_movies,
)
from src.plots import (
    plot_cluster_comparison_subplots,
    plot_cluster_distribution_pie,
    plot_clusters_with_pca,
    plot_clusters_with_tsne,
)

ALGO_TO_FUNCTION_MAPPING = {
    "K-Prototypes Clustering": {
        "func": k_prototypes_clustering,
        "kwargs": {"categorical_columns": ["original_language"]},
    },
    "K-Means Clustering": {"func": kmeans_clustering, "kwargs": {}},
    "Agglomerative Clustering": {"func": agglomerative_clustering, "kwargs": {}},
}


@st.fragment
def clustering_visualization(algo_name, preprocessed_df, reference_df, original_df):
    """
    Runs the selected clustering algorithm and visualizes the resulting clusters.

    Parameters:
    - algo_name (str): Display name of the clustering algorithm (key in ALGO_TO_FUNCTION_MAPPING).
    - preprocessed_df (pd.DataFrame): Feature-rich DataFrame to perform clustering on.
    - reference_df (pd.DataFrame): A DataFrame of reference movies for similarity comparisons.
    - original_df (pd.DataFrame): The original movie DataFrame with metadata for merging.

    Displays:
    - Cluster summaries, visualizations, and recommendations in the Streamlit UI.
    """
    st.write("Using default hyperparameters (8 clusters).")

    # Retrieve the function and kwargs from the mapping
    algo_config = ALGO_TO_FUNCTION_MAPPING.get(algo_name)
    if not algo_config:
        st.error(f"No configuration found for clustering algorithm: {algo_name}")
        return

    clustering_func = algo_config["func"]
    clustering_kwargs = algo_config.get("kwargs", {})

    # Include default n_clusters
    clustering_kwargs["n_clusters"] = 8

    try:
        clustered_df = clustering_func(preprocessed_df, **clustering_kwargs)
    except Exception as e:
        st.error(f"Error running clustering algorithm '{algo_name}': {str(e)}")
        return

    # Merge results with original metadata
    clustered_with_metadata = merge_clusters_with_preprocessed_df(
        original_df, clustered_df
    )

    # Reference movie cluster join
    st.subheader("Reference Movie Data")
    reference_with_cluster = reference_df.merge(
        clustered_with_metadata[["tmdb_id", "cluster"]],
        on="tmdb_id",
        how="left",
    )
    st.dataframe(reference_with_cluster, use_container_width=True, hide_index=True)

    # Genre and distribution analysis
    genre_col, dist_col = st.columns(2)
    with genre_col:
        st.subheader("Top Cluster Genre")
        st.dataframe(
            calculate_top_genres(clustered_with_metadata),
            use_container_width=True,
            hide_index=True,
        )
    with dist_col:
        st.subheader("Cluster Distribution")
        st.plotly_chart(
            plot_cluster_distribution_pie(clustered_with_metadata),
            use_container_width=True,
        )

    # Cluster averages
    st.subheader("Cluster Numerical Feature Averages")
    st.plotly_chart(
        plot_cluster_comparison_subplots(clustered_with_metadata),
        use_container_width=True,
    )

    # # PCA Visualization
    # st.subheader("2D Cluster Visualization using PCA")
    # features = [
    #     col
    #     for col in clustered_df.columns
    #     if col not in ["tmdb_id", "title", "cluster"]
    # ]
    # st.plotly_chart(
    #     plot_clusters_with_pca(
    #         clustered_df,
    #         cluster_column="cluster",
    #         title_column="title",
    #         id_column="tmdb_id",
    #         features=features,
    #     ),
    #     use_container_width=True,
    # )

    # t-SNE Visualization
    st.subheader("2D Cluster Visualization using t-SNE")
    features = [
        col
        for col in clustered_df.columns
        if col not in ["tmdb_id", "title", "cluster"]
    ]
    st.plotly_chart(
        plot_clusters_with_tsne(
            clustered_df,
            cluster_column="cluster",
            title_column="title",
            id_column="tmdb_id",
            features=features,
        ),
        use_container_width=True,
    )

    # Top 10 Recommendations
    st.subheader("Top 10 Movie Recommendations")
    st.write(
        "The recommended movies belong to the same cluster as the reference movie and "
        "are ranked by their Euclidean distance."
    )
    top_recommendations_dist = recommend_similar_movies(
        clustered_df, reference_df, features
    )
    top_recommendations = clustered_with_metadata.merge(
        top_recommendations_dist[["tmdb_id", "distance_to_reference"]],
        on="tmdb_id",
        how="right",
    )

    st.dataframe(
        top_recommendations.drop(columns=["tmdb_id"]),
        column_config={
            "imdb_link": st.column_config.LinkColumn(
                display_text="https://www.imdb.com/title/(.*?)/"
            ),
            "tmdb_link": st.column_config.LinkColumn(
                display_text="https://www.themoviedb.org/movie/(\\d+)"
            ),
        },
        use_container_width=True,
    )

    # st.header("Decision Tree Clustering")
    # st.subheader("Self-Organizing Maps (SOM)")
