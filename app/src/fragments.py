import pandas as pd
import streamlit as st
from src.clustering import (
    agglomerative_clustering,
    calculate_top_genres,
    evaluate_clustering,
    k_prototypes_clustering,
    kmeans_clustering,
    merge_with_preprocessed_df,
    optimal_agglomerative_clustering,
    optimal_analysis_kmeans,
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
        "kwargs": {"categorical_columns": ["original_language", "country_of_origin"]},
    },
    "K-Means Clustering": {
        "func": kmeans_clustering,
        "kwargs": {},
        "optimal_k_func": optimal_analysis_kmeans,
    },
    "Agglomerative Clustering": {
        "func": agglomerative_clustering,
        "kwargs": {},
        "optimal_k_func": optimal_agglomerative_clustering,
    },
}


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
    algo_config = ALGO_TO_FUNCTION_MAPPING.get(algo_name)

    clustering_func = algo_config["func"]
    clustering_kwargs = algo_config.get("kwargs", {}).copy()

    # Use optimal k finder if available
    optimal_k_func = algo_config.get("optimal_k_func")
    if optimal_k_func:
        try:
            best_k, *figs = optimal_k_func(preprocessed_df)

            # Check if optimal clusters are 2, then switch to 8 clusters
            if best_k == 2:
                st.info(
                    "Optimal number of clusters is 2, falling back to default: 8 clusters."
                )
                clustering_kwargs["n_clusters"] = 8
            else:
                clustering_kwargs["n_clusters"] = best_k

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Silhouette Score vs Number of Clusters")
                st.plotly_chart(figs[0], use_container_width=True)
            with col2:
                if algo_name == "Agglomerative Clustering":
                    st.subheader("Dendrogram (linkage='ward') - Last 50 Clusters")
                elif algo_name == "K-Means Clustering":
                    st.subheader(f"Silhouette Plot for Optimal k = {best_k}")
                st.plotly_chart(figs[1], use_container_width=True)

            st.success(
                f"Optimal number of clusters determined: {best_k}"
                if best_k != 2
                else "Using default: 8 clusters"
            )
        except Exception as e:
            st.warning(f"Could not determine optimal number of clusters: {str(e)}")
            clustering_kwargs["n_clusters"] = 8
            st.info("Falling back to default: 8 clusters.")
    else:
        st.success("Using default hyperparameters (8 clusters).")
        clustering_kwargs["n_clusters"] = 8

    # Run the clustering
    try:
        clustered_df = clustering_func(preprocessed_df, **clustering_kwargs)

        # Evaluate clustering quality (overall evaluation)
        numerical_features = [
            col
            for col in clustered_df.columns
            if pd.api.types.is_numeric_dtype(clustered_df[col])
            and col not in ["tmdb_id", "title", "cluster"]
        ]

        cluster_quality_scores = evaluate_clustering(clustered_df, numerical_features)

        # Initialize session_state DataFrame if it doesn't exist
        if "cluster_quality_df" not in st.session_state:
            st.session_state.cluster_quality_df = pd.DataFrame(
                columns=[
                    "algorithm",
                    "silhouette_score",
                    "davies_bouldin_index",
                    "calinski_harabasz_score",
                ]
            )

        # Add the overall quality scores as a new row
        new_row = {
            "algorithm": algo_name,
            "silhouette_score": cluster_quality_scores["silhouette_score"],
            "davies_bouldin_index": cluster_quality_scores["davies_bouldin_index"],
            "calinski_harabasz_score": cluster_quality_scores[
                "calinski_harabasz_score"
            ],
        }
        new_row_df = pd.DataFrame([new_row])

        st.session_state.cluster_quality_df = pd.concat(
            [st.session_state.cluster_quality_df, new_row_df], ignore_index=True
        )
    except Exception as e:
        st.error(f"Error running clustering algorithm '{algo_name}': {str(e)}")
        return

    # Merge results with original metadata
    clustered_with_metadata = merge_with_preprocessed_df(original_df, clustered_df)

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
    st.plotly_chart(
        plot_clusters_with_tsne(
            clustered_df,
            cluster_column="cluster",
            title_column="title",
            id_column="tmdb_id",
            features=numerical_features,
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
        clustered_df, reference_df, numerical_features
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
