import pandas as pd
import streamlit as st
from src.clustering import (
    calculate_top_genres,
    k_prototypes_clustering,
    merge_clusters_with_preprocessed_df,
    recommend_similar_movies,
)
from src.components.custom_html import (
    CUSTOM_ALERT_ERROR,
    CUSTOM_ALERT_SUCCESS,
    CUSTOM_BARS_TAB,
    CUSTOM_FORM,
    CUSTOM_METRIC,
)
from src.helper import get_mean_values, get_timestamp, process_uploaded_files
from src.local import (
    load_local_db,
    prepare_local_data_for_clustering,
    sample_local_df,
    search_movie_in_local_db,
)
from src.plots import (
    plot_categorical_column_percentages,
    plot_cluster_comparison_subplots,
    plot_cluster_distribution_pie,
    plot_clusters_with_pca,
    plot_column_distribution_pie,
    plot_country_counts,
    plot_movies_by_year,
    plot_popularity_vs_vote,
)
from src.scraper import get_reference_from_url
from src.tmdb import (
    get_movies_by_genre_from_reference_df,
    prepare_tmdb_data_for_clustering,
    search_first_movie_by_title_and_year_tmdb,
)

st.set_page_config(layout="wide")


def submit_form():
    st.session_state.data_submitted = True


def main():
    st.title("Movie Recommendation Tool for Data Scientists 🎬")

    col1, col2 = st.columns([2.5, 7.5])

    with col1:
        st.header("Choose your data source", divider="gray")

        st.markdown(
            "Choose your data input method by providing a **Reference Movie** and selecting one or more **Supporting Data Sources**. After adding all the required inputs, click the **Submit** button. Once submitted, proceed to the **Results** section to view your output."
        )
        st.markdown(
            "**Recommendations**:\n"
            "1. When using *TMDB* reference data, ensure that you also use *TMDB* supporting data.\n"
            "2. When providing an IMDb link, use data from the local database.\n"
            "3. If you choose to provide the supporting data yourself, make sure it includes all the necessary features corresponding to the reference data you selected."
        )

        with st.form("input_form", enter_to_submit=False):
            st.write(CUSTOM_FORM, unsafe_allow_html=True)

            if "data_submitted" not in st.session_state:
                st.session_state.data_submitted = False

            st.subheader("Reference Movie")

            st.write("**Option 1** : Search our local database for a movie.")

            ds1, ds2 = st.columns([3, 1])
            with ds1:
                movie_name_local_ref = st.text_input(
                    "Movie name", key="movie_name_input_local"
                )

            with ds2:
                movie_year_local_ref = st.text_input(
                    "Movie release year", key="movie_year_input_local"
                )
                movie_year_local_ref = (
                    int(movie_year_local_ref)
                    if movie_year_local_ref.isdigit()
                    else None
                )

            # Initialize the variable for movie reference in local DB
            movie_ref_local = False

            # Check if the entered movie name exists in local db
            if movie_name_local_ref:
                local_db = load_local_db("..\data\imdb_movies.csv")

                local_ref_dict = search_movie_in_local_db(
                    local_db, movie_name_local_ref, movie_year_local_ref
                )

                # If there are matching rows, use the first match
                if local_ref_dict.get("found_movie_data_flag"):
                    movie_ref_local = True
                    ref_movie_df = local_ref_dict.get("movie_data")
                    st.success(
                        f"Data for '{movie_name_local_ref}' found in local database."
                    )
                    st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                else:
                    st.error(
                        f"Reference movie '{movie_name_local_ref}' data not found in local database. Try other reference input options."
                    )
                    st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            st.write("**Option 2**: Search reference using TMDB API.")
            movie_ref_tmdb = False
            ds3, ds4 = st.columns([3, 1])

            with ds3:
                movie_name_tmdb_ref = st.text_input(
                    "Movie name", key="movie_name_input_tmdb"
                )

            with ds4:
                movie_year_tmdb_ref = st.text_input(
                    "Movie release year", key="movie_year_input_tmdb"
                )
                movie_year_tmdb_ref = (
                    int(movie_year_tmdb_ref) if movie_year_tmdb_ref.isdigit() else None
                )
            if movie_name_tmdb_ref:
                ref_movie_df = search_first_movie_by_title_and_year_tmdb(
                    movie_name_tmdb_ref, movie_year_tmdb_ref
                )
                if not ref_movie_df.empty:
                    if st.session_state.data_submitted:
                        st.success(
                            f"Data for '{movie_name_tmdb_ref}' found using TMDB API."
                        )
                        st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                        movie_ref_tmdb = True
                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the TMDB reference movies DataFrame:"
                        )
                        ref_movie_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")
                    elif st.session_state.data_submitted:
                        st.error(
                            f"Reference movie '{movie_name_tmdb_ref}' data not found using TMDB API . Try other reference input options."
                        )
                        st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            website_for_scraping = st.text_input("**Option 3**: IMDB link ")
            movie_ref_url = False

            if website_for_scraping and st.session_state.data_submitted:
                url_ref_movie_dict = get_reference_from_url(website_for_scraping)
                if url_ref_movie_dict.get("found_movie_data_flag"):
                    st.success("Your website is valid.")
                    st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                    movie_ref_url = True
                    ref_movie_df = url_ref_movie_dict.get("ref_movie_df")
                elif st.session_state.data_submitted:
                    st.error(
                        "Did not find movie data in reference URL. Try other reference input options."
                    )
                    st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            st.subheader("Supporting Data")
            use_tmdb_api = False
            use_local_db = False
            reference_data_source = st.radio(
                "Chose one of the following sources:",
                ["TMDB API data", "Local db data", "None"],
                horizontal=False,
                key="reference_data_input",
            )

            if reference_data_source == "TMDB API data":
                use_tmdb_api = True
            elif reference_data_source == "Local db data":
                use_local_db = True
                local_db = load_local_db("..\data\imdb_movies.csv")

            uploaded_files = st.file_uploader(
                "Add your own CSV/TSV files",
                accept_multiple_files=True,
                key="input_tabular_file",
            )

            add_external_local_df = False
            add_external_tmdb_df = False

            if uploaded_files and st.session_state.data_submitted:

                external_data_dict = process_uploaded_files(uploaded_files)
                external_df = external_data_dict.get("dataframe")
                df_type = external_data_dict.get("type_flag")

                if df_type == "local":
                    add_external_local_df = True
                elif df_type == "tmdb":
                    add_external_tmdb_df = True

            st.form_submit_button(
                "Submit", use_container_width=True, on_click=submit_form
            )
        if st.session_state.data_submitted:
            st.write(f"*Your input data was submitted!*")
        else:
            st.write("Please fill out the input form.")

    with col2:
        st.header("Results", divider="gray")

        tab1, tab2 = st.tabs(
            ["Exploratory Visualization of Data", "Clustering & Recommendations"]
        )
        st.write(CUSTOM_BARS_TAB, unsafe_allow_html=True)

        with tab1:
            if (
                (
                    use_tmdb_api
                    or use_local_db
                    or add_external_local_df
                    or add_external_tmdb_df
                )
                and (movie_ref_url or movie_ref_local or movie_ref_tmdb)
                and st.session_state.data_submitted
            ):

                st.subheader("Reference Movie Data")
                st.dataframe(ref_movie_df, use_container_width=True, hide_index=True)

                if use_tmdb_api or add_external_tmdb_df:
                    if use_tmdb_api:
                        tmdb_movies_df = get_movies_by_genre_from_reference_df(
                            ref_movie_df
                        )
                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                        )
                        tmdb_movies_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")

                        if add_external_tmdb_df:
                            tmdb_movies_df = pd.concat(
                                [tmdb_movies_df, external_df],
                                ignore_index=True,
                            )
                            st.write(" ")
                            st.write(
                                "The data is being sourced from *TMDB* and external files. From TMDB the initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                            )

                            print(f"\n{'='*50}")
                            print(
                                f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                            )
                            tmdb_movies_df.info(
                                verbose=True,
                                buf=None,
                                max_cols=None,
                                memory_usage="deep",
                            )
                            print(f"{'='*50}\n")

                        else:
                            st.write(" ")
                            st.write(
                                "The data is being sourced from *TMDB*. The initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                            )

                    elif add_external_tmdb_df:
                        tmdb_movies_df = external_df
                        st.write("The data is being sourced from the attached files.")

                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                        )
                        tmdb_movies_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")

                    st.subheader("Metrics")
                    mean_metrics = get_mean_values(tmdb_movies_df)
                    m1, m2, m3, m4 = st.columns(4, vertical_alignment="center")
                    m1.metric(
                        "Average Rating",
                        f"{mean_metrics.iloc[1]}#",
                        border=True,
                        help="Average rating for the movies in the dataset.",
                    )
                    m2.metric(
                        "Average Vote Count",
                        f"{mean_metrics.iloc[2]}#",
                        border=True,
                        help="Average vote count for the movies in the dataset.",
                    )
                    m3.metric(
                        "Average Popularity",
                        f"{mean_metrics.iloc[3]}#",
                        border=True,
                        help="Average popularity score for the movies in the dataset.",
                    )
                    m4.metric(
                        "% of IMDB movies",
                        f"{(tmdb_movies_df['imdb_id'].notnull().sum() / len(tmdb_movies_df) * 100):.2f} %",
                        border=True,
                        help="Percentage of movies that have IMDB id.",
                    )
                    st.write(CUSTOM_METRIC, unsafe_allow_html=True)
                    genre_col, lang_col = st.columns(2)
                    with genre_col:
                        st.subheader("Genre Distribution")
                        st.plotly_chart(
                            plot_categorical_column_percentages(
                                tmdb_movies_df, "genres"
                            ),
                            use_container_width=True,
                        )
                    with lang_col:
                        st.subheader("Language Distribution")
                        st.plotly_chart(
                            plot_column_distribution_pie(
                                tmdb_movies_df, "original_language", cap_percentage=2
                            ),
                            use_container_width=True,
                        )

                    st.subheader("Country Distribution")
                    map_col, pie_col = st.columns(2, vertical_alignment="center")
                    with map_col:
                        st.plotly_chart(
                            plot_country_counts(
                                tmdb_movies_df,
                                "country_of_origin",
                                "#a6edcd",
                                "#ffb0b1",
                            ),
                            use_container_width=True,
                        )
                    with pie_col:
                        st.plotly_chart(
                            plot_column_distribution_pie(
                                tmdb_movies_df, "country_of_origin"
                            ),
                            use_container_width=True,
                        )

                    yearly_dist_col, pop_vs_avg_col = st.columns(2)
                    with yearly_dist_col:
                        st.subheader("Yearly Distribution")
                        st.plotly_chart(
                            plot_movies_by_year(tmdb_movies_df, "release_date"),
                            use_container_width=True,
                        )
                    with pop_vs_avg_col:
                        st.subheader("TMDB Popularity Score VS Average Rating")
                        st.plotly_chart(
                            plot_popularity_vs_vote(tmdb_movies_df),
                            use_container_width=True,
                        )
                # elif use_local_db:
                #     st.write("")
                #     st.write(
                #         "The data is being sourced from local database (IMDB). The initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                #     )

                #     sampled_local_df = sample_local_df(
                #         local_db, ref_movie_df["genres"].iloc[0].split(",")
                #     )
                #     sampled_local_df = pd.concat(
                #         [sampled_local_df, ref_movie_df], axis=0, ignore_index=True
                #     )
                #     print(ref_movie_df.info())

                elif use_local_db or add_external_local_df:
                    if use_local_db:
                        sampled_local_df = sample_local_df(
                            local_db, ref_movie_df["genres"].iloc[0].split(",")
                        )
                        sampled_local_df = pd.concat(
                            [sampled_local_df, ref_movie_df], axis=0, ignore_index=True
                        )
                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the local movies DataFrame:"
                        )
                        sampled_local_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")

                        if add_external_local_df:
                            sampled_local_df = pd.concat(
                                sampled_local_df,
                                external_df,
                                on="tdb_id",
                                ignore_index=True,
                            )
                            st.write(" ")
                            st.write(
                                "The data is being sourced from local db and external files. From local db the initial selection criterion is **genre**. Approximately 1K movies with similar genres will be sources to create a custom dataset for your analysis."
                            )

                            print(f"\n{'='*50}")
                            print(
                                f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                            )
                            sampled_local_df.info(
                                verbose=True,
                                buf=None,
                                max_cols=None,
                                memory_usage="deep",
                            )
                            print(f"{'='*50}\n")

                        else:
                            st.write(" ")
                            st.write(
                                "The data is being sourced from *TMDB*. The initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                            )

                    elif add_external_local_df:
                        sampled_local_df = external_df
                        st.write("The data is being sourced from the attached files.")

                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                        )
                        sampled_local_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")

                    st.subheader("Metrics")
                    mean_metrics = get_mean_values(sampled_local_df)
                    m1, m2 = st.columns(2, vertical_alignment="center")
                    m1.metric(
                        "Average Rating",
                        f"{mean_metrics.loc['vote_average']}#",
                        border=True,
                        help="Average rating for the movies in the dataset.",
                    )
                    m2.metric(
                        " Average Vote Count",
                        f"{mean_metrics.loc['vote_count']}#",
                        border=True,
                        help="Average vote count for the movies in the dataset.",
                    )
                    st.write(CUSTOM_METRIC, unsafe_allow_html=True)

                    genre_col, yearly_dist_col = st.columns(2)
                    with genre_col:
                        st.subheader("Genre Distribution")
                        st.plotly_chart(
                            plot_categorical_column_percentages(
                                sampled_local_df, "genres"
                            ),
                            use_container_width=True,
                        )
                    with yearly_dist_col:
                        st.subheader("Yearly Distribution")
                        st.plotly_chart(
                            plot_movies_by_year(sampled_local_df, "release_year"),
                            use_container_width=True,
                        )
            else:
                st.write("No correct or not enough data submitted.")
        with tab2:
            if (
                (
                    use_tmdb_api
                    or use_local_db
                    or add_external_local_df
                    or add_external_tmdb_df
                )
                and (movie_ref_url or movie_ref_local or movie_ref_tmdb)
                and st.session_state.data_submitted
            ):
                if use_tmdb_api or add_external_tmdb_df:
                    st.header("K-Prototypes Clustering")
                    tmdb_movies_prepared_df = prepare_tmdb_data_for_clustering(
                        tmdb_movies_df
                    )
                    st.write("Using default hyperparameters (8 clusters).")
                    categorical_columns = [
                        "original_language",
                        "country_of_origin",
                        "genres",
                    ]
                    df_kproto = k_prototypes_clustering(
                        tmdb_movies_prepared_df, categorical_columns
                    )
                    df_kproto_final = merge_clusters_with_preprocessed_df(
                        tmdb_movies_df, df_kproto
                    )

                    st.subheader("Reference Movie Data")
                    df_reference_with_cluster = ref_movie_df.merge(
                        df_kproto_final[["tmdb_id", "cluster"]],
                        on="tmdb_id",
                        how="left",
                    )

                    st.dataframe(
                        df_reference_with_cluster,
                        use_container_width=True,
                        hide_index=True,
                    )

                    genre_top_col, cluster_dist_col = st.columns(2)
                    with genre_top_col:
                        st.subheader("Top Cluster Genre")
                        st.dataframe(
                            calculate_top_genres(df_kproto_final),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with cluster_dist_col:
                        st.subheader("Cluster distribution")
                        st.plotly_chart(
                            plot_cluster_distribution_pie(df_kproto_final),
                            use_container_width=True,
                        )

                    st.subheader("Cluster Numerical Feature Averages")
                    st.plotly_chart(
                        plot_cluster_comparison_subplots(df_kproto_final),
                        use_container_width=True,
                    )

                    st.subheader("2D Cluster Visualization using PCA")
                    features = [
                        col
                        for col in df_kproto.columns
                        if col not in ["tmdb_id", "title", "cluster"]
                    ]
                    st.plotly_chart(
                        plot_clusters_with_pca(
                            df_kproto,
                            cluster_column="cluster",
                            title_column="title",
                            id_column="tmdb_id",
                            features=features,
                        ),
                        use_container_width=True,
                    )

                    st.subheader("Top 10 Movie Recommendations")
                    st.write(
                        "The recommended movies belong to the same cluster as the reference movie and are ranked by their Euclidean distance."
                    )
                    features = [
                        col
                        for col in df_kproto.columns
                        if col not in ["tmdb_id", "title", "cluster"]
                    ]

                    top_similar_movies_dist_df = recommend_similar_movies(
                        df_kproto, ref_movie_df, features
                    )
                    top_similar_movies_df = df_kproto_final.merge(
                        top_similar_movies_dist_df[
                            ["tmdb_id", "distance_to_reference"]
                        ],
                        on="tmdb_id",
                        how="right",
                    )
                    st.dataframe(
                        top_similar_movies_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # st.header("Decision Tree Clustering")
                    # st.subheader("Self-Organizing Maps (SOM)")
                elif use_local_db or add_external_local_df:
                    st.header("K-Prototypes Clustering")
                    local_movies_prepared_df = prepare_local_data_for_clustering(
                        sampled_local_df
                    )

                    st.write("Using default hyperparameters (8 clusters).")
                    categorical_columns = [
                        "genres",
                    ]
                    df_kproto = k_prototypes_clustering(
                        local_movies_prepared_df,
                        categorical_columns,
                        id_column="imdb_id",
                        n_clusters=6,
                    )

                    df_kproto_final = merge_clusters_with_preprocessed_df(
                        sampled_local_df, df_kproto, id_column="imdb_id"
                    )

                    st.subheader("Reference Movie Data")
                    df_reference_with_cluster = ref_movie_df.merge(
                        df_kproto_final[["imdb_id", "cluster"]],
                        on="imdb_id",
                        how="left",
                    )

                    st.dataframe(
                        df_reference_with_cluster,
                        use_container_width=True,
                        hide_index=True,
                    )

                    genre_top_col, cluster_dist_col = st.columns(2)
                    with genre_top_col:
                        st.subheader("Top Cluster Genre")
                        st.dataframe(
                            calculate_top_genres(df_kproto_final),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with cluster_dist_col:
                        st.subheader("Cluster distribution")
                        st.plotly_chart(
                            plot_cluster_distribution_pie(df_kproto_final),
                            use_container_width=True,
                        )

                    top_col, pca_col = st.columns(2)
                    with top_col:
                        st.subheader("Cluster Numerical Feature Averages")
                        st.plotly_chart(
                            plot_cluster_comparison_subplots(
                                df_kproto_final,
                                numeric_columns=["vote_average", "vote_count"],
                            ),
                            use_container_width=True,
                        )

                    with pca_col:
                        st.subheader("2D Cluster Visualization using PCA")
                        features = [
                            col
                            for col in df_kproto.columns
                            if col not in ["imdb_id", "title", "cluster"]
                        ]
                        st.plotly_chart(
                            plot_clusters_with_pca(
                                df_kproto,
                                cluster_column="cluster",
                                title_column="title",
                                id_column="imdb_id",
                                features=features,
                            ),
                            use_container_width=True,
                        )

                    st.subheader("Top 10 Movie Recommendations")
                    st.write(
                        "The recommended movies belong to the same cluster as the reference movie and are ranked by their Euclidean distance."
                    )
                    features = [
                        col
                        for col in df_kproto.columns
                        if col not in ["imdb_id", "title", "cluster"]
                    ]

                    top_similar_movies_dist_df = recommend_similar_movies(
                        df_kproto, ref_movie_df, features, id_column="imdb_id"
                    )
                    top_similar_movies_df = df_kproto_final.merge(
                        top_similar_movies_dist_df[
                            ["imdb_id", "distance_to_reference"]
                        ],
                        on="imdb_id",
                        how="right",
                    )
                    st.dataframe(
                        top_similar_movies_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            else:
                st.write("No correct or not enough data submitted.")


if __name__ == "__main__":
    main()
