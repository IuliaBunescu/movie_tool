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
from src.helper import get_mean_values, get_reference_from_url, get_timestamp
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
from src.tmdb import (
    get_movies_by_genre_from_reference_df,
    prepare_data_for_clustering,
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

        # local database upload
        file_path = "../data/imdb_movie_titles.tsv"
        movie_names_df = pd.read_csv(file_path, sep="\t", dtype=str)

        with st.form("input_form", enter_to_submit=False):
            st.write(CUSTOM_FORM, unsafe_allow_html=True)

            if "data_submitted" not in st.session_state:
                st.session_state.data_submitted = False

            st.subheader("Reference Movie")

            title_type = st.radio(
                "**Option 1** : Search our local database for a movie.",
                ["Popular title (usually in English)", "Original title"],
                horizontal=True,
                index=0,
                key="input_title_type_local",
            )
            title_type_value = (
                0 if title_type == "Popular title (usually in English)" else 1
            )
            column_name = "primaryTitle" if title_type_value == 0 else "originalTitle"

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
                filtered_df = movie_names_df[
                    movie_names_df[column_name].str.contains(
                        movie_name_local_ref, case=False, na=False
                    )
                ]

                # If there are matching rows, use the first match
                if not filtered_df.empty:
                    first_matching_movie = filtered_df.iloc[0][
                        column_name
                    ]  # Get the first matching movie name
                    movie_ref_local = True
                    st.success(
                        f"Data for '{first_matching_movie}' found in local database."
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
            else:
                use_local_db = True

            uploaded_files = st.file_uploader(
                "Add your own CSV/TSV files",
                accept_multiple_files=True,
                key="input_tabular_file",
            )

            dataframes = []

            for uploaded_file in uploaded_files:
                # Determine the separator based on file extension
                file_extension = uploaded_file.name.split(".")[-1]
                if file_extension in ["csv", "tsv"]:
                    separator = "," if file_extension == "csv" else "\t"
                    # Read the file into a pandas dataframe
                    df = pd.read_csv(uploaded_file, sep=separator)
                    dataframes.append(df)
                    st.success(
                        f"Created Pandas dataframe from file {uploaded_file.name}."
                    )
                    st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                else:
                    st.error(f"Unsupported file type: {uploaded_file.name}")
                    st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            form_button = st.form_submit_button(
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
                (use_tmdb_api or use_local_db or dataframes)
                and (movie_ref_url or movie_ref_local or movie_ref_tmdb)
                and st.session_state.data_submitted
            ):

                st.subheader("Reference Movie Data")
                st.dataframe(ref_movie_df, use_container_width=True)

                if use_tmdb_api:
                    st.write(" ")
                    st.write(
                        "The data is being sourced from *TMDB*. The initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                    )

                    tmdb_movies_df = get_movies_by_genre_from_reference_df(ref_movie_df)
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
                        "Rating",
                        f"{mean_metrics.iloc[1]}#",
                        border=True,
                        help="Average rating for the movies in the dataset.",
                    )
                    m2.metric(
                        "Vote Count",
                        f"{mean_metrics.iloc[2]}#",
                        border=True,
                        help="Average vote count for the movies in the dataset.",
                    )
                    m3.metric(
                        "Popularity",
                        f"{mean_metrics.iloc[3]}#",
                        border=True,
                        help="Average popularity score for the movies in the dataset.",
                    )
                    m4.metric(
                        "IMDB movies",
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

                # for df in dataframes:
                #     st.dataframe(df.head(), use_container_width=True, hide_index=True)
            else:
                st.write("No correct or not enough data submitted.")
        with tab2:
            if (
                (use_tmdb_api or use_local_db or dataframes)
                and (movie_ref_url or movie_ref_local or movie_ref_tmdb)
                and st.session_state.data_submitted
            ):
                if use_tmdb_api:
                    st.header("K-Prototypes Clustering")
                    tmdb_movies_prepared_df = prepare_data_for_clustering(
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

                    st.dataframe(df_reference_with_cluster, use_container_width=True)

                    genre_top_col, cluster_dist_col = st.columns(2)
                    with genre_top_col:
                        st.subheader("Top Cluster Genre")
                        st.dataframe(
                            calculate_top_genres(df_kproto_final),
                            use_container_width=True,
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

            else:
                st.write("No correct or not enough data submitted.")


if __name__ == "__main__":
    main()
