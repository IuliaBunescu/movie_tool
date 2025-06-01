import pathlib

import pandas as pd
import streamlit as st
from src.clustering import merge_with_preprocessed_df
from src.fragments import clustering_visualization
from src.plots import (
    plot_average_popularity_by_year,
    plot_categorical_column_percentages,
    plot_column_distribution_pie,
    plot_country_counts,
    plot_movies_by_year,
    plot_popularity_vs_vote,
    plot_top_bigrams,
    plot_votes_vs_score,
    plot_word_cloud,
)
from src.preprocessing import apply_pca, prepare_data_for_clustering
from src.text_column_processing import (
    check_and_download_nltk_resources,
    clean_text,
    get_most_common_words,
    get_top_bigrams,
)
from src.tmdb import (
    extract_tmdb_id,
    get_movies_by_genres_from_reference_df,
    search_first_movie_by_title_and_year_tmdb,
)
from src.utils import get_median_values, get_timestamp, load_css

st.set_page_config(layout="wide")

CURRENT_DIR = pathlib.Path(__file__).parent
css_path = CURRENT_DIR / "assets" / "style.css"
load_css(css_path)


def submit_form():
    st.session_state.data_submitted = True


def main():
    st.title("Movie Recommendation Tool for Data Scientists")

    col1, col2 = st.columns([2, 8])

    with col1:
        st.header("Movie Input", divider="gray")

        with st.form("input_form", enter_to_submit=False):

            if "initialized" not in st.session_state:
                st.session_state.data_submitted = False
                st.session_state.initialized = True

            st.subheader("Reference Movie")

            st.write("**Option 1**: Search reference using TMDB API.")
            movie_ref_tmdb = False
            ds3, ds4 = st.columns([2, 1])

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

                if st.session_state.data_submitted:
                    if not ref_movie_df.empty:
                        st.success(
                            f"Data for '{movie_name_tmdb_ref}' found using TMDB API."
                        )
                        movie_ref_tmdb = True
                        print(f"\n{'='*50}")
                        print(
                            f"[{get_timestamp()}] Logging information about the TMDB reference movies DataFrame:"
                        )
                        ref_movie_df.info(
                            verbose=True, buf=None, max_cols=None, memory_usage="deep"
                        )
                        print(f"{'='*50}\n")
                    else:
                        st.error(
                            f"*Reference movie '{movie_name_tmdb_ref}' data not found using TMDB API . Try other reference input options.*"
                        )

            website_for_scraping = st.text_input("**Option 2**: TMDB link ")
            movie_ref_url = False

            if website_for_scraping and st.session_state.data_submitted:
                url_ref_movie_dict = extract_tmdb_id(website_for_scraping)
                if url_ref_movie_dict.get("found_movie_data_flag"):
                    st.success("Your website is valid.")
                    movie_ref_url = True
                    ref_movie_df = url_ref_movie_dict.get("ref_movie_df")
                elif st.session_state.data_submitted:
                    st.error(
                        "Did not find movie data in reference URL. Try other reference input options."
                    )

            st.form_submit_button(
                "Submit", use_container_width=True, on_click=submit_form
            )
        if st.session_state.data_submitted:
            st.info(f"*Your input data was submitted!*")
        else:
            st.warning("Please fill out the input form.")

        preprocessing_container = st.container()

    with col2:
        st.header("Results", divider="gray")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Exploratory Visualization of Data",
                "Cross-Encoder Similarity",
                "K-Means Clustering",
                "Agglomerative Clustering",
                "K-Prototypes Clustering",
                "Comparative Analysis",
            ]
        )

        with tab1:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:

                st.subheader("Reference Movie Data")
                st.dataframe(ref_movie_df, use_container_width=True, hide_index=True)

                st.write(
                    "The data is being sourced from *TMDB*. The initial selection criterion is **genre**. Approximately 1K movies with similar genres and at least 100 votes will be sources to create a custom dataset for your analysis."
                )

                tmdb_movies_df = get_movies_by_genres_from_reference_df(ref_movie_df)
                print(f"\n{'='*50}")
                print(
                    f"[{get_timestamp()}] Logging information about the TMDB movies DataFrame:"
                )
                tmdb_movies_df.info(
                    verbose=True, buf=None, max_cols=None, memory_usage="deep"
                )
                print(f"{'='*50}\n")

                st.subheader("Metrics")
                mean_metrics = get_median_values(tmdb_movies_df)

                m1, m2, m3, m4 = st.columns(4, vertical_alignment="center")
                m1.metric(
                    "Median Average Rating",
                    f"{mean_metrics.iloc[1]:,.2f}#",
                    border=True,
                    help="Median average rating for the movies in the dataset.",
                )
                m2.metric(
                    "Median Vote Count",
                    f"{mean_metrics.iloc[2]:,.2f}#",
                    border=True,
                    help="Median vote count for the movies in the dataset.",
                )
                m3.metric(
                    "Median Popularity Score",
                    f"{mean_metrics.iloc[3]:,.2f}#",
                    border=True,
                    help="Median popularity score for the movies in the dataset.",
                )
                m4.metric(
                    "% of IMDB movies",
                    f"{(tmdb_movies_df['imdb_id'].notnull().sum() / len(tmdb_movies_df) * 100):,.2f} %",
                    border=True,
                    help="Percentage of movies that have IMDB id.",
                )
                genre_col, lang_col = st.columns(2)
                with genre_col:
                    st.subheader("Genre Distribution")
                    st.plotly_chart(
                        plot_categorical_column_percentages(tmdb_movies_df, "genres"),
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

                st.subheader("Country Distribution (excluding USA)")
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
                release_year_vs_popularity_col, vote_avg_vs_counts_col = st.columns(2)
                with release_year_vs_popularity_col:
                    st.subheader("Average TMDB Popularity Score Yearly Distribution")
                    st.plotly_chart(
                        plot_average_popularity_by_year(tmdb_movies_df),
                        use_container_width=True,
                    )
                with vote_avg_vs_counts_col:
                    st.subheader("Average Rating VS Vote Counts")
                    st.plotly_chart(
                        plot_votes_vs_score(tmdb_movies_df),
                        use_container_width=True,
                    )

                wordcloud_col, top_bigrams_col = st.columns(2)
                check_and_download_nltk_resources()

                # Clean text column and get word and bigram frequency
                tmdb_movies_df["cleaned_overview"] = tmdb_movies_df["overview"].apply(
                    clean_text
                )
                common_words = get_most_common_words(
                    tmdb_movies_df["cleaned_overview"], top_n=100
                )
                top_bigrams = get_top_bigrams(
                    tmdb_movies_df["cleaned_overview"], top_n=20
                )

                with wordcloud_col:
                    st.subheader("Most Frequent 100 Words")
                    st.pyplot(plot_word_cloud(common_words), use_container_width=True)

                with top_bigrams_col:
                    st.subheader("Most Frequent 20 Bigrams")
                    st.plotly_chart(
                        plot_top_bigrams(top_bigrams), use_container_width=True
                    )

            else:
                st.info("*Please complete the input section.*")

        if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
            with preprocessing_container:
                st.header("Preprocessing Data", divider="gray")

                # Preparing data for clustering
                df_basic, df_numerical, df_full = prepare_data_for_clustering(
                    tmdb_movies_df, ref_movie_df["tmdb_id"].values[0]
                )

                # Select numerical features
                features = df_numerical.drop(
                    columns=["tmdb_id", "title"]
                ).columns.tolist()

                # Apply PCA once on numerical
                df_numerical_reduced = apply_pca(
                    df_numerical,
                    features,
                    n_components=len(features),
                    explained_variance_threshold=0.9,
                )

                df_full_reduced = df_full[
                    [
                        "tmdb_id",
                        "title",
                        "country_of_origin",
                        "original_language",
                    ]
                ].merge(
                    df_numerical_reduced.drop(columns=["title"]),
                    on="tmdb_id",
                    how="left",
                )

        with tab2:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
                st.header("Cross-Encoder Similarity")

                st.write(
                    "The similarity between the target movie and others was computed using a RoBERTa-based cross-encoder, which directly compares each pair of movie overviews from the `cleaned_overview` column."
                )
                # Merge results with original metadata
                merged_similarity_df = merge_with_preprocessed_df(
                    tmdb_movies_df, df_basic, expect_cluster=False
                )
                df_sorted = merged_similarity_df.sort_values(
                    by="similarity_to_reference", ascending=False
                )

                # Select top 11 rows: first one is reference (similarity = 1), next 10 are recommendations
                df_final_recommendations = df_sorted.head(11).reset_index(drop=True)

                st.subheader("Top 10 Movie Recommendations")
                st.dataframe(
                    df_final_recommendations.drop(columns=["tmdb_id"]),
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

            else:
                st.info("*Please complete the input section.*")

        with tab3:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
                st.header("K-Means Clustering")
                clustering_visualization(
                    algo_name="K-Means Clustering",
                    preprocessed_df=df_numerical_reduced,
                    reference_df=ref_movie_df,
                    original_df=tmdb_movies_df,
                )
            else:
                st.info("*Please complete the input section.*")

        with tab4:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
                st.header("Agglomerative Clustering")
                clustering_visualization(
                    algo_name="Agglomerative Clustering",
                    preprocessed_df=df_numerical_reduced,
                    reference_df=ref_movie_df,
                    original_df=tmdb_movies_df,
                )
            else:
                st.info("*Please complete the input section.*")

        with tab5:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
                st.header("K-Prototypes Clustering")

                clustering_visualization(
                    algo_name="K-Prototypes Clustering",
                    preprocessed_df=df_full_reduced,
                    reference_df=ref_movie_df,
                    original_df=tmdb_movies_df,
                )

            else:
                st.info("*Please complete the input section.*")

        with tab6:
            if (movie_ref_url or movie_ref_tmdb) and st.session_state.data_submitted:
                st.header("Comparative Analysis")

                st.subheader("Cluster Quality Evaluation")
                st.session_state.cluster_quality_df.drop_duplicates(
                    subset=["algorithm"], inplace=True, keep="last"
                )
                st.dataframe(st.session_state.cluster_quality_df)

            else:
                st.info("*Please complete the input section.*")


if __name__ == "__main__":
    main()
