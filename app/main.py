import pandas as pd
import streamlit as st
from src.components.custom_html import (
    CUSTOM_ALERT_ERROR,
    CUSTOM_ALERT_SUCCESS,
    CUSTOM_FORM,
)
from src.helpers import (
    get_reference_from_url,
    search_first_movie_by_title_and_year_tmdb,
)

st.set_page_config(layout="wide")


def main():
    st.title("Movie Recommendation Tool for Data Scientists 🎬")

    col1, col2 = st.columns([3, 7])

    with col1:
        st.header("Choose your data source", divider="gray")

        st.markdown(
            "Choose your data input method by providing a **Reference Movie** and selecting one or more **Supporting Data Sources**. After adding all the required inputs, click the **Submit** button. Once submitted, proceed to the **Results** section to view your output."
        )
        st.markdown(
            "As a recommendation, when using *TMDB* reference data, use *TMDB* supporting data as well. "
        )

        # local database upload
        file_path = "../data/imdb_movie_titles.tsv"
        movie_names_df = pd.read_csv(file_path, sep="\t", dtype=str)

        with st.form("input_form", enter_to_submit=False):
            st.write(CUSTOM_FORM, unsafe_allow_html=True)
            data_submitted = False

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
                print("First Movie Found:")
                st.success(f"Data for '{movie_name_tmdb_ref}' found using TMDB API.")
                st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                movie_ref_tmdb = True
            elif data_submitted:
                st.error(
                    f"Reference movie '{movie_name_tmdb_ref}' data not found using TMDB API . Try other reference input options."
                )
                st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            website_for_scraping = st.text_input(
                "**Option 3**: Rotten Tomatoes or IMDB link "
            )
            movie_ref_url = False

            if website_for_scraping:
                url_ref_movie_dict = get_reference_from_url()
                if url_ref_movie_dict.get("found_movie_data_flag"):
                    st.success("Your website is valid.")
                    st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
                    movie_ref_url = True
                else:
                    st.error(
                        "Did not find movie data in refernece URL. Try other reference input options."
                    )
                    st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            st.subheader("Supporting Data")

            uploaded_files = st.file_uploader(
                "Choose CSV/TSV files",
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

            use_tmdb_api = False
            use_local_db = False
            reference_data_source = st.radio(
                "Chose one of the following sources or leave empty:",
                ["TMDB API data", "Local db data"],
                horizontal=False,
                key="reference_data_input",
            )

            if reference_data_source == "TMDB API data":
                use_tmdb_api = True
            else:
                use_local_db = True

            data_submitted = st.form_submit_button("Submit", use_container_width=True)

        if data_submitted:
            st.write("*Your input data was submitted!*")

    with col2:
        st.header("Results", divider="gray")

        tab1, tab2 = st.tabs(
            ["Exploratory Visualizations of Your Data", "Clustering & Recommendations"]
        )

        with tab1:
            if (
                (use_tmdb_api or use_local_db or dataframes)
                and (movie_ref_url or movie_ref_local or movie_ref_tmdb)
                and data_submitted
            ):

                st.subheader("Reference Movie Data")
                st.dataframe(ref_movie_df)

                if use_tmdb_api:
                    st.write(" ")
                    st.write(
                        "The data has been sourced from *TMDB*. The initial selection criterion is **genre**. 1K movies with similar genres will be sources to create a custom dataset for your analysis. "
                    )

                    st.subheader("General Metrics")

                for df in dataframes:
                    st.dataframe(df.head())
            else:
                st.write("No correct or not enough data submitted.")


if __name__ == "__main__":
    main()
