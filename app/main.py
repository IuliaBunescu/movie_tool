import pandas as pd
import streamlit as st
from src.components.custom_html import (
    CUSTOM_ALERT_ERROR,
    CUSTOM_ALERT_SUCCESS,
    CUSTOM_FORM,
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

        # local database upload
        file_path = "../data/imdb_movie_titles.tsv"
        movie_names_df = pd.read_csv(file_path, sep="\t", dtype=str)

        with st.form("input_form"):

            st.subheader("Reference Movie")

            website_for_scraping = st.text_input(
                "**Option 1**: Rotten Tomatoes or IMDB link "
            )
            if website_for_scraping:
                st.success("Your website is valid.")
                st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)

            title_type = st.radio(
                "**Option 2** : Search our local database for a movie.",
                ["Popular title (usually in English)", "Original title"],
                horizontal=True,
                index=0,
            )
            title_type_value = (
                0 if title_type == "Popular title (usually in English)" else 1
            )
            column_name = "primaryTitle" if title_type_value == 0 else "originalTitle"
            movie_name_ref = st.text_input("Movie name")

            # Initialize the variable for movie reference in local DB
            movie_ref_local = False

            # Check if the entered movie name exists in the selected title column
            if movie_name_ref:
                # Filter the DataFrame to check if the entered movie name is in the selected column
                filtered_df = movie_names_df[
                    movie_names_df[column_name].str.contains(
                        movie_name_ref, case=False, na=False
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
                        f"Reference movie '{movie_name_ref}' data not found in local database. Try another movie or use the website *Option 1*."
                    )
                    st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

            st.subheader("Supporting Data")

            uploaded_files = st.file_uploader(
                "Choose CSV/TSV files", accept_multiple_files=True
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

            use_tmdb_api = st.toggle("Use TMDB API", value=True)
            use_jikan_api = st.toggle("Use Jikan API for adding anime data")

            data_submitted = st.form_submit_button("Submit", use_container_width=True)

        st.write(CUSTOM_FORM, unsafe_allow_html=True)
        if data_submitted:
            st.write("*Your input data was submitted!*")

    with col2:
        st.header("Results", divider="gray")

        tab1, tab2 = st.tabs(
            ["Exploratory Visualizations of Your Data", "Clustering & Recommendations"]
        )

        with tab1:
            if not data_submitted:
                st.write("No data submitted.")
            else:
                for df in dataframes:
                    st.dataframe(df.head())

        with tab2:
            if not data_submitted:
                st.write("No data submitted.")


if __name__ == "__main__":
    main()
