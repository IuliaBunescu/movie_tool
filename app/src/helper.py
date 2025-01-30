import datetime

import pandas as pd
import streamlit as st
from src.components.custom_html import CUSTOM_ALERT_ERROR, CUSTOM_ALERT_SUCCESS


def get_timestamp():
    """Generate a timestamp in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_mean_values(df, decimal_places=2):
    """Returns the mean values of numeric columns from the DataFrame."""

    stats = df.describe()

    mean_values = stats.loc["mean"]

    print(mean_values)

    mean_values_rounded = mean_values.round(decimal_places)

    return mean_values_rounded


def process_uploaded_files(uploaded_files):
    """
    Processes uploaded files to create a consolidated dataframe and determines its type.

    Args:
        uploaded_files (list): List of uploaded file objects.

    Returns:
        dict: A dictionary containing the combined dataframe and a flag indicating its type.
              {
                  "dataframe": <pd.DataFrame>,
                  "type_flag": "local" or "tmdb" or "unknown"
              }
    """
    # Define expected columns for each type
    local_columns = set(
        [
            "imdb_id",
            "title",
            "originalTitle",
            "release_year",
            "genres",
            "vote_average",
            "vote_count",
        ]
    )
    tmdb_columns = set(
        [
            "tmdb_id",
            "title",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
            "imdb_id",
            "original_language",
            "country_of_origin",
            "genres",
        ]
    )

    # List to store dataframes from all files
    dataframes = []

    for uploaded_file in uploaded_files:
        # Determine the separator based on file extension
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension in ["csv", "tsv"]:
            separator = "," if file_extension == "csv" else "\t"

            try:
                # Read the file into a pandas dataframe
                df = pd.read_csv(uploaded_file, sep=separator)
                dataframes.append(df)
                st.success(f"Created Pandas dataframe from file {uploaded_file.name}.")
                st.write(CUSTOM_ALERT_SUCCESS, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
                st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            st.write(CUSTOM_ALERT_ERROR, unsafe_allow_html=True)

    # Combine all dataframes
    if not dataframes:
        st.error("No valid dataframes created from the uploaded files.")
        return {"dataframe": pd.DataFrame(), "type_flag": "unknown"}

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Determine the type of the dataframe based on its columns
    combined_columns = set(combined_df.columns)
    if local_columns.issubset(combined_columns):
        type_flag = "local"
    elif tmdb_columns.issubset(combined_columns):
        type_flag = "tmdb"
    else:
        type_flag = "unknown"

    return {"dataframe": combined_df, "type_flag": type_flag}
