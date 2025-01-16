import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def search_movie_in_local_db(df, movie_name, year=None):
    # Load the local database

    # Convert the movie name to lowercase for case-insensitive matching
    movie_name = movie_name.lower()

    # Filter based on movie name
    matches = df[df["title"].str.lower().str.contains(movie_name, na=False)]

    # If a year is provided, further filter by year
    if year is not None:
        matches = matches[matches["release_year"] == year]

    # Check if any matches were found
    found_movie_data_flag = not matches.empty

    # Return the result
    if found_movie_data_flag:
        return {
            "movie_data": matches.iloc[0:1],
            "found_movie_data_flag": found_movie_data_flag,
        }
    else:
        return {
            "movie_data": pd.DataFrame(),
            "found_movie_data_flag": found_movie_data_flag,
        }


def load_local_db(file_path):
    """
    This function loads a CSV file, ensures that the columns of the DataFrame have the correct types,
    and returns the corrected DataFrame.
    The expected types for columns are:
    - 'imdb_id' -> str
    - 'title' -> str
    - 'originalTitle' -> str
    - 'release_year' -> int
    - 'genres' -> str
    - 'vote_average' -> float
    - 'vote_count' -> int
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Ensure correct column types
    df["imdb_id"] = df["imdb_id"].astype(str)
    df["title"] = df["title"].astype(str)
    df["originalTitle"] = df["originalTitle"].astype(str)
    df["release_year"] = df["release_year"].astype(int)
    df["genres"] = df["genres"].astype(str)  # Adjust if genres are stored differently
    df["vote_average"] = df["vote_average"].astype(float)
    df["vote_count"] = df["vote_count"].astype(int)

    return df


def sample_local_df(df, reference_genres, n_samples=1000):
    """
    Returns a DataFrame with up to 1,000 entries selected where genres match the reference movie.
    First tries to find movies with all reference genres. If not enough, includes movies with at least one genre.

    Parameters:
    - df: The DataFrame containing movie data.
    - reference_genres: A list of genres from the reference movie.
    - n_samples: The number of samples to return (default is 1,000).

    Returns:
    - A DataFrame with up to n_samples entries matching the genres of the reference movie.
    """
    # Filter rows where genres contain all genres from the reference movie
    print("Filtering rows with all genres...")
    filtered_all_genres = df[
        df["genres"].apply(
            lambda x: all(genre.strip() in x.split(",") for genre in reference_genres)
        )
    ]

    # Check if the filtered DataFrame meets the sample size
    if len(filtered_all_genres) >= n_samples:
        sampled_df = filtered_all_genres.sample(n=n_samples, random_state=42)
        print(
            f"Selected {len(sampled_df)} entries matching ALL genres: {reference_genres}."
        )
        return sampled_df

    # If not enough movies, use movies with at least one matching genre
    print(
        f"Not enough movies found with all genres ({len(filtered_all_genres)} found). Relaxing criteria..."
    )
    filtered_any_genre = df[
        df["genres"].apply(
            lambda x: any(genre.strip() in x.split(",") for genre in reference_genres)
        )
    ]

    # Randomly sample from this filtered DataFrame
    if len(filtered_any_genre) > n_samples:
        sampled_df = filtered_any_genre.sample(n=n_samples, random_state=42)
    else:
        sampled_df = filtered_any_genre

    print(
        f"Selected {len(sampled_df)} entries matching AT LEAST ONE genre: {reference_genres}."
    )

    return sampled_df


@st.cache_data(
    ttl=datetime.timedelta(hours=12), show_spinner="Preparing data for clustering..."
)
def prepare_local_data_for_clustering(df):
    """
    Prepares a DataFrame for clustering by scaling numerical features and encoding categorical features.

    Parameters:
    - df: Original DataFrame with movie data.

    Returns:
    - Transformed DataFrame ready for clustering.
    """
    print(f"\n{'='*50}")
    print(f"Started preparing dataset for clustering.")

    # Step 1: Keep imdb_id and title for reference
    reference_columns = df[["imdb_id", "title"]]

    # Step 2: Separate features into numerical and categorical
    numerical_features = [
        "vote_average",
        "vote_count",
        "release_year",
    ]

    # Step 3: Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]),
        columns=numerical_features,
        index=df.index,  # Preserve original indices
    )

    print(f"- Scaled numerical columns: {numerical_features} using StandardScaler.")

    # Step 4: Encode multi-value categorical features (multi-hot encoding)
    def multi_hot_encode(column, unique_values):
        """
        Multi-hot encodes a column with lists of values.
        """
        multi_hot = np.zeros((len(column), len(unique_values)), dtype=int)
        for i, values in enumerate(column):
            for value in values.split(","):
                if value in unique_values:
                    multi_hot[i, unique_values.index(value)] = 1
        return pd.DataFrame(
            multi_hot,
            columns=[f"{column.name}_{val}" for val in unique_values],
            index=column.index,  # Preserve original indices
        )

    # Get unique values for the genres column
    unique_genres = sorted(
        set(genre.strip() for genres in df["genres"] for genre in genres.split(","))
    )
    encoded_genres = multi_hot_encode(df["genres"], unique_genres)
    print(
        f"- Multi-hot encoded 'genres' feature with {len(unique_genres)} unique values."
    )

    # Step 5: Combine all transformed features
    transformed_df = pd.concat(
        [reference_columns, scaled_numerical, encoded_genres],
        axis=1,
    )

    print(f"\n{'='*50}")
    print(f"Dataset preparation completed.")

    return transformed_df
