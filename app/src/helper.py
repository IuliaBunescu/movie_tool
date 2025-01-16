import datetime

import pandas as pd
from src.scraper import scrape_imdb_url


def get_reference_from_url(imdb_url):
    """
    Scrapes IMDb movie details and returns a dataframe with the movie data.

    Args:
    imdb_url (str): The IMDb URL of the movie to scrape.

    Returns:
    dict: A dictionary containing the dataframe and a flag indicating if the data was found.
    """
    movie_details = scrape_imdb_url(imdb_url)

    # Check if data was found
    if movie_details:
        # Create DataFrame
        ref_movie_df = pd.DataFrame(
            [
                {
                    "imdb_id": movie_details.get("imdb_id", "N/A"),
                    "title": movie_details.get("title", "N/A"),
                    "description": movie_details.get("description", "N/A"),
                    "release_date": movie_details.get("release_date", "N/A"),
                    "rating": movie_details.get("rating", "N/A"),
                    "vote_count": movie_details.get("vote_count", "N/A"),
                    "genres": movie_details.get("genres", "N/A"),
                    "original_language": movie_details.get("original_language", "N/A"),
                    "country_of_origin": movie_details.get("country_of_origin", "N/A"),
                }
            ]
        )

        found_movie_data_flag = True
    else:
        ref_movie_df = pd.DataFrame()
        found_movie_data_flag = False

    return {
        "ref_movie_df": ref_movie_df,
        "found_movie_data_flag": found_movie_data_flag,
    }


def get_timestamp():
    """Generate a timestamp in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_mean_values(df, decimal_places=2):
    """Returns the mean values of numeric columns from the DataFrame."""

    stats = df.describe()

    mean_values = stats.loc["mean"]

    mean_values_rounded = mean_values.round(decimal_places)

    return mean_values_rounded
