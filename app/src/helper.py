import datetime

import pandas as pd


def get_reference_from_url():
    """
    Get movie data from a URL
    """
    ref_movie_df = pd.DataFrame()
    found_movie_data_flag = True
    res = {"ref_movie_df": ref_movie_df, "found_movie_data_flag": found_movie_data_flag}
    return res


def get_timestamp():
    """Generate a timestamp in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_mean_values(df, decimal_places=2):
    """Returns the mean values of numeric columns from the DataFrame."""

    stats = df.describe()

    mean_values = stats.loc["mean"]

    mean_values_rounded = mean_values.round(decimal_places)

    return mean_values_rounded
