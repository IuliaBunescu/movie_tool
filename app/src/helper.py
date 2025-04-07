import datetime

import pandas as pd
import streamlit as st


def get_timestamp():
    """Generate a timestamp in the format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_median_values(df, decimal_places=2):
    """Returns the median values (50th percentile) of numeric columns from the DataFrame."""

    stats = df.describe()

    # Extract median values (50th percentile)
    median_values = stats.loc["50%"]

    # Print the median values
    print(median_values)

    # Round the median values to the specified decimal places
    median_values_rounded = median_values.round(decimal_places)

    return median_values_rounded
