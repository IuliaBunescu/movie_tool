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
