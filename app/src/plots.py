import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud


def plot_categorical_column_percentages(dataframe, column_name):
    """
    Create a Plotly bar plot showing the percentage of unique values in a specified string column.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to analyze (should contain string data).
        title (str): The title of the plot. If None, a default title will be generated.

    Returns:
        None: Displays the Plotly plot.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return

    # Process the column to calculate percentages
    value_counts = (
        dataframe[column_name]
        .dropna()
        .apply(lambda x: x.split(",") if isinstance(x, str) else x)
        .apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else x)
        .explode()  # Flatten lists into rows
        .value_counts(normalize=True)  # Count occurrences
        .reset_index()  # Reset index for easier plotting
    )
    value_counts.columns = ["Value", "Percentage"]
    value_counts["Percentage"] = (value_counts["Percentage"] * 100).round(2)

    # Create a Plotly bar chart
    fig = px.bar(
        value_counts,
        x="Value",
        y="Percentage",
        labels={"Value": column_name, "Percentage": "Percentage (%)"},
        text="Percentage",
        color_discrete_sequence=["#b95d8d"],
    )

    # Update layout for better appearance
    fig.update_layout(
        xaxis_title=column_name.title(),
        yaxis_title="Percentage (%)",
        xaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        hoverlabel=dict(font=dict(size=14)),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="x",
    )

    # Format the percentage values on the bars
    fig.update_traces(texttemplate="%{text:.2f}%")
    return fig


def plot_country_counts(df, country_column, color1, color2):
    """
    Plots a world map showing the counts for each country with a custom continuous color scale.
    Assumes that `df` has a column with country names, where some entries may contain multiple countries.
    """
    # Preprocess the country data (split entries with multiple countries)
    countries = (
        df[country_column]
        .dropna()
        .str.split(",", expand=True)
        .stack()
        .str.strip()
        .reset_index(drop=True)
    )

    # Count the occurrences of each country
    country_counts = countries.value_counts().reset_index()
    country_counts.columns = ["Country", "Count"]

    # Remove United States from the data
    country_counts = country_counts[
        country_counts["Country"].str.lower() != "united states of america"
    ]

    # Define a custom continuous color scale (with two colors)
    color_scale = [color1, color2]

    # Create the world map plot using Plotly
    fig = px.choropleth(
        country_counts,
        locations="Country",
        locationmode="country names",
        color="Count",
        hover_name="Country",
        color_continuous_scale=color_scale,
    )

    # Remove the default rectangular borders around the map and set map contour to white
    fig.update_geos(
        showcoastlines=False,
        showland=True,
        showlakes=False,
        landcolor="lightgray",
        showcountries=True,
        projection_type="natural earth",
        countrycolor="lightgray",
        showframe=True,
        framecolor="white",
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        hoverlabel=dict(font=dict(size=14)),
    )

    # Show the plot
    return fig


def plot_column_distribution_pie(
    df,
    column_name,
    cap_percentage=3,
    title=None,
    color_sequence=px.colors.sequential.Magenta,
):
    """
    Plots a pie chart showing the distribution of values in a specified column as percentages.
    If there are values with less than `cap_percentage` of the total, they are grouped into "Others".

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to analyze and plot.
        cap_percentage (float): The minimum percentage threshold; values below this will be grouped into "Others".
        title (str): Optional title for the plot.
        color_sequence (list): Optional color sequence for the pie chart.

    Returns:
        fig (plotly.graph_objs._figure.Figure): The Plotly pie chart figure.
    """
    # Preprocess the data (split entries with multiple values if they exist)
    values = (
        df[column_name]
        .dropna()
        .str.split(",", expand=True)
        .stack()
        .str.strip()
        .reset_index(drop=True)
    )

    # Count the occurrences of each unique value
    value_counts = values.value_counts().reset_index()
    value_counts.columns = [column_name, "Count"]

    # Calculate the total count
    total_count = value_counts["Count"].sum()

    # Calculate the percentage of each value
    value_counts["Percentage"] = (value_counts["Count"] / total_count) * 100

    # Create a new category "Others" for values with less than `cap_percentage`
    small_values = value_counts[value_counts["Percentage"] < cap_percentage]
    others_count = small_values["Count"].sum()

    # Filter out the small values and keep only the larger values
    value_counts = value_counts[value_counts["Percentage"] >= cap_percentage]

    # Add the "Others" category using pd.concat
    if others_count > 0:
        others_df = pd.DataFrame(
            {
                column_name: [f"Others (<{cap_percentage}%)"],
                "Count": [others_count],
                "Percentage": [(others_count / total_count) * 100],
            }
        )
        value_counts = pd.concat([value_counts, others_df], ignore_index=True)

    # Plot the pie chart using Plotly
    fig = px.pie(
        value_counts,
        names=column_name,
        values="Count",
        hole=0.3,
        color_discrete_sequence=color_sequence,
        title=title or None,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        hoverlabel=dict(font=dict(size=14)),
    )

    # Show the plot
    return fig


def plot_movies_by_year(df, date_column):
    """
    Plots a line plot showing the number of movies released each year.
    Assumes that `df` has a `release_date` column with date values in 'YYYY-MM-DD' format.
    """
    # Check if the date_column is an integer (year)
    if df[date_column].dtype in ["int64", "float64"]:  # Year as integer
        df["Year"] = df[date_column].astype(int)
    else:  # Assuming the date_column is a full date or datetime
        # Convert to datetime format and extract the year
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df["Year"] = df[date_column].dt.year

    # Count the number of movies released each year
    movie_count_by_year = df.groupby("Year").size().reset_index(name="Movie Count")

    # Create the line plot using Plotly
    fig = px.line(
        movie_count_by_year,
        x="Year",
        y="Movie Count",
        labels={"Year": "Release Year", "Movie Count": "Number of Movies"},
        markers=True,
    )

    fig.update_traces(line=dict(color="#e60000"))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(font=dict(size=14)),
        xaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        hovermode="x unified",
    )

    # Show the plot
    return fig


def plot_popularity_vs_vote(
    df,
    popularity_column="popularity",
    vote_column="vote_average",
    vote_count_column="vote_count",
):
    """
    Plots a scatter plot comparing the popularity score and vote average for each movie.
    Assumes that `df` has columns `popularity` and `vote_average`.

    Parameters:
    - popularity_column: The name of the column for popularity score.
    - vote_column: The name of the column for vote average.
    """
    # Ensure the popularity and vote_average columns are numeric
    df[popularity_column] = pd.to_numeric(df[popularity_column], errors="coerce")
    df[vote_column] = pd.to_numeric(df[vote_column], errors="coerce")

    # Remove any rows with missing data
    df = df.dropna(subset=[popularity_column, vote_column])

    # Create the scatter plot
    fig = px.scatter(
        df,
        x=popularity_column,
        y=vote_column,
        labels={popularity_column: "Popularity Score", vote_column: "Vote Average"},
        color=vote_count_column,
        color_continuous_scale=px.colors.sequential.Agsunset,
        hover_data=[popularity_column, vote_column],
        marginal_x="histogram",
    )  # Show additional info on hover

    # Update the layout to adjust the appearance
    fig.update_layout(
        xaxis_title="Popularity Score",
        yaxis_title="Vote Average (0-10)",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def plot_votes_vs_score(
    df,
    vote_average_column="vote_average",
    vote_count_column="vote_count",
    color_by="popularity",
):
    """
    Plots a scatter plot comparing the vote average (x-axis) and vote count (y-axis) for each movie.
    Points are colored by a third metric, typically popularity.

    Parameters:
    - df: DataFrame containing the movie data.
    - vote_average_column: Column name for the vote average (x-axis).
    - vote_count_column: Column name for the vote count (y-axis).
    - color_by: Column name to use for coloring the points (e.g., 'popularity').
    """
    # Ensure numeric types
    df[vote_average_column] = pd.to_numeric(df[vote_average_column], errors="coerce")
    df[vote_count_column] = pd.to_numeric(df[vote_count_column], errors="coerce")
    df[color_by] = pd.to_numeric(df[color_by], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=[vote_average_column, vote_count_column, color_by])

    # Plot
    fig = px.scatter(
        df,
        x=vote_average_column,
        y=vote_count_column,
        color=color_by,
        color_continuous_scale=px.colors.sequential.Agsunset,
        labels={
            vote_average_column: "Vote Average",
            vote_count_column: "Vote Count",
            color_by: "Popularity Score",
        },
        hover_data=[vote_average_column, vote_count_column, color_by],
        marginal_x="histogram",
    )

    fig.update_layout(
        xaxis_title="Vote Average (0–10)", yaxis_title="Vote Count", showlegend=False
    )

    return fig


def plot_average_popularity_by_year(
    df, date_column="release_date", popularity_column="popularity"
):
    """
    Plots a line graph showing the evolution of average popularity score per year.

    Args:
        df (pd.DataFrame): DataFrame containing movie data.
        date_column (str): Name of the column with release dates (e.g., 'release_date').
        popularity_column (str): Name of the column with popularity scores.

    Returns:
        fig: A Plotly line plot showing average popularity per year.
    """
    # Convert dates to datetime if needed and extract year
    if df[date_column].dtype in ["int64", "float64"]:
        df["Year"] = df[date_column].astype(int)
    else:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df["Year"] = df[date_column].dt.year

    # Ensure popularity is numeric
    df[popularity_column] = pd.to_numeric(df[popularity_column], errors="coerce")

    # Group by year and compute average popularity
    median_popularity_by_year = (
        df.dropna(subset=["Year", popularity_column])
        .groupby("Year")[popularity_column]
        .mean()
        .reset_index(name="Average Popularity")
    )

    # Plot it
    fig = px.line(
        median_popularity_by_year,
        x="Year",
        y="Average Popularity",
        markers=True,
        labels={
            "Year": "Release Year",
            "Average Popularity": "Average Popularity Score",
        },
    )

    fig.update_traces(line=dict(color="#e60000"))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        hoverlabel=dict(font=dict(size=14)),
        xaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        hovermode="x unified",
    )

    return fig


def plot_cluster_comparison_subplots(
    df,
    cluster_column="cluster",
    numeric_columns=["vote_average", "vote_count", "popularity"],
):
    """
    Plots a comparison of average values for specified numeric columns across different clusters in separate subplots.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cluster_column (str): The column that contains the cluster assignments.
        numeric_columns (list of str): List of numeric columns to compare across clusters (e.g., ['vote_average', 'vote_count', 'popularity']).

    Returns:
        fig (plotly.graph_objs._figure.Figure): The Plotly subplots figure comparing the average values for each cluster.
    """
    # Group by the cluster and calculate the mean of the numeric columns
    cluster_means = df.groupby(cluster_column)[numeric_columns].mean().reset_index()

    # Round the values to 2 decimal places
    cluster_means[numeric_columns] = cluster_means[numeric_columns].round(2)

    # Create subplots for each metric
    fig = make_subplots(
        rows=1,
        cols=len(numeric_columns),  # 1 row, 3 columns
        shared_yaxes=False,  # Allow independent y-axes for each subplot
        horizontal_spacing=0.1,
    )

    # Define custom y-axis labels for each metric
    y_axis_labels = {
        "vote_average": "Average Vote Score",
        "vote_count": "Average Vote Count",
        "popularity": "Average Popularity Score",
    }

    # Add each metric as a separate subplot
    for i, column in enumerate(numeric_columns):
        fig.add_trace(
            go.Bar(
                x=cluster_means[cluster_column],
                y=cluster_means[column],
                name=column,
                marker=dict(color=["#ffb0b1", "#ff6666", "#a6edcd"][i % 3]),
            ),
            row=1,
            col=i + 1,
        )

    # Update layout to make the plot more readable
    fig.update_layout(
        showlegend=False,
        barmode="group",
        margin=dict(
            l=0, r=0, t=0, b=40
        ),  # Adjust margin to make space for x-axis label
        hovermode="x",
        xaxis=dict(tickmode="linear"),
        hoverlabel=dict(font=dict(size=14)),
    )

    # Update y-axes labels for each subplot with custom labels
    for i, column in enumerate(numeric_columns):
        fig.update_yaxes(title_text=y_axis_labels[column], row=1, col=i + 1)

    # Update x-axes titles for each subplot
    for i, column in enumerate(numeric_columns):
        fig.update_xaxes(title_text="Cluster", row=1, col=i + 1)

    # Show the plot
    return fig


def plot_cluster_distribution_pie(df, cluster_column="cluster"):
    """
    Plots a pie chart showing the distribution of counts for each cluster.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cluster_column (str): The column that contains the cluster assignments.

    Returns:
        fig (plotly.graph_objs._figure.Figure): The Plotly pie chart figure showing the distribution of clusters.
    """
    cluster_counts = df[cluster_column].value_counts().reset_index()
    cluster_counts.columns = [cluster_column, "Count"]

    fig = go.Figure(
        go.Pie(
            labels=cluster_counts[cluster_column],
            values=cluster_counts["Count"],
            hole=0.3,
            marker=dict(colors=px.colors.sequential.Magenta[: len(cluster_counts)]),
            textinfo="percent+label",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        hoverlabel=dict(font=dict(size=14)),
    )

    return fig


@st.cache_data(ttl=datetime.timedelta(hours=12), show_spinner="Applying PCA...")
def plot_clusters_with_pca(
    df,
    cluster_column="cluster",
    title_column="title",
    id_column="tmdb_id",
    features=None,
):
    """
    Visualizes clusters in 2D space using PCA for dimensionality reduction and includes hover data for title and tmdb_id.

    Args:
        df (pd.DataFrame): The DataFrame containing the preprocessed data.
        cluster_column (str): The column containing cluster labels.
        title_column (str): The column containing movie titles.
        id_column (str): The column containing tmdb_ids.
        features (list): List of features to visualize (exclude non-numeric columns like 'tmdb_id' and 'title').

    Returns:
        fig: A Plotly scatter plot showing the clusters.
    """
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[features])

    # Add PCA results to the dataframe
    df["PCA_1"] = pca_result[:, 0]
    df["PCA_2"] = pca_result[:, 1]

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="PCA_1",
        y="PCA_2",
        color=cluster_column,
        hover_data={id_column: True, title_column: True},
        labels={"PCA_1": "Principal Component 1", "PCA_2": "Principal Component 2"},
        color_discrete_sequence=px.colors.sequential.Agsunset,
    )

    return fig


def plot_word_cloud(word_freq, max_words=100):
    """
    Plots a word cloud from a word frequency dictionary.

    Args:
        word_freq (dict or list of tuples): Word frequency data.
        max_words (int): Max number of words to display in the word cloud.
        title (str): Title of the word cloud plot.
    """
    # Convert list of tuples to dict if needed
    if isinstance(word_freq, list):
        word_freq = dict(word_freq)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="PuRd_r",
        max_words=max_words,
    ).generate_from_frequencies(word_freq)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt


def plot_top_bigrams(bigram_counts):
    """
    Plots the most common bigrams as an interactive horizontal bar chart using Plotly.

    Args:
        bigram_counts (list of tuples): Output from get_top_bigrams.
    """
    bigrams, counts = zip(*bigram_counts)

    fig = px.bar(
        x=counts,
        y=bigrams,
        orientation="h",
        labels={"x": "Frequency", "y": "Bigram"},
        color=counts,
        color_continuous_scale=px.colors.sequential.Magenta,
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending",
            tickfont=dict(size=12),
            automargin=True,  # ensures no labels are cut off
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def plot_clusters_with_tsne(
    df,
    cluster_column="cluster",
    title_column="title",
    id_column="tmdb_id",
    features=None,
    perplexity=50,
    max_iter=1000,
):
    """
    Visualizes clusters in 2D space using t-SNE for dimensionality reduction.

    Args:
        df (pd.DataFrame): DataFrame containing features and cluster labels.
        cluster_column (str): Column with cluster labels.
        title_column (str): Column with movie titles.
        id_column (str): Column with movie IDs.
        features (list): List of numerical feature column names to include in t-SNE.
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): Number of optimization iterations for t-SNE.

    Returns:
        fig: A Plotly figure object with the t-SNE scatter plot.
    """

    # If no features are specified, use all columns except 'title' and 'id'
    if features is None:
        features = [
            col
            for col in df.columns
            if col not in [title_column, id_column, cluster_column]
        ]

    # Apply t-SNE on the selected feature columns
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
    )
    tsne_result = tsne.fit_transform(df[features])

    # Add t-SNE results to the dataframe
    df["TSNE_1"] = tsne_result[:, 0]
    df["TSNE_2"] = tsne_result[:, 1]

    # Plot the t-SNE scatter plot
    fig = px.scatter(
        df,
        x="TSNE_1",
        y="TSNE_2",
        color=cluster_column,
        hover_data={id_column: True, title_column: True},
        labels={"TSNE_1": "t-SNE Dimension 1", "TSNE_2": "t-SNE Dimension 2"},
        color_discrete_sequence=px.colors.sequential.Agsunset,
    )

    return fig
