import pandas as pd
import plotly.express as px


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
        .dropna()  # Remove NaN values
        .apply(
            lambda x: x.split(", ") if isinstance(x, str) else x
        )  # Split strings by commas
        .explode()  # Flatten lists into rows
        .value_counts(normalize=True)  # Calculate percentages
        .reset_index()  # Reset index for easier plotting
    )
    value_counts.columns = ["Value", "Percentage"]  # Rename columns for clarity
    value_counts["Percentage"] = (value_counts["Percentage"] * 100).round(
        2
    )  # Convert to percentage format and round

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
            titlefont=dict(size=16),  # Font size for X-axis title
            tickfont=dict(size=14),  # Font size for X-axis tick labels
        ),
        yaxis=dict(
            titlefont=dict(size=16),  # Font size for Y-axis title
            tickfont=dict(size=14),  # Font size for Y-axis tick labels
        ),
        hoverlabel=dict(font=dict(size=14)),  # Font size for hover labels
        margin=dict(l=0, r=0, t=0, b=0),  # Set all margins to 0
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


def plot_country_distribution_pie(df, country_column, cap_percentage=3):
    """
    Plots a pie chart showing the distribution of country counts as percentages.
    If there are countries with less than `cap_percentage` of the total, they are grouped into "Others".
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

    # Calculate the total count
    total_count = country_counts["Count"].sum()

    # Calculate the percentage of each country
    country_counts["Percentage"] = (country_counts["Count"] / total_count) * 100

    # Create a new category "Others" for countries with less than `cap_percentage`
    small_countries = country_counts[country_counts["Percentage"] < cap_percentage]
    others_count = small_countries["Count"].sum()

    # Filter out the small countries and keep only the larger countries
    country_counts = country_counts[country_counts["Percentage"] >= cap_percentage]

    # Add the "Others" category using pd.concat
    if others_count > 0:
        others_df = pd.DataFrame(
            {
                "Country": ["Others (<3%)"],
                "Count": [others_count],
                "Percentage": [(others_count / total_count) * 100],
            }
        )
        country_counts = pd.concat([country_counts, others_df], ignore_index=True)

    # Plot the pie chart using Plotly
    fig = px.pie(
        country_counts,
        names="Country",
        values="Count",
        hole=0.5,  # Optional: creates a donut chart effect
        color_discrete_sequence=px.colors.sequential.Magenta,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(font=dict(size=14)),
    )

    # Show the plot
    return fig


def plot_movies_by_year(df, date_column):
    """
    Plots a line plot showing the number of movies released each year.
    Assumes that `df` has a `release_date` column with date values in 'YYYY-MM-DD' format.
    """
    # Convert the release_date column to datetime format (if it's not already)
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract the year from the release_date
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
            titlefont=dict(size=16),  # Font size for X-axis title
            tickfont=dict(size=14),  # Font size for X-axis tick labels
        ),
        yaxis=dict(
            titlefont=dict(size=16),  # Font size for Y-axis title
            tickfont=dict(size=14),  # Font size for Y-axis tick labels
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
        color_continuous_scale=px.colors.sequential.Magenta,
        hover_data=[popularity_column, vote_column],
        marginal_x="histogram",
    )  # Show additional info on hover

    # Update the layout to adjust the appearance
    fig.update_layout(
        xaxis_title="Popularity Score (0-100)",
        yaxis_title="Vote Average (0-10)",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
