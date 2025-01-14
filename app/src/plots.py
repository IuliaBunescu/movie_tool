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
        hole=0.3,  # Optional: creates a donut chart effect
        color_discrete_sequence=px.colors.sequential.Plasma,
    )  # You can change the color palette

    # Show the plot
    return fig
