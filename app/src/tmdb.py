import datetime
import os

import pandas as pd
import streamlit as st
import tmdbsimple as tmdb
from helper import get_timestamp

tmdb.API_KEY = os.getenv("TMDB_API_KEY")


@st.cache_data(
    ttl=datetime.timedelta(hours=6), show_spinner="Getting movie data ready..."
)
def search_first_movie_by_title_and_year_tmdb(movie_title, release_year=None):
    """
    Search for the first movie by title and optionally release year using the tmdbsimple wrapper.
    Fetch additional details for the first movie found.
    Return a pandas DataFrame with the movie data.
    """
    if not movie_title:
        return pd.DataFrame()

    search = tmdb.Search()
    query_params = {"query": movie_title}

    # If release year is provided, add it to the query
    if release_year:
        query_params["primary_release_year"] = release_year

    # Perform the search
    response = search.movie(**query_params)

    if not search.results:
        print("No results found for the search query.")
        return pd.DataFrame()  # Return an empty DataFrame if no results are found

    # Get the first movie from the results
    first_result = search.results[0]

    # Extract basic information
    movie = {
        "tmdb_id": first_result.get("id"),  # TMDB movie ID
        "title": first_result.get("title"),
        "overview": first_result.get("overview"),
        "release_date": first_result.get("release_date"),
        "vote_average": first_result.get("vote_average"),
        "vote_count": first_result.get("vote_count"),
        "popularity": first_result.get("popularity"),
    }

    # Fetch additional details using the movie ID
    movie_details = tmdb.Movies(first_result.get("id")).info()

    # Extract additional details
    movie["imdb_id"] = movie_details.get("imdb_id")
    movie["original_language"] = movie_details.get("original_language")
    movie["country_of_origin"] = ", ".join(
        [
            country.get("name")
            for country in movie_details.get("production_countries", [])
        ]
    )
    movie["genres"] = ", ".join(
        [genre.get("name") for genre in movie_details.get("genres", [])]
    )

    # Create and return a pandas DataFrame
    df = pd.DataFrame([movie])
    return df


def get_movie_ids_by_genres(genre_names, max_results=1000):
    """
    Get a list of movie IDs based on multiple specified genre names.
    First try an AND search (movies must belong to all genres),
    and if not enough results are found, switch to an OR search (movies can belong to any genre).
    Args:
        genre_names (list of str): A list of genres to search for (e.g., ["Action", "Animation"]).
        max_results (int): The maximum number of movie IDs to return.
    Returns:
        set: A set of unique movie IDs matching the specified genres.
    """
    # Fetch all genres and find the genre IDs for the given genre names
    genre_list = tmdb.Genres().movie_list()["genres"]
    genre_ids = [
        genre["id"]
        for genre in genre_list
        if genre["name"].lower() in [name.lower() for name in genre_names]
    ]

    # print(f"\n{'='*50}")
    # print(f"[{get_timestamp()}] Started gathering movie ids for genres: {genre_names} ")
    # print(f"{'='*50}\n")

    if not genre_ids:
        print(f"One or more genres from {genre_names} not found.")
        return set()

    # Use the Discover endpoint to search for movies by multiple genres (AND operation using commas)
    discover = tmdb.Discover()
    movie_ids = set()  # Using set to ensure uniqueness
    page = 1

    # Fetch movies using AND operation (comma-separated genres)
    while len(movie_ids) < max_results:
        response = discover.movie(with_genres=",".join(map(str, genre_ids)), page=page)

        # Add movie IDs from the current page
        movie_ids.update([movie["id"] for movie in response["results"]])

        # Break if we've fetched all available pages
        if page >= response["total_pages"]:
            break

        page += 1

    # If we have fewer than the requested number of movies, switch to the OR operation
    if len(movie_ids) < max_results:
        print("Not enough movies found using AND operation. Switching to OR operation.")
        movie_ids = set()  # Reset movie_ids
        page = 1

        # Fetch movies using OR operation (pipe-separated genres)
        while len(movie_ids) < max_results:
            response = discover.movie(
                with_genres="|".join(map(str, genre_ids)), page=page
            )

            # Add movie IDs from the current page
            movie_ids.update([movie["id"] for movie in response["results"]])

            # Break if we've fetched all available pages
            if page >= response["total_pages"]:
                break

            page += 1

    # Return the movie IDs as a set to ensure uniqueness
    return movie_ids


def get_movie_details_by_id(movie_id):
    """
    Fetch the movie details by its TMDB ID.
    """
    movie_details = tmdb.Movies(movie_id).info()
    movie_data = {
        "tmdb_id": movie_id,
        "title": movie_details.get("title"),
        "overview": movie_details.get("overview"),
        "release_date": movie_details.get("release_date"),
        "vote_average": movie_details.get("vote_average"),
        "vote_count": movie_details.get("vote_count"),
        "popularity": movie_details.get("popularity"),
        "imdb_id": movie_details.get("imdb_id"),
        "original_language": movie_details.get("original_language"),
        "country_of_origin": ", ".join(
            [
                country.get("name")
                for country in movie_details.get("production_countries", [])
            ]
        ),
        "genres": ", ".join(
            [genre.get("name") for genre in movie_details.get("genres", [])]
        ),
    }
    return movie_data


@st.cache_data(
    ttl=datetime.timedelta(hours=6), show_spinner="Getting custom data ready..."
)
def get_movies_by_genre_from_reference_df(
    reference_df, filter_column="genres", max_results=1000
):
    """
    Use the genres from the reference DataFrame to get movie IDs and fetch movie details.
    Returns a DataFrame with movie details concatenated to the reference DataFrame.
    """
    # Choose the genres based on the reference DataFrame (assumes `genre_column` contains genre information)

    genres = (
        reference_df[filter_column]
        .dropna()
        .apply(lambda x: [genre.strip() for genre in x.split(",")])
        .explode()
        .unique()
    )
    # Get movie IDs by using the genres from the reference DataFrame
    movie_ids = get_movie_ids_by_genres(genres, max_results)

    all_movie_data = []

    # For each movie ID, get the movie details and store them
    for movie_id in movie_ids:
        movie_data = get_movie_details_by_id(movie_id)
        all_movie_data.append(movie_data)

    # Convert the list of movie data to a DataFrame
    movie_details_df = pd.DataFrame(all_movie_data)

    full_df = pd.DataFrame()
    # Concatenate the new movie details DataFrame with the reference DataFrame
    full_df = pd.concat([reference_df, movie_details_df], ignore_index=True)

    full_df = full_df.drop_duplicates(subset="tmdb_id")

    return full_df
