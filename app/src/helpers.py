import os

import pandas as pd
import tmdbsimple as tmdb

tmdb.API_KEY = os.getenv("TMDB_API_KEY")


def get_reference_from_url():
    """
    Get movie data from a URL
    """
    ref_movie_df = pd.DataFrame()
    found_movie_data_flag = True
    res = {"ref_movie_df": ref_movie_df, "found_movie_data_flag": found_movie_data_flag}
    return res


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
        "title": first_result.get("title"),
        "overview": first_result.get("overview"),
        "release_date": first_result.get("release_date"),
        "vote_average": first_result.get("vote_average"),
        "vote_count": first_result.get("vote_count"),
        "popularity": first_result.get("popularity"),
        "tmdb_id": first_result.get("id"),  # TMDB movie ID
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
