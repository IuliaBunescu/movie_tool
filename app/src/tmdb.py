import datetime
import re

import pandas as pd
import streamlit as st
import tmdbsimple as tmdb


@st.cache_data(
    ttl=datetime.timedelta(hours=12), show_spinner="Getting movie data ready..."
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
    Ensures movies have at least 100 vote counts.

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

    # Fetch movies using AND operation (comma-separated genres) with minimum vote count filter
    while len(movie_ids) < max_results:
        response = discover.movie(
            with_genres=",".join(map(str, genre_ids)),
            vote_count_gte=100,  # Minimum vote count filter
            page=page,
        )

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

        # Fetch movies using OR operation (pipe-separated genres) with minimum vote count filter
        while len(movie_ids) < max_results:
            response = discover.movie(
                with_genres="|".join(map(str, genre_ids)),
                vote_count_gte=100,  # Minimum vote count filter
                page=page,
            )

            # Add movie IDs from the current page
            movie_ids.update([movie["id"] for movie in response["results"]])

            # Break if we've fetched all available pages
            if page >= response["total_pages"]:
                break

            page += 1

    # Return the movie IDs as a set to ensure uniqueness
    return movie_ids


def get_movie_ids_by_genres_decade_weighted(
    genre_names, max_results=1000, start_year=1950, end_year=2024
):
    """
    Retrieve a set of movie IDs from TMDB, filtered by genre, with results distributed across decades
    to reduce recency bias and improve temporal diversity.

    This function divides the specified time range into decades (e.g., 1990s, 2000s), dynamically assesses
    how many qualifying movies exist in each decade, and proportionally allocates the number of results
    to fetch from each based on that availability.

    It prioritizes quality by default (using vote_average sorting), ensures movies have at least 100 votes,
    and fills in with additional results if the requested amount is not met.

    Args:
        genre_names (list of str): List of genre names (e.g., ["Drama", "Adventure"]).
        max_results (int): Total number of movie IDs to return (default is 1000).
        start_year (int): Start year of the search range (default is 1950).
        end_year (int): End year of the search range (default is 2024).

    Returns:
        set: A set of unique movie IDs distributed across decades.
    """

    genre_list = tmdb.Genres().movie_list()["genres"]
    genre_ids = [
        genre["id"]
        for genre in genre_list
        if genre["name"].lower() in [name.lower() for name in genre_names]
    ]

    if not genre_ids:
        print(f"One or more genres from {genre_names} not found.")
        return set()

    discover = tmdb.Discover()
    movie_ids = set()

    # Define decades
    decades = list(range(start_year, end_year + 1, 10))
    decade_totals = {}

    # Step 1: Determine how many pages of movies exist per decade (proxy for availability)
    print("Analyzing decade availability...")
    for decade_start in decades:
        response = discover.movie(
            with_genres=",".join(map(str, genre_ids)),
            vote_count_gte=100,
            primary_release_date_gte=f"{decade_start}-01-01",
            primary_release_date_lte=f"{min(decade_start + 9, end_year)}-12-31",
            page=1,
        )
        total_results = response.get("total_results", 0)
        decade_totals[decade_start] = total_results

    # Step 2: Normalize weights
    total_available = sum(decade_totals.values())
    if total_available == 0:
        print("No movies found across selected decades.")
        return set()

    decade_allocations = {
        decade: max(1, int((count / total_available) * max_results))
        for decade, count in decade_totals.items()
    }

    # Step 3: Collect results per decade
    for decade_start, num_to_fetch in decade_allocations.items():
        print(f"Fetching {num_to_fetch} from {decade_start}s...")
        page = 1
        collected_ids = set()

        while len(collected_ids) < num_to_fetch:
            response = discover.movie(
                with_genres=",".join(map(str, genre_ids)),
                vote_count_gte=100,
                primary_release_date_gte=f"{decade_start}-01-01",
                primary_release_date_lte=f"{min(decade_start + 9, end_year)}-12-31",
                sort_by="vote_average.desc",  # optional: prioritize quality
                page=page,
            )

            collected_ids.update([movie["id"] for movie in response["results"]])

            if page >= response["total_pages"]:
                break
            page += 1

        movie_ids.update(list(collected_ids)[:num_to_fetch])

    # Pad if underfilled
    if len(movie_ids) < max_results:
        print(
            f"Only collected {len(movie_ids)} movies. Filling remaining with OR search."
        )
        fallback = get_movie_ids_by_genres(
            genre_names, max_results=max_results - len(movie_ids)
        )
        movie_ids.update(fallback)

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
    ttl=datetime.timedelta(hours=12),
    show_spinner="Getting custom dataset ready (can take up to 3 minutes)...",
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
    movie_ids = get_movie_ids_by_genres_decade_weighted(genres, max_results)

    print(f"\n{'='*50}")
    print(f"Gathered {len(movie_ids)} movie ids with similar genres.")
    print(f"{'='*50}\n")

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


def extract_tmdb_id(tmdb_url):
    """
    Extracts the TMDB movie or TV show ID from a given TMDB URL and retrieves movie details.

    Args:
    tmdb_url (str): The URL of a TMDB movie or TV show.

    Returns:
    dict: A dictionary containing:
        - 'found_movie_data_flag' (bool): True if a movie ID was found, False otherwise.
        - 'ref_movie_df' (DataFrame or None): A DataFrame with movie details if available, None otherwise.
    """
    # Extract the movie or TV show ID from the URL
    match = re.search(r"(movie|tv)/(\d+)", tmdb_url)
    movie_id = match.group(2) if match else None

    # Initialize the response dictionary
    result = {"found_movie_data_flag": False, "ref_movie_df": None}

    if movie_id:
        # Get movie details using the extracted ID
        ref_movie_dic = get_movie_details_by_id(int(movie_id))
        if ref_movie_dic:
            # Convert movie details to DataFrame if available
            ref_movie_df = pd.DataFrame([ref_movie_dic])
            result["found_movie_data_flag"] = True
            result["ref_movie_df"] = ref_movie_df

    return result
