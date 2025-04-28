import concurrent.futures
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
    Retrieve a set of movie IDs from TMDB, filtered by genre, distributed across decades to ensure
    temporal diversity and minimize recency bias.

    This function attempts to balance the number of results across different decades within the specified
    time range. It prioritizes finding movies that match **all** requested genres (AND search) with at least
    100 votes. If not enough movies are found for a decade, it automatically falls back to a looser search
    where movies matching **any** of the specified genres (OR search) are allowed.

    If the total number of movies across all decades is still below the requested `max_results`, a final
    global OR search (across all years) is used to fill the gap.

    Key Features:
    - Distributes results based on available movie counts per decade.
    - Prioritizes stricter (AND) genre matching before falling back to broader (OR) matching.
    - Automatically handles underfilled decades and ensures filling up to `max_results` if possible.
    - Prefers higher-rated movies by sorting by `vote_average.desc`.

    Args:
        genre_names (list of str): List of genre names (e.g., ["Drama", "Adventure"]) to filter movies by.
        max_results (int): The maximum number of movie IDs to return (default is 1000).
        start_year (int): The start year of the search range (default is 1950).
        end_year (int): The end year of the search range (default is 2024).

    Returns:
        set: A set of unique movie IDs matching the criteria, balanced across decades.

    Example:
        movie_ids = get_movie_ids_by_genres_decade_weighted(["Action", "Adventure"], max_results=500)
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

    # Step 1: Analyze how many movies per decade (using AND)
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

    # Step 3: Fetch movies per decade
    for decade_start, num_to_fetch in decade_allocations.items():
        print(f"Fetching {num_to_fetch} from {decade_start}s...")
        collected_ids = set()
        page = 1

        # --- Try AND search first ---
        while len(collected_ids) < num_to_fetch:
            response = discover.movie(
                with_genres=",".join(map(str, genre_ids)),
                vote_count_gte=100,
                primary_release_date_gte=f"{decade_start}-01-01",
                primary_release_date_lte=f"{min(decade_start + 9, end_year)}-12-31",
                sort_by="vote_average.desc",
                page=page,
            )
            collected_ids.update([movie["id"] for movie in response["results"]])

            if page >= response["total_pages"]:
                break
            page += 1

        # --- If not enough, fallback to OR search ---
        if len(collected_ids) < num_to_fetch:
            print(f"Not enough using AND for {decade_start}s. Trying OR search...")
            page = 1
            while len(collected_ids) < num_to_fetch:
                response = discover.movie(
                    with_genres="|".join(map(str, genre_ids)),
                    vote_count_gte=100,
                    primary_release_date_gte=f"{decade_start}-01-01",
                    primary_release_date_lte=f"{min(decade_start + 9, end_year)}-12-31",
                    sort_by="vote_average.desc",
                    page=page,
                )
                collected_ids.update([movie["id"] for movie in response["results"]])

                if page >= response["total_pages"]:
                    break
                page += 1

        movie_ids.update(list(collected_ids)[:num_to_fetch])

    # Step 4: Global fallback if still under max_results
    if len(movie_ids) < max_results:
        print(
            f"Only collected {len(movie_ids)} movies. Filling remaining globally with OR search..."
        )
        page = 1
        while len(movie_ids) < max_results:
            response = discover.movie(
                with_genres="|".join(map(str, genre_ids)),
                vote_count_gte=100,
                sort_by="vote_average.desc",
                page=page,
            )
            movie_ids.update([movie["id"] for movie in response["results"]])

            if page >= response["total_pages"]:
                break
            page += 1

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
    show_spinner="Getting custom dataset ready",
)
def get_movies_by_genres_from_reference_df(
    reference_df, filter_column="genres", max_results=1000
):
    """
    Use genres from the reference DataFrame to fetch additional movie details.
    Fetches movie details concurrently for improved speed.
    Returns a combined DataFrame.
    """
    print("Extracting genres from reference DataFrame...")
    options = (
        reference_df[filter_column]
        .dropna()
        .apply(lambda x: [genre.strip() for genre in x.split(",")])
        .explode()
        .unique()
    )

    print(f"Found {len(options)} unique genres: {options}")

    print("Fetching movie IDs using decade-weighted search...")
    movie_ids = get_movie_ids_by_genres_decade_weighted(options, max_results)
    print(f"Retrieved {len(movie_ids)} movie IDs.")

    all_movie_data = []

    print(f"Fetching movie details concurrently (target: {len(movie_ids)} movies)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_movie_id = {
            executor.submit(get_movie_details_by_id, movie_id): movie_id
            for movie_id in movie_ids
        }

        for i, future in enumerate(
            concurrent.futures.as_completed(future_to_movie_id), 1
        ):
            movie_id = future_to_movie_id[future]
            try:
                movie_data = future.result()
                if movie_data:
                    all_movie_data.append(movie_data)
            except Exception as e:
                print(f"Error fetching movie ID {movie_id}: {e}")

            if i % 100 == 0:
                print(f"Fetched details for {i} movies...")

    print("Converting fetched movie data to DataFrame...")
    movie_details_df = pd.DataFrame(all_movie_data)

    print("Concatenating with the reference DataFrame and dropping duplicates...")
    full_df = pd.concat([reference_df, movie_details_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset="tmdb_id")

    print(f"Final DataFrame contains {len(full_df)} unique movies.")

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
