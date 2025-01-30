import json

import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_imdb_url(imdb_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(imdb_url, headers=headers)

    if response.status_code == 403:
        print("Access denied. IMDb is blocking the request.")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract JSON-LD data
    json_script = soup.find("script", type="application/ld+json")
    if not json_script:
        print("No structured JSON data found on the page.")
        return None

    movie_data = json.loads(json_script.string)

    # Extract fields from JSON-LD
    title = movie_data.get("name", "N/A")
    description = movie_data.get("description", "N/A")
    release_date = movie_data.get("datePublished", "N/A")
    rating = movie_data.get("aggregateRating", {}).get("ratingValue", "N/A")
    vote_count = movie_data.get("aggregateRating", {}).get("ratingCount", "N/A")
    genres = movie_data.get("genre", [])

    # Extract additional fields directly from the page
    imdb_id = imdb_url.split("/")[-2]  # Extract IMDb ID from the URL
    original_language = (
        soup.find("li", {"data-testid": "title-details-languages"})
        .find("a")
        .text.strip()
        if soup.find("li", {"data-testid": "title-details-languages"})
        else "N/A"
    )
    country_of_origin = (
        [
            a.text.strip()
            for a in soup.find("li", {"data-testid": "title-details-origin"}).find_all(
                "a"
            )
        ]
        if soup.find("li", {"data-testid": "title-details-origin"})
        else "N/A"
    )

    # Return extracted data
    return {
        "title": title,
        "description": description,
        "release_date": release_date,
        "rating": rating,
        "vote_count": vote_count,
        "genres": genres,
        "imdb_id": imdb_id,
        "original_language": original_language,
        "country_of_origin": country_of_origin,
    }


def get_reference_from_url(imdb_url):
    """
    Scrapes IMDb movie details and returns a dataframe with the movie data.

    Args:
    imdb_url (str): The IMDb URL of the movie to scrape.

    Returns:
    dict: A dictionary containing the dataframe and a flag indicating if the data was found.
    """
    movie_details = scrape_imdb_url(imdb_url)

    # Check if data was found
    if movie_details:
        # Transform genres and country_of_origin into comma-separated strings
        genres = movie_details.get("genres", [])
        if isinstance(genres, list):
            genres = ", ".join(genres)
        else:
            genres = "N/A"

        country_of_origin = movie_details.get("country_of_origin", [])
        if isinstance(country_of_origin, list):
            country_of_origin = ", ".join(country_of_origin)
        else:
            country_of_origin = "N/A"

        # Extract the release year from the release_date
        release_date = movie_details.get("release_date", "N/A")
        if release_date != "N/A":
            # Try to extract year if the release_date is a string that contains the year
            try:
                release_year = int(pd.to_datetime(release_date).year)
            except Exception as e:
                release_year = "N/A"  # In case of an error (invalid date format)
        else:
            release_year = "N/A"

        # Create DataFrame
        ref_movie_df = pd.DataFrame(
            [
                {
                    "imdb_id": movie_details.get("imdb_id", "N/A"),
                    "title": movie_details.get("title", "N/A"),
                    "release_year": release_year,
                    "vote_average": movie_details.get("rating", "N/A"),
                    "vote_count": movie_details.get("vote_count", "N/A"),
                    "genres": genres,
                }
            ]
        )

        found_movie_data_flag = True
    else:
        ref_movie_df = pd.DataFrame()
        found_movie_data_flag = False

    return {
        "ref_movie_df": ref_movie_df,
        "found_movie_data_flag": found_movie_data_flag,
    }
