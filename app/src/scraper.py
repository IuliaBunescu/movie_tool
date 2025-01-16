import json

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
