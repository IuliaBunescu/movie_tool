import pandas as pd


def create_local_imdb_database(output_file="imdb_movies.csv"):
    # Load datasets
    print("Loading datasets...")
    basics = pd.read_csv(
        r"C:\Users\iulia\Documents\Projects\movie_recommender_tool\movie_tool\data\title.basics.tsv",
        sep="\t",
        na_values="\\N",
    )
    ratings = pd.read_csv(
        r"C:\Users\iulia\Documents\Projects\movie_recommender_tool\movie_tool\data\title.ratings.tsv",
        sep="\t",
        na_values="\\N",
    )

    # Filter for movies only (titleType = 'movie')
    print("Filtering movies...")
    movies = basics[basics["titleType"] == "movie"]

    # Select relevant columns from basics
    movies = movies[["tconst", "primaryTitle", "originalTitle", "startYear", "genres"]]

    # Drop rows where 'genres' or 'startYear' is null or NaN
    print("Dropping rows with null 'genres' or 'startYear'...")
    movies = movies.dropna(subset=["genres", "startYear"])

    # Convert startYear to integer
    print("Processing release_year...")
    movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce").astype(
        int
    )

    # Merge with ratings
    print("Merging with ratings...")
    movies = movies.merge(ratings, on="tconst", how="left")

    # Drop rows where 'vote_average' or 'vote_count' is null or NaN
    print("Dropping rows with null 'vote_average' or 'vote_count'...")
    movies = movies.dropna(subset=["vote_average", "vote_count"])

    # Rename columns after merging
    print("Renaming columns...")
    movies.rename(
        columns={
            "tconst": "imdb_id",
            "primaryTitle": "title",
            "originalTitle": "originalTitle",
            "startYear": "release_year",
            "averageRating": "vote_average",
            "numVotes": "vote_count",
        },
        inplace=True,
    )

    # Final cleanup
    movies = movies.drop_duplicates(subset=["imdb_id"])

    # Save to file
    print(f"Saving to {output_file}...")
    movies.to_csv(output_file, index=False)
    print("Database created successfully.")


create_local_imdb_database("imdb_movies.csv")
