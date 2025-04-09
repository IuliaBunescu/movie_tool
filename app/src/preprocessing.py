import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data_for_clustering(df):
    """
    Prepares a DataFrame for clustering by scaling numerical features, encoding categorical features,
    and adding BERT embeddings from the cleaned_overview (using SentenceTransformer).

    Parameters:
    - df: Original DataFrame with movie data.

    Returns:
    - Transformed DataFrame ready for clustering.
    """
    print(f"\n{'='*50}")
    print(f"Started preparing dataset for clustering.")

    # Step 1: Keep tmdb_id and title for reference
    reference_columns = df[["tmdb_id", "title"]]

    # Step 2: Exclude unused columns
    df = df.drop(columns=["imdb_id", "overview"])
    print(f"- Dropped unused columns: 'imdb_id', 'overview'.")

    # Step 3: Handle release_date (extract year)
    df["release_year"] = pd.to_datetime(df["release_date"]).dt.year
    df = df.drop(columns=["release_date"])
    print(
        f"- Extracted 'release_year' from 'release_date' and dropped the 'release_date' column."
    )

    # Step 4: Apply log transformation to 'popularity' to reduce its range
    df["log_popularity"] = np.log1p(df["popularity"])
    print(f"- Applied log transformation to 'popularity' to create 'log_popularity'.")

    # Step 5: Separate features into numerical and categorical
    numerical_features = [
        "vote_average",
        "vote_count",
        "log_popularity",  # kept, but will scale down
        "release_year",
    ]
    categorical_features = ["original_language", "country_of_origin"]
    multivalue_features = ["genres"]

    # Step 6: Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]), columns=numerical_features
    )
    print(f"- Scaled numerical columns: {numerical_features} using StandardScaler.")

    # Apply a small weight to the 'log_popularity' to reduce its impact
    scaled_numerical["log_popularity"] *= 0.1
    print("- Scaled 'log_popularity' by a factor of 0.1 to reduce its impact.")

    # Step 7: Encode single-value categorical features (one-hot encoding)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_categorical = pd.DataFrame(
        encoder.fit_transform(df[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
    )
    print(f"- One-hot encoded categorical features: {categorical_features}.")

    # Step 8: Encode multi-value categorical features (multi-hot encoding)
    def multi_hot_encode(column, unique_values):
        """
        Multi-hot encodes a column with lists of values.
        """
        multi_hot = np.zeros((len(column), len(unique_values)), dtype=int)
        for i, values in enumerate(column):
            for value in values.split(", "):  # Assuming values are comma-separated
                if value in unique_values:
                    multi_hot[i, unique_values.index(value)] = 1
        return pd.DataFrame(
            multi_hot, columns=[f"{column.name}_{val}" for val in unique_values]
        )

    # Get unique values for the genres column
    unique_genres = set(
        genre.strip() for genres in df["genres"] for genre in genres.split(", ")
    )
    encoded_genres = multi_hot_encode(df["genres"], sorted(unique_genres))
    print(
        f"- Multi-hot encoded 'genres' feature with {len(unique_genres)} unique values."
    )

    # Step 9: Generate BERT embeddings using SentenceTransformer for the 'cleaned_overview'
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # or use another model
    embeddings = model.encode(df["cleaned_overview"].tolist(), show_progress_bar=True)
    print("- Generated BERT embeddings for 'cleaned_overview'.")

    # Step 10: Combine all transformed features (numerical, categorical, genres, and BERT embeddings)
    transformed_df = pd.concat(
        [
            reference_columns,
            scaled_numerical,
            encoded_categorical,
            encoded_genres,
            pd.DataFrame(embeddings),
        ],
        axis=1,
    )

    # Step 11: Drop rows with any missing values
    transformed_df = transformed_df.dropna()
    print(f"- Dropped rows with missing values.")

    print(f"\n{'='*50}")
    print(f"Dataset preparation completed.")

    return transformed_df
