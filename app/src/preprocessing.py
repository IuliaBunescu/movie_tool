import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import CrossEncoder
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@st.cache_data(ttl=datetime.timedelta(hours=12), show_spinner=False)
def prepare_data_for_clustering(df, reference_tmdb_id):
    """
    Prepares DataFrames for clustering at different levels of detail:
    1. Basic: TMDB ID, title, cosine similarity
    2. Numerical: Adds scaled numerical features
    3. Full: Adds categorical features (genres, country_of_origin, original_language)

    Parameters:
    - df: Original DataFrame with movie data.
    - reference_tmdb_id: TMDB ID of the reference movie.

    Returns:
    - df_basic: Basic DataFrame (reference columns + similarity)
    - df_numerical: Adds numerical features
    - df_full: Adds categorical features
    """

    st.markdown("### 🛠 Preparing data for clustering...")
    print(f"\n{'='*50}")
    print(f"Started preparing dataset for clustering.")

    # Save reference columns
    reference_columns = df[["tmdb_id", "title"]].copy()

    # Save categorical features (not encoded)
    categorical_cols = df[["country_of_origin", "original_language"]].copy()

    # Drop irrelevant columns
    df = df.drop(
        columns=["imdb_id", "overview", "genres", "release_date", "country_of_origin"]
    )
    st.markdown("- Dropped columns: `imdb_id`, `overview`, `genres`, `release_date`")
    print(f"- Dropped unused columns.")

    # Feature engineering
    df["log_popularity"] = np.log1p(df["popularity"])
    st.markdown("- Log-transformed `popularity` → `log_popularity`")
    print(f"- Applied log transformation to 'popularity'.")

    numerical_features = ["vote_average", "vote_count", "log_popularity"]

    # Scaling numerical features
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    scaled = standard_scaler.fit_transform(df[numerical_features])
    scaled = minmax_scaler.fit_transform(scaled)

    scaled_numerical = pd.DataFrame(scaled, columns=numerical_features)
    scaled_numerical["log_popularity"] *= 0.1  # downweight
    st.markdown("- Scaled numerical features and down-weighted `log_popularity` by 90%")
    print(f"- Scaled numerical features.")

    # Similarity
    st.markdown(
        "<span style='color:gray'>Computing pairwise similarity using a RoBERTa-based cross-encoder on <code>cleaned_overview</code>...</span>",
        unsafe_allow_html=True,
    )
    model = CrossEncoder("cross-encoder/stsb-roberta-base")

    try:
        ref_index = df[df["tmdb_id"] == reference_tmdb_id].index[0]
    except IndexError:
        raise ValueError(
            f"Reference TMDB ID {reference_tmdb_id} not found in the dataset."
        )

    reference_text = df.loc[ref_index, "cleaned_overview"]

    # Create sentence pairs: (reference, other)
    pairs = [(reference_text, other) for other in df["cleaned_overview"].tolist()]

    # Predict similarity scores
    similarity_scores = model.predict(pairs, show_progress_bar=True)
    similarity_df = pd.DataFrame(similarity_scores, columns=["similarity_to_reference"])
    st.markdown(
        f"- Predicted similarity to reference movie (TMDB ID: `{reference_tmdb_id}`)"
    )
    print(f"- Predicted cosine similarity based on reference movie.")

    # --- Assemble outputs ---
    df_basic = pd.concat([reference_columns, similarity_df], axis=1)
    df_numerical = pd.concat([df_basic, scaled_numerical], axis=1)
    df_full = pd.concat([df_numerical, categorical_cols], axis=1)

    # Remove any missing values
    df_basic = df_basic.dropna()
    df_numerical = df_numerical.dropna()
    df_full = df_full.dropna()

    st.success("✅ Data is ready for clustering.")
    print(f"\n{'='*50}")
    print(
        f"Dataset preparation completed using TMDB ID {reference_tmdb_id} as reference."
    )

    return df_basic, df_numerical, df_full


def apply_pca(
    df,
    features,
    n_components=50,
    explained_variance_threshold=0.95,
    id_column="tmdb_id",
    title_column="title",
):
    """
    Applies PCA to reduce dimensionality of the selected features and reattaches the 'title' and 'tmdb_id' columns.

    Args:
        df (pd.DataFrame): DataFrame containing the feature data.
        features (list): List of column names to apply PCA on.
        n_components (int): Maximum number of principal components to keep.
        explained_variance_threshold (float): Minimum cumulative explained variance to retain.
        id_column (str): Name of the ID column to reattach.
        title_column (str): Name of the title column to reattach.

    Returns:
        reduced_df (pd.DataFrame): DataFrame with PCA components as features, plus 'tmdb_id' and 'title' columns.
    """
    st.markdown("### 📉 Applying PCA for dimensionality reduction...")

    # Make sure column names and feature names are strings
    df.columns = df.columns.astype(str)
    features = [str(feature) for feature in features]

    reference_columns = (
        df[[id_column, title_column]]
        if id_column in df.columns and title_column in df.columns
        else None
    )

    # Fit PCA
    with st.spinner("Fitting PCA model..."):
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df[features])

    # Calculate optimal number of components based on explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    optimal_components = (cumulative_variance < explained_variance_threshold).sum() + 1

    if optimal_components == 0:
        optimal_components = 1  # Ensure at least one component is kept

    explained_percent = cumulative_variance[optimal_components - 1] * 100
    st.markdown(
        f"- Retained **{optimal_components}** components "
        f"covering **{explained_percent:.1f}%** of variance"
    )
    print(
        f"PCA: Retained {optimal_components} components explaining ~{explained_percent:.2f}% of variance."
    )

    # Create reduced feature DataFrame
    reduced_df = pd.DataFrame(
        principal_components[:, :optimal_components],
        columns=[f"PCA_{i+1}" for i in range(optimal_components)],
        index=df.index,
    )

    # Reattach reference columns if present
    if reference_columns is not None:
        reduced_df = pd.concat([reference_columns, reduced_df], axis=1)

    st.success("✅ PCA dimensionality reduction complete.")
    return reduced_df
