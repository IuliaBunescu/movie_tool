import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@st.cache_data(ttl=datetime.timedelta(hours=12), show_spinner=False)
def prepare_data_for_clustering(df):
    """
    Prepares a DataFrame for clustering by scaling numerical features, encoding categorical features,
    and adding BERT embeddings from the cleaned_overview (using SentenceTransformer).

    Parameters:
    - df: Original DataFrame with movie data.

    Returns:
    - Transformed DataFrame ready for clustering (with all features).
    - DataFrame with movie id, name, and embeddings.
    """

    st.markdown("### 🛠 Preparing data for clustering...")
    print(f"\n{'='*50}\nStarted preparing dataset for clustering.")

    reference_columns = df[["tmdb_id", "title"]]

    df = df.drop(
        columns=["imdb_id", "overview", "genres", "release_date", "country_of_origin"]
    )
    st.markdown(
        "- Dropped columns: `imdb_id`, `overview`, `genres`, `release_date`, `country_of_origin`"
    )
    print("- Dropped unused columns.")

    df["log_popularity"] = np.log1p(df["popularity"])
    st.markdown("- Log-transformed `popularity`")
    print("- Applied log transformation to 'popularity'.")

    numerical_features = ["vote_average", "vote_count", "log_popularity"]
    categorical_features = ["original_language"]

    scaler = StandardScaler()
    scaled_numerical = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]), columns=numerical_features
    )
    st.markdown(
        "- Scaled numerical features: "
        + ", ".join([f"`{feature}`" for feature in numerical_features])
    )

    print("- Scaled numerical columns.")

    scaled_numerical["log_popularity"] *= 0.1
    st.markdown("- Down-weighted `log_popularity` by 90%.")
    print("- Adjusted 'log_popularity' by ×0.1.")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_categorical = pd.DataFrame(
        encoder.fit_transform(df[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
    )
    st.markdown(f"- Encoded categorical features: `{', '.join(categorical_features)}`")

    print("- One-hot encoded categorical features.")

    st.markdown(
        "<span style='color:gray'>Generating BERT embeddings on `overview` feature...</span>",
        unsafe_allow_html=True,
    )
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(df["cleaned_overview"].tolist(), show_progress_bar=True)
    st.markdown("- BERT embeddings added")
    print("- Generated BERT embeddings.")

    transformed_df = pd.concat(
        [
            reference_columns,
            scaled_numerical,
            encoded_categorical,
            pd.DataFrame(embeddings),
        ],
        axis=1,
    )

    transformed_df = transformed_df.dropna()
    st.markdown("- Removed rows with missing values")
    print("- Dropped rows with missing values.")

    embeddings_df = pd.concat(
        [reference_columns, pd.DataFrame(embeddings)],
        axis=1,
    )

    st.success("✅ Data is ready for clustering.")
    print("Dataset preparation completed.\n" + "=" * 50)

    return transformed_df, embeddings_df


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
