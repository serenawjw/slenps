"""
This module provides utilities for embedding documents, saving and loading embeddings,
and performing dimensionality reduction on embeddings.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

from slenps.eclusters.embedding_models import EmbeddingModelRegistry, BaseEmbeddingModel
from slenps.utils import check_memory_usage

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2", mode: Optional[str] = None, **kwargs
) -> BaseEmbeddingModel:
    """
    Load an embedding model based on the mode and model name.

    Args:
        model_name: Name of the model to load.
        mode: Type of model to load ('huggingface', 'tfidf', 'word2vec', 'doc2vec').
        **kwargs: Additional keyword arguments for the model.

    Returns:
        A model with an .encode method or equivalent functionality.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if mode == "huggingface":
        return EmbeddingModelRegistry.get_registry("SbertEM")(model_name, **kwargs)
    elif mode is None:
        return EmbeddingModelRegistry.get_registry(model_name)(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def embed_and_save(
    model: BaseEmbeddingModel,
    documents: Union[np.ndarray, pd.Series, List[str]],
    output_path: Union[Path, str],
) -> None:
    """
    Embeds documents using the embedding model and saves the embeddings along with the documents.

    Args:
        model: The embedding model to use.
        documents: The documents to be embedded.
        output_path: Path where the pickle file will be stored.

    Raises:
        ValueError: If the output file already exists to prevent overwriting of data.
    """
    if not isinstance(documents, np.ndarray):
        documents = np.array(documents)
    documents = np.unique(documents)

    embeddings = model.encode(documents)

    if os.path.exists(output_path):
        raise ValueError(f"File already exists at {output_path}")

    with open(output_path, "wb") as file:
        pickle.dump((embeddings, documents), file)
        logger.info(f"Embeddings and documents saved to {output_path}")


def get_data_from_paths(
    filepaths: List[Union[Path, str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and documents from multiple pickle files.

    Args:
        filepaths: List of paths to pickle files containing embeddings and documents.

    Returns:
        A tuple containing concatenated embeddings and documents.

    Raises:
        ValueError: If a specified file does not exist.
    """
    embeddings_list = []
    documents_list = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            raise ValueError(f"No file named {filepath}!")

        with open(filepath, "rb") as file:
            embeddings, documents = pickle.load(file)
            embeddings_list.append(embeddings)
            documents_list.append(documents)

    concatenated_embeddings = (
        np.vstack(embeddings_list) if embeddings_list else np.array([])
    )
    concatenated_documents = (
        np.concatenate(documents_list) if documents_list else np.array([])
    )

    check_memory_usage(concatenated_documents, concatenated_embeddings)

    return concatenated_embeddings, concatenated_documents


def sample(
    embeddings: np.ndarray, documents: np.ndarray, percent: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly samples a percentage of embeddings and documents.

    Args:
        embeddings: Array of embeddings.
        documents: Array of documents.
        percent: Fraction of the data to sample (between 0 and 1).

    Returns:
        A tuple containing sampled embeddings and documents.

    Raises:
        ValueError: If 'percent' is not within the range (0, 1).
    """
    if not 0 < percent < 1:
        raise ValueError("Percent must be between 0 and 1")

    sampled_embeddings, _, sampled_documents, _ = train_test_split(
        embeddings, documents, train_size=percent, random_state=42
    )

    return sampled_embeddings, sampled_documents


def reduce_dimension(
    embeddings: np.ndarray, model_name: str = "pca", n_dim: int = 2
) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings using PCA or UMAP.

    Args:
        embeddings: Array of embeddings.
        model_name: Dimensionality reduction model to use ('pca' or 'umap').
        n_dim: Number of dimensions to reduce to.

    Returns:
        Array of reduced embeddings.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    if model_name == "pca":
        embeddings_standardized = StandardScaler().fit_transform(embeddings)
        model = PCA(n_components=n_dim)
        embeddings_reduced = model.fit_transform(embeddings_standardized)
    elif model_name == "umap":
        model = umap.UMAP(n_components=n_dim)
        embeddings_reduced = model.fit_transform(embeddings)
    else:
        raise ValueError("Model name must be 'pca' or 'umap'")

    return embeddings_reduced
