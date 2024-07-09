"""
This script provides utilities for clustering embeddings using various algorithms and evaluating their performance.
It includes functions for clustering, loading models, finding the best algorithm, and sampling documents from clusters.
"""

import os
import csv
from typing import Union, List, Tuple, Dict, Optional, Any
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    Birch,
)
from sklearn.base import ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances


def cluster(
    embeddings: np.ndarray,
    model: ClusterMixin,
    metrics: List[str] = ["dbs"],
    distance: str = "euclidean",
    return_model: bool = False,
) -> Union[
    Tuple[np.ndarray, Dict[str, float]],
    Tuple[np.ndarray, Dict[str, float], ClusterMixin],
]:
    """
    Fits a clustering model to the embeddings and calculates specified metrics.

    Args:
        embeddings: Array of embeddings.
        model: A clustering model with the method .fit_predict.
        metrics: List of metric names to calculate.
        distance: Distance measure to use for silhouette score.
        return_model: Whether to return the fitted model.

    Returns:
        Tuple containing labels, calculated metrics, and optionally the fitted model.

    Raises:
        ValueError: If only one label is predicted.
    """
    labels = model.fit_predict(embeddings)
    if len(np.unique(labels)) <= 1:
        raise ValueError(
            f"Only {len(np.unique(labels))} labels are predicted. Increase data size or reduce cluster number."
        )

    metric_funcs = {
        "dbs": davies_bouldin_score,
        "silhouette": lambda x, labels: silhouette_score(x, labels, metric=distance),
        "calinski": calinski_harabasz_score,
    }

    calculated_metrics = {
        metric: metric_funcs[metric](embeddings, labels)
        for metric in metrics
        if metric in metric_funcs
    }

    return (
        (labels, calculated_metrics, model)
        if return_model
        else (labels, calculated_metrics)
    )


def get_clustering_model_dict() -> Dict[str, ClusterMixin]:
    """
    Returns a dictionary of clustering models.

    Returns:
        A dictionary mapping model names to model instances.
    """
    return {
        "kmeans": KMeans(n_init="auto"),
        "affinity_propagation": AffinityPropagation(),
        "mean_shift": MeanShift(),
        "spectral_clustering": SpectralClustering(),
        "agglomerative_clustering": AgglomerativeClustering(),
        "birch": Birch(threshold=0.2),
    }


def load_clustering_model(
    model_name: str, model_dict: Optional[Dict[str, ClusterMixin]] = None
) -> ClusterMixin:
    """
    Fetches a clustering model instance by its name from a given dictionary.

    Args:
        model_name: Name of the model to fetch.
        model_dict: Dictionary mapping model names to model instances.

    Returns:
        Clustering model instance.

    Raises:
        ValueError: If the model name does not exist in the dictionary.
    """
    all_model_dict = get_clustering_model_dict()
    if model_dict is not None:
        all_model_dict.update(model_dict)

    if model_name not in all_model_dict:
        raise ValueError(f"Model {model_name} is not available.")
    return all_model_dict[model_name]


def find_best_algorithm(
    embeddings: np.ndarray,
    model_dict: Optional[Dict[str, Any]] = None,
    model_names: List[str] = ["kmeans", "agglomerative_clustering"],
    min_cluster_num: int = 2,
    max_cluster_num: int = 5,
    metrics: List[str] = ["dbs"],
    test_metric: str = "dbs",
    print_topk: bool = True,
    topk: int = 3,
    result_filepath: str = "clustering_results.csv",
) -> List[Dict[str, Any]]:
    """
    Finds the best clustering algorithm based on specified metrics.

    Args:
        embeddings: Array of embeddings.
        model_dict: Dictionary mapping model names to model instances.
        model_names: List of model names to test.
        min_cluster_num: Minimum number of clusters.
        max_cluster_num: Maximum number of clusters.
        metrics: List of metric names to calculate.
        test_metric: Metric name to sort results.
        print_topk: Whether to print the top k results.
        topk: Number of top results to print.
        result_filepath: Path to save the results CSV file.

    Returns:
        List of dictionaries containing model name, number of clusters, and metrics.

    Raises:
        ValueError: If the result file already exists.
    """
    if os.path.exists(result_filepath):
        raise ValueError(f"{result_filepath} already exists.")

    model_dict = model_dict or get_clustering_model_dict()

    results = []

    for model_name in model_names:
        if model_name not in model_dict:
            print(f"{model_name} not included in model_dict")
            continue

        model_base = load_clustering_model(model_name, model_dict)
        for cluster_num in tqdm(
            range(min_cluster_num, max_cluster_num + 1), leave=False
        ):
            model = model_base.set_params(n_clusters=cluster_num)
            try:
                _, metrics_results = cluster(embeddings, model, metrics)
                results.append(
                    {
                        "model_name": model_name,
                        "cluster_num": cluster_num,
                        **metrics_results,
                    }
                )
            except Exception:
                results.append(
                    {
                        "model_name": model_name,
                        "cluster_num": cluster_num,
                        **{key: float("-inf") for key in metrics},
                    }
                )

    with open(result_filepath, "w", newline="") as csvfile:
        fieldnames = ["model_name", "cluster_num"] + metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metric_directions = {"dbs": False, "silhouette": True, "calinski": True}

    print("Results saved")
    results.sort(
        key=lambda x: (
            x[test_metric]
            if metric_directions.get(test_metric, True)
            else -x[test_metric]
        )
    )
    if print_topk:
        pprint(results[:topk])
    return results


def sample_random_documents(
    embeddings: np.ndarray,
    documents: List[str],
    cluster_num: int,
    cluster_labels: Optional[np.ndarray] = None,
    model: Optional[ClusterMixin] = None,
    n_samples: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Samples random documents from each cluster.

    Args:
        embeddings: Array of embeddings.
        documents: List of document texts.
        cluster_num: Number of clusters.
        cluster_labels: Pre-computed cluster labels (optional).
        model: Clustering model (optional, used if cluster_labels not provided).
        n_samples: Number of samples per cluster.
        verbose: Whether to print sampled documents.

    Returns:
        DataFrame with sampled documents and their cluster IDs.
    """
    if cluster_labels is None:
        cluster_labels = model.fit_predict(embeddings)

    sampled_docs_df = pd.DataFrame()

    for cluster_id in range(cluster_num):
        indices = np.where(cluster_labels == cluster_id)[0]
        unique_documents = np.unique(np.array(documents)[indices])
        tn_samples = min(n_samples, len(unique_documents))

        sampled_docs = np.random.choice(
            unique_documents, size=tn_samples, replace=False
        )

        cluster_data = pd.DataFrame(
            {
                "cluster_id": [cluster_id] * tn_samples,
                "document": sampled_docs,
            }
        )

        sampled_docs_df = pd.concat([sampled_docs_df, cluster_data], ignore_index=True)

        if verbose:
            print(f"cluster {cluster_id}")
            print("\n".join(sampled_docs))
            print()

    return sampled_docs_df


def sample_centroids_documents(
    embeddings: np.ndarray,
    documents: List[str],
    cluster_num: int,
    cluster_labels: Optional[np.ndarray] = None,
    model: Optional[ClusterMixin] = None,
    n_samples: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Samples documents closest to cluster centroids.

    Args:
        embeddings: Array of embeddings.
        documents: List of document texts.
        cluster_num: Number of clusters.
        cluster_labels: Pre-computed cluster labels (optional).
        model: Clustering model (optional, used if cluster_labels not provided).
        n_samples: Number of samples per cluster.
        verbose: Whether to print sampled documents.

    Returns:
        DataFrame with sampled documents, their cluster IDs, and distances to centroids.
    """
    if cluster_labels is None:
        cluster_labels = model.fit_predict(embeddings)
    closest_docs_df = pd.DataFrame()

    for cluster_id in range(cluster_num):
        indices = np.where(cluster_labels == cluster_id)[0]
        centroid = embeddings[indices].mean(axis=0).reshape(1, -1)
        distances = euclidean_distances(embeddings[indices], centroid)
        sorted_indices = np.argsort(distances.ravel())
        unique_documents, unique_indices = np.unique(
            np.array(documents)[indices[sorted_indices]], return_index=True
        )

        tn_samples = min(n_samples, len(unique_documents))

        closest_indices = unique_indices[:tn_samples].tolist()
        closest_docs = [unique_documents[i] for i in closest_indices]
        closest_distances = distances[sorted_indices][unique_indices][
            :tn_samples
        ].ravel()

        cluster_data = pd.DataFrame(
            {
                "cluster_id": [cluster_id] * tn_samples,
                "document": closest_docs,
                "distance_to_centroid": closest_distances,
            }
        )

        closest_docs_df = pd.concat([closest_docs_df, cluster_data], ignore_index=True)

        if verbose:
            print(f"cluster {cluster_id}")
            print("\n".join(closest_docs))
            print()

    return closest_docs_df
