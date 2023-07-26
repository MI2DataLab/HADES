from typing import Callable, Union
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
from sklearn.cluster import KMeans, HDBSCAN
import scipy.spatial as sp


def calculate_linkage_matrix(
    topic_probs: pd.DataFrame, method: str = "average", metric: str = "cosine"
) -> np.ndarray:
    return hc.linkage(topic_probs, method=method, metric=_get_metric(metric))


def calculate_distance_matrix(topic_probs: pd.DataFrame, metric: str = "hd") -> pd.DataFrame:
    distances = sp.distance.squareform(
        sp.distance.pdist(topic_probs.values, metric=_get_metric(metric))
    )
    return pd.DataFrame(distances, index=topic_probs.index, columns=topic_probs.index)


def _get_metric(metric: str) -> Union[str, Callable]:
    if metric == "ir":
        metric = lambda p, q: np.sum(p * np.log(2 * p / (p + q))) + np.sum(
            q * np.log(2 * q / (p + q))
        )
    if metric == "hd":
        metric = lambda p, q: np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    return metric


def get_hierarchical_clusters(linkage: np.ndarray, t: float = 1.0):
    return hc.fcluster(linkage, t=t, criterion="distance")


def get_kmeans_clusters(topic_probs: pd.DataFrame,
                        n_clusters: int,
                        n_init: Union[str, int] = "auto",
                        random_state: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(topic_probs)
    return kmeans.labels_


def get_hdbscan_clusters(
    distance_matrix: pd.DataFrame,
    min_cluster_size: int = 2,
    min_samples: int = None,
    cluster_selection_epsilon: int = 0.0,
):
    hdbscan = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    hdbscan.fit(distance_matrix)
    return hdbscan.labels_
