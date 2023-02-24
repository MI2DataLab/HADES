from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp


def calculate_linkage_matrix(
    topic_probs: pd.DataFrame, method: str = "average", metric: str = "ir"
) -> np.ndarray:
    return hc.linkage(topic_probs, method=method, metric=_get_metric(metric))


def calculate_distance_matrix(topic_probs: pd.DataFrame, metric: str = "ir") -> pd.DataFrame:
    distances = sp.distance.squareform(sp.distance.pdist(topic_probs.values, metric=_get_metric(metric)))
    return pd.DataFrame(distances, index=topic_probs.index, columns=topic_probs.index)


def _get_metric(metric: str) -> Union[str, Callable]:
    if metric == "ir":
        metric = lambda p, q: np.sum(p * np.log(2 * p / (p + q))) + np.sum(
            q * np.log(2 * q / (p + q))
        )
    if metric == "hd":
        metric = lambda p, q: np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    return metric


def get_similarities(topic_probs: pd.DataFrame, metric: str = "ir") -> np.ndarray:
    return sp.distance.squareform(sp.distance.pdist(topic_probs.values, metric=_get_metric(metric)))


def get_hierarchical_clusters(linkage: np.ndarray, t: float = 1.0, criterion: str = "distance"):
    return hc.fcluster(linkage, t=t, criterion=criterion)
