import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN


def calculate_linkage_matrix(
    topic_probs: pd.DataFrame, method: str = "average", metric: str = "cosine"
) -> np.ndarray:
    return hc.linkage(topic_probs, method=method, metric=metric)


def get_hierarchical_clusters(linkage: np.ndarray, t: float = 1.0):
    return hc.fcluster(linkage, t=t)
    
def get_kmeans_clusters(topic_probs: pd.DataFrame, n_clusters: int, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(topic_probs)
    return kmeans.labels_

def get_hdbscan_clusters(topic_probs: pd.DataFrame, min_cluster_size: int=5, min_samples: int=None,
                         cluster_selection_epsilon: int=0.0):
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     cluster_selection_epsilon=cluster_selection_epsilon).fit(topic_probs)
    return hdbscan.labels_
