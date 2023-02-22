import warnings
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel



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


def shift_similarity(
    modeling_results: pd.DataFrame,
    num_topics: int,
    filter_dicts: List[Dict[str, str]],
    ir_delta: int = 1,
) -> pd.DataFrame:
    countries = modeling_results.country.unique()
    assert len(filter_dicts) == 2
    dimension_change = {"country": [], "cosine_sim": [], "IR_sim": [], "H_sim": []}
    for country in countries:
        df1 = modeling_results.loc[
            (modeling_results["country"] == country)
            & (modeling_results[list(filter_dicts[0])] == pd.Series(filter_dicts[0])).all(axis=1)
        ].loc[:, np.arange(num_topics)]
        df2 = modeling_results.loc[
            (modeling_results["country"] == country)
            & (modeling_results[list(filter_dicts[1])] == pd.Series(filter_dicts[1])).all(axis=1)
        ].loc[:, np.arange(num_topics)]
        if df1.shape[0] == 1:
            dimension_change["country"].append(country)
            dimension_change["cosine_sim"].append(1 - sp.distance.cosine(df1, df2))
            p = df1.values
            q = df2.values
            ir = np.sum(p * np.log(2 * p / (p + q))) + np.sum(q * np.log(2 * q / (p + q)))
            dimension_change["IR_sim"].append(10 ** (-ir_delta * ir))
            dimension_change["H_sim"].append(
                1 - (np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))
            )
    return pd.DataFrame(dimension_change)


def get_hierarchical_clusters(linkage: np.ndarray, t: float = 1.0, criterion: str = "distance"):
    return hc.fcluster(linkage, t=t, criterion=criterion)
