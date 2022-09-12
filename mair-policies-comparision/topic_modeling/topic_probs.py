import warnings
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


def get_topic_probs(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    model: LdaModel,
    num_topics: int,
    encoded_docs: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corpus_model = model[encoded_docs]
    df_metainfo = df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]
    res_len = len(df_metainfo)
    res = np.zeros((res_len, num_topics))
    for i, doc in enumerate(corpus_model):
        for topic in doc:
            res[i][topic[0]] = np.round(topic[1], 4)
    modeling_results = pd.concat([df_metainfo.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    topic_probs = modeling_results.groupby("country").mean().loc[:, np.arange(num_topics)]

    return modeling_results, topic_probs


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


def get_similarities(topic_probs: pd.DataFrame) -> np.ndarray:
    return sp.distance.squareform(sp.distance.pdist(topic_probs.values, metric="cosine"))


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


def topic_probs_by_column_binded(
    modeling_results: pd.DataFrame, num_topics: int, column: str = "country"
) -> pd.DataFrame:
    result = []
    column_vals_added = []
    column_vals = modeling_results[column].unique()
    rows_by_column = modeling_results.groupby(column).count()[0].max()
    for column_val in column_vals:
        df_tmp = modeling_results[modeling_results[column] == column_val]
        if df_tmp.shape[0] != rows_by_column:
            warnings.warn(f"{column} - {column_val} has missing rows!")
            continue
        result.append(df_tmp.iloc[:, -num_topics:].values.flatten())
        column_vals_added.append(column_val)
    res = pd.DataFrame(np.vstack(result), index=column_vals_added)
    res.index.name = column
    return res


def get_hierarchical_clusters(linkage: np.ndarray, t: float = 1.0, criterion: str = "distance"):
    return hc.fcluster(linkage, t=t, criterion=criterion)
