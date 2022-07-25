from typing import Dict, Tuple

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


def get_linkage_matrix(topic_probs, method="average", metric="cosine"):
    return hc.linkage(topic_probs, method=method, metric=metric)


def get_similarities(topic_probs) -> np.ndarray:
    return sp.distance.squareform(sp.distance.pdist(topic_probs.values, metric="cosine"))
