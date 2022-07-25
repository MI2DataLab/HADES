from typing import Dict, List, Tuple

import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from .utils import check_coherence_for_topics_num


def find_best_model(
    encoded_docs: pd.Series,
    lemmas_dictionary: Dictionary,
    cvs: List[float],
    topic_numbers_range: Tuple[int, int] = (2, 11),
    random_state: int = 42,
) -> LdaModel:
    topics_num = find_best_topics_num(cvs, topic_numbers_range)
    alpha = find_best_alpha()
    lda_model = LdaModel(
        encoded_docs,
        num_topics=topics_num,
        id2word=lemmas_dictionary,
        chunksize=200,
        eval_every=None,
        passes=25,
        iterations=100,
        random_state=random_state,
        eta="auto",
        alpha=alpha,
    )
    return lda_model


def find_best_topics_num(cvs: List[float], topic_numbers_range: Tuple[int, int] = (2, 11)) -> int:
    return topic_numbers_range[0] + cvs.index(max(cvs))


def find_best_alpha() -> int:
    return 100
