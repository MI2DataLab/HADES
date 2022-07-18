from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

import pandas as pd


def check_coherence(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    common_words_filtered: List[str],
    topic_numbers_range: Tuple[int, int] = (3, 13),
    passes: int = 8,
    iterations: int = 100,
    random_state: Optional[int] = None,
):
    filtered_lemmas = get_filtered_lemmas(df, filter_dict, common_words_filtered)
    lemmas_dictionary = get_lemmas_dictionary(filtered_lemmas)
    encoded_docs = filtered_lemmas.apply(lemmas_dictionary.doc2bow)
    models = get_lda_models(encoded_docs, topic_numbers_range, passes, iterations, random_state)
    cvs = get_coherences(models, filtered_lemmas, filter_dict)
    return (
        filtered_lemmas,
        models,
        encoded_docs,
        lemmas_dictionary,
        cvs,
    )


def get_filtered_lemmas(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    common_words_filtered: List[str],
):
    filtered_lemmas = df[(df[key] == value for key, value in filter_dict.items())][
        "lemmas"
    ].copy()
    filtered_lemmas = filtered_lemmas.apply(
        lambda doc: [lemma for lemma in doc if not (lemma in common_words_filtered)]
    )
    return filtered_lemmas


def get_lemmas_dictionary(filtered_lemmas: pd.Series):
    lemmas_dictionary = Dictionary(filtered_lemmas)
    lemmas_dictionary.filter_extremes(
        no_below=4, no_above=1
    )


def get_lda_models(
    corpus: Union[pd.Series, List[List[str]]],
    topic_numbers_range: Tuple[int, int] = (3, 13),
    passes: int = 8,
    iterations: int = 100,
    random_state: Optional[int] = None,
):
    return [
        LdaMulticore(
            corpus,
            num_topics=topic_numbers,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
        )
        for topic_numbers in tqdm(topic_numbers_range)
    ]


def get_coherences(
    models: List[LdaMulticore], texts: Union[pd.Series, List[List[str]]], dictionary: Dictionary
) -> List[float]:
    return [
        CoherenceModel(model, texts=texts, dictionary=dictionary).get_coherence()
        for model in models
    ]
