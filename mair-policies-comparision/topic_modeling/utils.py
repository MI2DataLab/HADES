from typing import Dict, List, Optional, Tuple, Union

from collections import Counter
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from tqdm import tqdm
from gensim.models import LdaModel


def check_coherence_for_topics_num(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    common_words_filtered: List[str] = [],
    topic_numbers_range: Tuple[int, int] = (2, 11),
    passes: int = 8,
    iterations: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[pd.Series, List[LdaMulticore], pd.Series, Dictionary, List[float]]:
    filtered_lemmas = get_filtered_lemmas(df, filter_dict, common_words_filtered)
    lemmas_dictionary = get_lemmas_dictionary(filtered_lemmas)
    encoded_docs = filtered_lemmas.apply(lemmas_dictionary.doc2bow)
    models = get_lda_models(encoded_docs, topic_numbers_range, passes, iterations, random_state)
    cvs = get_coherences(models, filtered_lemmas, lemmas_dictionary)
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
) -> pd.Series:
    filtered_lemmas = df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)][
        "lemmas"
    ].copy()
    filtered_lemmas = filtered_lemmas.apply(
        lambda doc: [lemma for lemma in doc if not (lemma in common_words_filtered)]
    )
    return filtered_lemmas


def get_lemmas_dictionary(filtered_lemmas: pd.Series):
    lemmas_dictionary = Dictionary(filtered_lemmas)
    lemmas_dictionary.filter_extremes(no_below=4, no_above=1)
    return lemmas_dictionary


def get_lda_models(
    corpus: Union[pd.Series, List[List[str]]],
    topic_numbers_range: Tuple[int, int] = (2, 11),
    passes: int = 8,
    iterations: int = 100,
    random_state: Optional[int] = None,
) -> List[LdaMulticore]:
    return [
        LdaMulticore(
            corpus,
            num_topics=topic_numbers,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
        )
        for topic_numbers in tqdm(range(*topic_numbers_range))
    ]


def get_coherences(
    models: List[LdaMulticore], texts: Union[pd.Series, List[List[str]]], dictionary: Dictionary
) -> List[float]:
    return [
        CoherenceModel(model, texts=texts, dictionary=dictionary).get_coherence()
        for model in tqdm(models)
    ]


def _topics_df(model: LdaModel, docs: pd.Series, num_words: int = 10) -> pd.DataFrame:
    topics = model.show_topics(formatted=False, num_words=num_words)
    counter = Counter(docs.sum())
    out = [[word, i, weight, counter[word]] for i, topic in topics for word, weight in topic]
    df = pd.DataFrame(out, columns=["word", "topic_id", "importance", "word_count"])
    df = df.sort_values(by=["importance"], ascending=False)
    return df
