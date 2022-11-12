import string
from typing import Dict, List, Optional, Tuple, Union

from collections import Counter
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from tqdm import tqdm
from gensim.models import LdaModel
from sklearn.manifold import TSNE
from umap import UMAP


def check_coherence_for_topics_num(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    common_words_filtered: List[str] = [],
    topic_numbers_range: Tuple[int, int] = (2, 11),
    alpha: Union[float, str] = "symmetric",
    passes: int = 8,
    iterations: int = 100,
    random_state: Optional[int] = None,
    coherence_file: str = False,
) -> Tuple[pd.Series, List[LdaMulticore], pd.Series, Dictionary, List[float]]:
    filtered_lemmas = get_filtered_lemmas(df, filter_dict, common_words_filtered)
    lemmas_dictionary = get_lemmas_dictionary(filtered_lemmas)
    encoded_docs = filtered_lemmas.apply(lemmas_dictionary.doc2bow)
    models = get_lda_models(encoded_docs, topic_numbers_range, passes, iterations, alpha, random_state)
    cvs = get_coherences(models, filtered_lemmas, lemmas_dictionary)
    if coherence_file:
        umass = get_coherences(models, filtered_lemmas, lemmas_dictionary, coherence="u_mass")
        pd.DataFrame({"topic_num": range(*topic_numbers_range), "c_v": cvs, "u_mass": umass}).to_csv(
            coherence_file + ".csv"
        )
    return (
        filtered_lemmas,
        models,
        encoded_docs,
        lemmas_dictionary,
        cvs,
    )


def get_filtered_lemmas(
    df: pd.DataFrame,
    common_words_filtered: List[str],
) -> pd.Series:
    filtered_lemmas = df["lemmas"].copy()
    filtered_lemmas = filtered_lemmas.apply(
        lambda doc: [lemma for lemma in doc if not (lemma in common_words_filtered)]
    )
    return filtered_lemmas


def get_lemmas_dictionary(filtered_lemmas: pd.Series):
    lemmas_dictionary = Dictionary(filtered_lemmas)
    lemmas_dictionary.filter_extremes(no_below=4, no_above=1)
    lemmas_dictionary.filter_extremes()
    return lemmas_dictionary



def _topics_df(model: LdaModel, docs: pd.Series, num_words: int = 10) -> pd.DataFrame:
    topics = model.show_topics(formatted=False, num_words=num_words)
    counter = Counter(docs.sum())
    out = [[word, i, weight, counter[word]] for i, topic in topics for word, weight in topic]
    df = pd.DataFrame(out, columns=["word", "topic_id", "importance", "word_count"])
    df = df.sort_values(by=["importance"], ascending=False)
    return df


def tsne_dim_reduction(
    result_df: pd.DataFrame,
    random_state: int = 42,
    perplexity: int = 40,
    n_iter: int = 1000,
    init: str = "pca",
    learning_rate: Union[str, float] = "auto",
) -> pd.DataFrame:
    tsne_result_df = result_df.copy()
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity,
        n_iter=n_iter,
        init=init,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    tsne_raw_result = tsne.fit_transform(result_df)
    tsne_result_df["c1"] = tsne_raw_result[:, 0]
    tsne_result_df["c2"] = tsne_raw_result[:, 1]
    return tsne_result_df[["c1", "c2"]]


def umap_dim_reduction(
    result_df: pd.DataFrame,
    random_state: int = 42,
    n_neighbors: int = 7,
    metric: str = "euclidean",
    min_dist: float = 0.1,
    learning_rate: float = 1,
) -> pd.DataFrame:
    umap_result_df = result_df.copy()
    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    umap_raw_result = umap.fit_transform(result_df)
    umap_result_df["u1"] = umap_raw_result[:, 0]
    umap_result_df["u2"] = umap_raw_result[:, 1]
    return umap_result_df[["u1", "u2"]]
