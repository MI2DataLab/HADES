import string
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel, EnsembleLda, LdaModel
from gensim.models.ldamulticore import LdaMulticore
from sklearn.manifold import TSNE
from topic_modeling.utils import (
    get_filtered_lemmas,
    get_lemmas_dictionary,
    tsne_dim_reduction,
    umap_dim_reduction,
)
from tqdm import tqdm
from umap import UMAP


class ModelOptimizer:
    def __init__(
        self,
        df: pd.DataFrame,
        column_filter: Dict[str, str],
        words_to_remove: List[str] = [],
        topic_numbers_range: Tuple[int, int] = (2, 11),
        lda_alpha: Union[float, str] = "symmetric",
        lda_passes: int = 8,
        lda_iterations: int = 100,
        random_state: Optional[int] = None,
    ):
        self.column_filter = column_filter
        self.random_state = random_state
        self.lda_alpha = lda_alpha
        self.lda_passes = lda_passes
        self.lda_iterations = lda_iterations
        self.data = df.loc[(df[list(column_filter)] == pd.Series(column_filter)).all(axis=1)]
        self.filtered_lemmas = get_filtered_lemmas(self.data, words_to_remove)
        self.lemmas_dictionary = get_lemmas_dictionary(self.filtered_lemmas)
        self.encoded_docs = self.filtered_lemmas.apply(self.lemmas_dictionary.doc2bow)
        self.models = get_lda_models(
            self.encoded_docs, topic_numbers_range, lda_passes, lda_iterations, lda_alpha, random_state
        )
        self.cvs = get_coherences(self.models, self.filtered_lemmas, self.lemmas_dictionary)
        self.topics_num = get_best_topics_num(self.cvs)

    @property
    def best_model(self):
        return self.models[self.topics_num]

    def get_topics_df(self, num_words: int = 10) -> pd.DataFrame:
        topics = self.best_model.show_topics(formatted=False, num_words=num_words)
        counter = Counter(self.filtered_lemmas.sum())
        out = [[word, i, weight, counter[word]] for i, topic in topics for word, weight in topic]
        df = pd.DataFrame(out, columns=["word", "topic_id", "importance", "word_count"])
        df = df.sort_values(by=["importance"], ascending=False)
        return df

    def get_topic_probs_df(self) -> pd.DataFrame:
        """Returns original data frame with added columns for topic probabilites."""
        corpus_model = self.best_model[self.encoded_docs]
        res_len = len(self.data)
        res = np.zeros((res_len, self.topics_num))
        for i, doc in enumerate(corpus_model):
            for topic in doc:
                res[i][topic[0]] = np.round(topic[1], 4)
        modeling_results = pd.concat([self.data.reset_index(drop=True), pd.DataFrame(res)], axis=1)

        return modeling_results

    def get_topic_probs_averaged_over_column(self, column: str = "country") -> pd.DataFrame:
        """Returns topic probabilities averaged over given column."""
        modeling_results = self.get_topic_probs_df()
        result = []
        column_vals_added = []
        column_vals = modeling_results[column].unique()
        rows_by_column = modeling_results.groupby(column).count()[0].max()
        for column_val in column_vals:
            df_tmp = modeling_results[modeling_results[column] == column_val]
            if df_tmp.shape[0] != rows_by_column:
                warnings.warn(f"{column} - {column_val} has missing rows!")
                continue
            result.append(df_tmp.iloc[:, -self.topics_num :].values.flatten())
            column_vals_added.append(column_val)
        res = pd.DataFrame(np.vstack(result), index=column_vals_added)
        res.index.name = column
        return res

    def get_tsne_mapping(
        self,
        column: str = "country",
        perplexity: int = 40,
        n_iter: int = 1000,
        init: str = "pca",
        learning_rate: Union[str, float] = "auto",
    ):
        topics_by_country = self.get_topic_probs_averaged_over_column(column)
        mapping = tsne_dim_reduction(
            topics_by_country, self.random_state, perplexity, n_iter, init, learning_rate
        )
        return mapping

    def get_umap_mapping(
        self,
        column: str = "country",
        n_neighbors: int = 7,
        metric: str = "euclidean",
        min_dist: float = 0.1,
        learning_rate: float = 1,
    ):
        topics_by_country = self.get_topic_probs_averaged_over_column(column)
        mapping = umap_dim_reduction(
            topics_by_country,
            self.random_state,
            n_neighbors,
            metric,
            min_dist,
            learning_rate,
        )
        return mapping

    def save(self):
        filter_name = "_".join([value.replace(" ", "_") for value in self.column_filter.values()])
        self.encoded_docs.to_csv(str(self.alpha) + "_" + filter_name + "_encoded_docs.csv")
        self.lemmas_dictionary.save(str(self.alpha) + "_" + filter_name + "_dictionary.dict")
        self.best_model.save(str(self.alpha) + "_" + filter_name + "_lda_model.model")


def save_data_for_app(
    model: ModelOptimizer,
    num_words: int = 10,
    column: str = "country",
    n_neighbors: int = 7,
    metric: str = "euclidean",
    min_dist: float = 0.1,
    learning_rate: float = 1,
):
    filter_name = "_".join([value.replace(" ", "_") for value in model.column_filter.values()])
    topic_words = model.get_topics_df(num_words)
    topics_by_country = model.get_topic_probs_averaged_over_column(column)
    model.save()
    topic_words.save(str(model.lda_alpha) + "_" + filter_name + "_topic_words.csv")
    topics_by_country.save(str(model.lda_alpha) + "_" + filter_name + "_probs.csv")
    tsne_mapping = model.get_tsne_mapping(
        column,
        n_neighbors,
        metric,
        min_dist,
        learning_rate,
    )
    umap_mapping = model.get_umap_mapping(
        column,
        n_neighbors,
        metric,
        min_dist,
        learning_rate,
    )
    mappings = tsne_mapping.join(umap_mapping)
    mappings.to_csv(str(model.lda_alpha) + "_" + filter_name +"_mapping.csv")


def get_best_topics_num(cvs: Dict[int, float]) -> int:
    return max(cvs, key=cvs.get)


def get_lda_models(
    corpus: Union[pd.Series, List[List[str]]],
    topic_numbers_range: Tuple[int, int] = (2, 11),
    passes: int = 8,
    iterations: int = 100,
    alpha: Union[float, str] = "symmetric",
    random_state: Optional[int] = None,
) -> Dict[int, LdaMulticore]:
    return {
        num_topics: LdaMulticore(
            corpus,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            alpha=alpha,
        )
        for num_topics in tqdm(range(*topic_numbers_range))
    }


def get_coherences(
    models: Dict[int, LdaMulticore],
    texts: Union[pd.Series, List[List[str]]],
    dictionary: Dictionary,
    coherence: str = "c_v",
) -> Dict[int, float]:
    return {
        num_topics: CoherenceModel(
            model, texts=texts, dictionary=dictionary, coherence=coherence
        ).get_coherence()
        for num_topics, model in tqdm(models.items())
    }
