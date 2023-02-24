from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import \
    TopicModelDataPreparation
from gensim import models
from gensim.corpora.dictionary import Dictionary


class Model:
    """Class for topic modeling.

    Parameters
    ----------
    num_topics : int
        Number of topics to be extracted.
    docs : Union[pd.Series, List[List[str]]]
        List of documents.
    encoded_docs : Union[pd.Series, List[List[str]]]
        List of encoded documents.
    filtered_lemmas : Union[pd.Series, List[List[str]]]
        List of filtered lemmas.
    model_type : str, optional
        Type of model to be used, by default "lda"
    random_state : int, optional
        Random state, by default 42
    **kwargs : dict
        Keyword arguments for the model.
    """
    def __init__(self,
                 num_topics: int,
                 docs: Union[pd.Series, List[List[str]]],
                 encoded_docs: Union[pd.Series, List[List[str]]],
                 filtered_lemmas: Union[pd.Series, List[List[str]]],
                 model_type: str = "lda",
                 random_state: int = 42,
                 **kwargs):
        self.model_type = model_type
        self.encoded_docs = encoded_docs
        self.num_topics = num_topics

        if self.model_type == "lda":
            self.int_model = models.LdaMulticore(
                corpus=self.encoded_docs,
                num_topics=num_topics,
                random_state=random_state,
                passes=kwargs.get("passes", 8),
                iterations=kwargs.get("iterations", 100),
                alpha=kwargs.get("alpha", "symmetric"),
            )
        elif self.model_type == "nmf":
            self.int_model = models.Nmf(
                corpus=self.encoded_docs,
                num_topics=num_topics,
                random_state=random_state,
                passes=kwargs.get("passes", 8),
                kappa=kwargs.get("kappa", 1.0),
            )
        elif self.model_type == "ctm":
            self.tp = TopicModelDataPreparation(kwargs.get("contextualized_model", "paraphrase-distilroberta-base-v2"))
            self.training_dataset = self.tp.fit(
                text_for_contextual=docs.values.tolist(),
                text_for_bow=filtered_lemmas.apply(lambda x: " ".join(x)).values.tolist())
            self.int_model = CombinedTM(bow_size=len(self.tp.vocab),
                                        contextual_size=kwargs.get("contextual_size", 768),
                                        n_components=num_topics)
            self.int_model.fit(self.training_dataset)

    def get_topics(self, num_words: int = 10) -> Tuple[pd.DataFrame, bool]:
        if self.model_type == "lda" or self.model_type == "nmf":
            res = self.int_model.show_topics(
                num_topics=self.int_model.num_topics,
                num_words=num_words,
                formatted=False,
            )
            return pd.DataFrame([[topic_id, int(word_id), word_imp] for topic_id, topic_words in res
                                 for word_id, word_imp in topic_words]), False
        elif self.model_type == "ctm":
            topic_words_distr = self.int_model.get_topic_word_distribution()
            words_ids = np.apply_along_axis(lambda x: x.argsort()[::-1][:num_words], 1, topic_words_distr)
            return pd.DataFrame([[topic_id, self.tp.vocab[word_id], topic_words_distr[topic_id, word_id]]
                                 for topic_id, topic_words_ids in enumerate(words_ids)
                                 for word_id in topic_words_ids]), True

    def get_topics_list(self, dictionary: Dictionary, num_words: int = 20) -> List[List[str]]:
        if self.model_type == "lda" or self.model_type == "nmf":
            if not dictionary.id2token:
                dictionary.id2token = {v: k for k, v in dictionary.token2id.items()}
            res = self.int_model.show_topics(
                num_topics=self.int_model.num_topics,
                num_words=num_words,
                formatted=False,
            )
            return [[dictionary.id2token[int(word_id)] for word_id, word_imp in topic_words]
                    for topic_id, topic_words in res]
        elif self.model_type == "ctm":
            return self.int_model.get_topic_lists(num_words)

    def get_topic_probs(self, corpus: Union[pd.Series, List[List[str]]]) -> np.ndarray:
        if self.model_type == "lda" or self.model_type == "nmf":
            corpus_model = self.int_model[corpus]
            res_len = len(corpus)
            res = np.zeros((res_len, self.num_topics))
            for i, doc in enumerate(corpus_model):
                for topic in doc:
                    res[i][topic[0]] = np.round(topic[1], 4)
            return res
        elif self.model_type == "ctm":
            return self.int_model.get_thetas(self.training_dataset)

    def get_term_topics(self, word_id: int, min_prob: float = 0) -> pd.DataFrame:
        if self.model_type == "lda" or self.model_type == "nmf":
            return self.int_model.get_term_topics(word_id, min_prob)
        elif self.model_type == "ctm":
            pass

    def save(self, path):
        self.int_model.save(path)