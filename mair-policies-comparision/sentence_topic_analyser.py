from typing import List, Tuple, Dict, Iterable

from topic_modeling.model_optimizer import ModelOptimizer
from spacy.tokens import Token


from collections import defaultdict

import numpy as np


class SentenceTopicAnalyser():
    def __init__(self, model: ModelOptimizer, minimal_set_len:int=3):
        self.model = model
        self.minimal_set_len = minimal_set_len

    def process_documents(self, documents: List[Iterable[Token]], top_sents = 3):
        for doc in documents:
            doc_sent_topics = self.calculate_sentences_topics_distribution(doc)
            for sent, distr in doc_sent_topics:
                doc_sent_topics
    
    def calculate_sentences_topics_distribution(self, tokens: Iterable[Token]) -> List[Tuple[str, List[float]]]:
        sentence_to_topics=defaultdict(list)
        id_to_sent = {}
        for token in tokens:
            try:
                tok_id = self.model.lemmas_dictionary.token2id[token.lemma_.lower()]
            except KeyError:
                continue
            sent_id = token.sent.start
            sentence_topics = self.model.best_model.get_term_topics(tok_id,0)
            sentence_topics = [prob for _, prob in sentence_topics]
            sentence_to_topics[sent_id].append(sentence_topics)
            if sent_id not in id_to_sent:
                id_to_sent[sent_id] = str(token.sent)

        results = [(id_to_sent[key], value) for key, value in sentence_to_topics.items() if len(value)>self.minimal_set_len]
        
        return results


def get_top_sents(sentences_topics_distribution: List[Tuple[str, List[float]]], topic_id: int, top_sents: int=3):
    sent_to_topics=dict()
    for sent, distrs in sentences_topics_distribution:
        distrs = np.array(distrs)
        sent_to_topics[sent]=distrs.mean(axis=0)
    a = [(sent, probs[topic_id]) for sent, probs in sent_to_topics.items()]
    return sorted(a, reverse=True)[:top_sents]

from dataclasses import dataclass


@dataclass
class SentenceWithTopics:
    ...