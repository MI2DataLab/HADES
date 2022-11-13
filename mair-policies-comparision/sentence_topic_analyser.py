from typing import List, Tuple, Dict, Iterable

from topic_modeling.model_optimizer import ModelOptimizer
from spacy.tokens import Token


from collections import defaultdict
from dataclasses import dataclass

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
    

    def calculate_sentences_topics_distribution(self, tokens: Iterable[Token]) -> List[Tuple[str, List[str], List[float]]]:
        sentence_to_words_topics=defaultdict(list)
        sentence_to_words=defaultdict(list)
        id_to_sent = {}
        for token in tokens:
            try:
                word = token.lemma_.lower()
                tok_id = self.model.lemmas_dictionary.token2id[word]
            except KeyError:
                continue
            sent_id = token.sent.start
            sentence_topics = self.model.best_model.get_term_topics(tok_id, 0)
            sentence_topics = [prob for _, prob in sentence_topics]
            sentence_to_words_topics[sent_id].append(sentence_topics)
            sentence_to_words[sent_id].append(word)
            if sent_id not in id_to_sent:
                id_to_sent[sent_id] = str(token.sent)

        results = [(id_to_sent[key], sentence_to_words[key], value) for key, value in sentence_to_words_topics.items() if len(value)>self.minimal_set_len]
        
        return results


    def get_sentences_from_topics(self, tokens: Iterable[Token], top_sents=3):
        results = dict()
        sentences_topics_distribution = self.calculate_sentences_topics_distribution(tokens)
        for topic_id in range(self.model.topics_num):
            best_sents, words = get_top_sents(sentences_topics_distribution,topic_id, top_sents)
            results[topic_id]={"sentences":best_sents, "words": words}
        return results


def get_top_sents(sentences_topics_distribution: List[Tuple[str, Tuple[str, List[str], List[float]]]], topic_id: int, top_sents: int):
    sent_to_topics=dict()
    sent_to_words_weights=dict()
    for sent, words, distrs in sentences_topics_distribution:
        distrs = np.array(distrs)
        sent_to_topics[sent]=distrs.mean(axis=0)
        sent_to_words_weights[sent]={word: distrs[i][topic_id] for i, word in enumerate(words)}
    sent_to_topic_weight = [(sent, probs[topic_id]) for sent, probs in sent_to_topics.items()]
    best_sents = sorted(sent_to_topic_weight, reverse=True)[:top_sents]
    words = {word:weight for sent in best_sents for word, weight in sent_to_words_weights[sent[0]].item()}
    return best_sents, words
