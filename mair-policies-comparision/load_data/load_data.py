import json
import pickle
from typing import List

from gensim.models import Phrases
import pandas as pd
import spacy
import swifter


def read_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines[0] if len(lines) > 0 else ""


def load_dataframe(path: str, text_path_col: str = "text_path") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    assert text_path_col in df.columns
    df["text"] = df[text_path_col].apply(read_txt)
    return df


def process_tokens(
    doc: pd.Series, nlp: spacy.language.Language, stop_words: List[str]
) -> List[str]:
    spacy_text = nlp(doc)
    return [
        token
        for token in spacy_text
        if not any([token.is_stop, token.is_punct, token.lemma_ in stop_words, not token.is_alpha])
    ]


def process_lemmas(doc: pd.Series) -> List[str]:
    return [token.lemma_.lower() for token in doc]


def _multiply_ngrams(tokens: List[str]):
    for token in tokens:
        if " " in token:
            yield token
            yield token
        yield token


def process_text(
    docs: pd.DataFrame,
    stop_words: List = [],
    spacy_model: str = "en_core_web_lg",
) -> pd.DataFrame:
    nlp = spacy.load(spacy_model)
    df = docs.copy()

    df["tokens"] = df["text"].swifter.apply(process_tokens, args=(nlp, stop_words))
    df["lemmas"] = df["tokens"].apply(process_lemmas)
    ngram_data = df["tokens"].apply(lambda doc: [token.lemma_.lower() for token in doc])
    bigram = Phrases(ngram_data, min_count=1, delimiter=" ")
    trigram = Phrases(bigram[ngram_data], min_count=1, delimiter=" ")
    df["lemmas"] = df["lemmas"].apply(
        lambda doc: doc + list(_multiply_ngrams(trigram[bigram[doc]]))
    )
    return df
