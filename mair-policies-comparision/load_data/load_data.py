import os
from typing import List
import joblib

import pandas as pd
import pycountry
import spacy
from gensim.models import Phrases

from .utils import _multiply_ngrams, get_filtered_tokens, process_lemmas, process_tokens


def read_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines[0] if len(lines) > 0 else ""


def load_dataframe(path: str, text_path_col: str = "text_path") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    assert text_path_col in df.columns
    df["text"] = df[text_path_col].apply(read_txt)
    df = convert_country(df)
    return df


def load_processed_data(
    data_path: str,
    text_path_col: str = "text_path",
    stop_words: List = [],
    spacy_model: str = "en_core_web_lg",
    processed_filename: str = "data_processed.joblib",
    data_filename: str = "data.csv",
) -> pd.DataFrame:
    processed_path = data_path + processed_filename
    data_path = data_path + data_filename
    if os.path.isfile(processed_path):
        processed_data = read_processed_data(processed_path)
    else:
        processed_data = preprocess_text(load_dataframe(data_path, text_path_col), spacy_model)
        save_processed_data(processed_data, processed_path)
    processed_data = process_text(processed_data, stop_words)
    return processed_data


def save_processed_data(processed_data: pd.DataFrame, processed_path: str):
    joblib.dump(processed_data, processed_path)


def read_processed_data(processed_path: str) -> pd.DataFrame:
    return joblib.load(processed_path)


def preprocess_text(docs: pd.DataFrame, spacy_model: str = "en_core_web_lg") -> pd.DataFrame:
    nlp = spacy.load(spacy_model)
    df = docs.copy()
    df["doc"] = df["text"].apply(nlp)
    return df


def process_text(
    docs: pd.DataFrame,
    stop_words: List = [],
) -> pd.DataFrame:
    df = docs.copy()
    df["tokens"] = df["doc"].apply(get_filtered_tokens, args=(stop_words,))
    df["lemmas"] = df["tokens"].apply(process_lemmas)
    ngram_data = df["tokens"].apply(lambda doc: [token.lemma_.lower() for token in doc])

    bigram = Phrases(ngram_data, min_count=1, delimiter=" ")
    trigram = Phrases(bigram[ngram_data], min_count=1, delimiter=" ")

    df["lemmas"] = df["lemmas"].apply(lambda doc: doc + list(_multiply_ngrams(trigram[bigram[doc]])))
    return df


def convert_country(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["country"] = df["country"].apply(
            lambda country: pycountry.countries.search_fuzzy(country.replace("_", " "))[0].name
        )
    except:
        pass
    return df
