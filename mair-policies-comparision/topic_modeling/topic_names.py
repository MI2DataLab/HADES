from typing import Dict, List, Union
import openai
import os
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from utils import _topics_df


def name_topics_manually(
    topic_df: pd.DataFrame, topic_names: Union[List[str], Dict[str, str]], num_topics: int
) -> pd.DataFrame:
    if isinstance(topic_names, list):
        colnames = topic_df.columns.to_list()
        colnames[-num_topics:] = topic_names
        topic_df.columns = colnames

    if isinstance(topic_names, dict):
        topic_df.rename(columns=topic_names, inplace=True)
    return topic_df


def name_topics_automatically_gpt3(
    topic_df: pd.DataFrame,
    model: LdaModel,
    docs: pd.Series,
    num_topics: int,
    num_keywords: int = 10,
    gpt3_model: str = "text-davinci-002",
    temperature: int = 0.5,
) -> pd.DataFrame:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    topics_keywords = _topics_df(model, docs, num_keywords)
    colnames = topic_df.columns.to_list()
    topic_colnames = colnames[-num_topics:]
    for i, colname in enumerate(topic_colnames):
        keywords = topics_keywords[topics_keywords["topic_id"] == colname].word.to_list()
        prompt = _generate_prompt(keywords)
        title = _generate_title(prompt, gpt3_model, temperature)
        topic_colnames[i] = title
    colnames[-num_topics:] = topic_colnames
    topic_df.columns = colnames
    return topic_df


def _generate_prompt(keywords: list) -> str:
    print(keywords)
    return f"""Suggest short (maximum three words) name for a topic based on given keywords:
    {', '.join(keywords)}"""


def _generate_title(prompt: str, model: str, temperature: int) -> str:
    response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature)
    return response.choices[0].text.split("\n")[-1]
