import json

import openai
import pandas as pd
from summarizer import Summarizer
from typing import Dict
import warnings

from hades.topic_modeling.model_optimizer import ModelOptimizer

model = Summarizer()


def extract_n_most_important_sentences(text: str, n_of_sentences: int):
    result = model(text, num_sentences=n_of_sentences)
    return result


def abstractive_summary(
    extractive_summary: str,
    gpt3_model: str = "text-davinci-003", 
    temperature: int = 0.7
):
    prompt = extractive_summary + ' Summarize the text above in three sentences: \n'
    response = openai.Completion.create(
        model=gpt3_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=120
    )
    return response['choices'][0]['text']


def make_summary(text: str, n_extract_sentences: int):
    """
    Function making abstractive summaries out of previously extracted most important sentences
    Args:
        text:
        n_extract_sentences: Number of sentences to extract
    Returns:
        summary: abstractive summary
    """
    if openai.api_key == None:
        warnings.warn(
            """
            Summary can't be made: no api key set;
            Key can be set using function set_openai_key(key)
            """
        )
        return
    extracted_sentences = extract_n_most_important_sentences(text, n_extract_sentences)
    summary = abstractive_summary(extracted_sentences)
    return summary


def make_section_summaries(
    model_optimizer: ModelOptimizer,
    n_extract_sentences: int,
    section_name: str = "-",
    do_summaries: bool = True,
    verbose: bool = False,
) -> Dict[str, str]:
    if openai.api_key == None and do_summaries:
        do_summaries = False
        verbose = False
        warnings.warn(
            """
            Summaries can't be made: no api key set;
            Key can be set using function set_openai_key(key)
            """
        )
    data = model_optimizer.data
    ids = list(set(data[model_optimizer.id_column]))
    section_summaries_dict = dict()
    for id in ids:
        if verbose:
            print(f'Section: {section_name}, id: {id}')
        if not do_summaries:
            summary = 'Summary is not available for given ID'
        else:
            try:
                text = data[(data[model_optimizer.id_column] == id)]['text'].values[0]
                summary = make_summary(text, n_extract_sentences)
                summary = summary.strip('\n')
            except:
                if verbose:
                    print(f'No text for section: {section_name}, id: {id}')
                summary = 'This section is not available for given ID'
        section_summaries_dict[id] = summary
    return section_summaries_dict
