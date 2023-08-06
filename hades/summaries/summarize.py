import json
import pandas as pd
from summarizer import Summarizer
from typing import Dict
import warnings
from transformers import pipeline
from hades.topic_modeling.model_optimizer import ModelOptimizer

model = Summarizer()


def extract_n_most_important_sentences(text: str, n_of_sentences: int) -> str:
    """Function extracting n most important sentences from text using BERT model"""
    result = model(text, num_sentences=n_of_sentences)
    return result


def abstractive_summary(extractive_summary: str, model:str = 'EleutherAI/gpt-neo-1.3B') -> str:
    """Function making abstractive summaries out of previously extracted most important sentences"""
    prompt = extractive_summary + ' Summarize the text above in three sentences: \n'
    generator = pipeline('text-generation', model=model)
    response = generator(prompt, do_sample=True, min_length=50)
    
    return response[0]['generated_text']


def make_summary(text: str, n_extract_sentences: int) -> str:
    """
    Function making abstractive summaries out of previously extracted most important sentences
    Args:
        text (str): text to summarize
        n_extract_sentences (int): Number of sentences to extract
    Returns:
        summary (str): abstractive summary
    """
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
    """
    Function making summaries for section given in model_optimizer
    """
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
