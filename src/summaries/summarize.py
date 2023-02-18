import pandas as pd
from summarizer import Summarizer
import openai
import json

model = Summarizer()


def extract_n_most_important_sentences(text: str, n_of_sentences: int):
    result = model(text, num_sentences=n_of_sentences)
    return result


def abstractive_summary(extractive_summary: str, api_key: str, openai_organization: str,
                        gpt3_model: str = "text-davinci-002", temperature: int = 0.7):
    openai.api_key = api_key
    openai.organization = openai_organization
    prompt = extractive_summary + ' Summarize the text above in three sentences: \n'
    response = openai.Completion.create(model=gpt3_model, prompt=prompt, temperature=temperature, max_tokens=120)
    return response['choices'][0]['text']


def make_summary(text: str, n_extract_sentences: int, api_key: str, openai_organization: str):
    """
    Function making abstractive summaries out of previously extracted most important sentences
    Args:
        text:
        n_extract_sentences: Number of sentences to extract
        api_key: OpenAI API key
        openai_organization: OpenAI organization name
    Returns:
        summary: abstractive summary
    """
    extracted_sentences = extract_n_most_important_sentences(text, n_extract_sentences)
    summary = abstractive_summary(extracted_sentences, api_key, openai_organization)
    return summary


def prepare_app_summaries(df: pd.DataFrame, n_extract_sentences: int, dump_path: str,  api_key: str, openai_organization: str, verbose=False):
    final_summaries = dict()
    paragraphs = list(set(df['paragraph']))
    # TODO: change to ids
    ids = list(set(df['country']))
    for paragraph in paragraphs:
        paragraph_dict = dict()
        for id in ids:
            if verbose:
                print(f'Paragraph: {paragraph}, id: {id}')
            try:
                text_path = df[(df['paragraph'] == paragraph) & (df['country'] == id)]['text_path'].values[0]
                text_path = text_path.strip('../')
                with open(text_path, 'r') as f:
                    text = f.read()
                summary = make_summary(text, n_extract_sentences, api_key, openai_organization)
                summary = summary.strip('\n')
            except:
                if verbose:
                    print(f'No text for paragraph: {paragraph}, id: {id}')
                summary = 'This section is not available for given ID'
            paragraph_dict[id] = summary
            if verbose:
                print(f'Summary: {summary}')
        final_summaries[paragraph] = paragraph_dict

    # Saving summaries to json file
    with open(dump_path, 'w') as fp:
        json.dump(final_summaries, fp)