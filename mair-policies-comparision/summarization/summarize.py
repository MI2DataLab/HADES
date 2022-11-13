from summarizer import Summarizer
import openai

model = Summarizer()


def extract_n_most_important_sentences(text: str, n_of_sentences: int):
    result = model(text, num_sentences=n_of_sentences)
    return result


def abstractive_summary(extractive_summary: str, api_key: str, openai_organization: str,
                        gpt3_model: str = "text-davinci-002", temperature: int = 0.7):
    openai.api_key = api_key
    openai.organization = openai_organization
    prompt = extractive_summary + ' Tl;dr'
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
