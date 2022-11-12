from summarizer import Summarizer
import openai

model = Summarizer()
openai.organization = "oai-hackathon-2022-team-27"

def extract_n_most_important_sentences(text: str, n_of_sentences: int):
    result = model(text, num_sentences=n_of_sentences)
    return result


def abstractive_summary(extractive_summary: str, api_key: str, gpt3_model: str = "text-davinci-002", temperature: int = 0.7):
    openai.api_key = api_key
    prompt = extractive_summary + ' Tl;dr'
    response = openai.Completion.create(model=gpt3_model, prompt=prompt, temperature=temperature, max_tokens=120)
    return response['choices'][0]['text']
