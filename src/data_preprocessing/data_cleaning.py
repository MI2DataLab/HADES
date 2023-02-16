import re


def text_cleaning(text):
    # deleting URLs
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)
    # deleting headlines
    text = re.sub(r'((\d\.)+\d) +[A-Z]([a-z]|\s|,)+', '', text)
    # deleting random numbers
    text = re.sub(r'((\d)+ +)+\(\d+\)', '', text)
    # deleting picture descriptions
    text = re.sub(r' \d+ [A-Z](\w|\s|,)+.', '', text)
    # deleting tables
    text = re.sub(r' -', '', text)
    sentences = text.split('. ')
    to_delete = False
    sentences_copy = sentences.copy()
    for i, sentence in enumerate(sentences):
        if to_delete:
            to_delete = False
            sentences_copy[i] = ''
        if re.match(r'\s+Table \d+', sentence):
            to_delete = True
            sentences_copy[i] = ''
    text = '. '.join(sentences_copy)
    # deleting multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r' . ', '', text)
    return text