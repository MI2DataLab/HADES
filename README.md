# HADES: Homologous Automated Document Exploration and Summarization
A powerful tool for comparing similarly structured documents

[![PyPI version](https://badge.fury.io/py/hades-nlp.svg)](https://pypi.org/project/hades-nlp/)
[![Downloads](https://static.pepy.tech/badge/hades-nlp)](https://pepy.tech/project/hades-nlp)

## Overview
`HADES` is a **Python** package for comparing similarly structured documents. HADES is designed to streamline the work of professionals dealing with large volumes of documents, such as policy documents, legal acts, and scientific papers. The tool employs a multi-step pipeline that begins with processing PDF documents using topic modeling, summarization, and analysis of the most important words for each topic. The process concludes with an interactive web app with visualizations that facilitate the comparison of the documents. HADES has the potential to significantly improve the productivity of professionals dealing with high volumes of documents, reducing the time and effort required to complete tasks related to comparative document analysis.

## Installation
Latest released version of the `HADES` package is available on [Python Package Index (PyPI)](https://pypi.org/project/hades-nlp/):

1. Install spacy `en-core-web-sm` or `en-core-web-lg` model for English language according to the [instructions](https://spacy.io/models/en)

2. Install `HADES` package using pip:

```sh
pip install -U hades-nlp
```
The source code and development version is currently hosted on [GitHub](https://github.com/MI2DataLab/HADES).
## Usage
The `HADES` package is designed to be used in a Python environment. The package can be imported as follows:

```python
from hades.data_loading import load_processed_data
from hades.topic_modeling import ModelOptimizer, save_data_for_app, set_openai_key
from my_documents_data import PARAGRAPHS, COMMON_WORDS, STOPWORDS
```
The `load_processed_data` function loads the documents to be processed. The `ModelOptimizer` class is used to optimize the topic modeling process. The `save_data_for_app` function saves the data for the interactive web app. The `set_openai_key` function sets the OpenAI API key.
`my_documents_data` contains the informations about the documents to be processed. The `PARAGRAPHS` variable is a list of strings that represent the paragraphs of the documents. The `COMMON_WORDS` variable is a list of strings that represent the most common words in the documents. The `STOPWORDS` variable is a list of strings that represent the most common words in the documents that should be excluded from the analysis.

First, the documents are loaded and processed:
```python
set_openai_key("my openai key")
data_path = "my/data/path"
processed_df = load_processed_data(
    data_path=data_path,
    stop_words=STOPWORDS,
    id_column='country',
    flattened_by_col='my_column',
)
```
After the documents are loaded, the topic modeling process is optimized for each paragraph:
```python
model_optimizers = []
for paragraph in PARAGRAPHS:
    filter_dict = {'paragraph': paragraph}
    model_optimizer = ModelOptimizer(
        processed_df,
        'country',
        'section',
        filter_dict,
        "lda",
        COMMON_WORDS[paragraph],
        (3,6),
        alpha=100
    )
    model_optimizer.name_topics_automatically_gpt3()
    model_optimizers.append(model_optimizer)

```
For each paragraph, the `ModelOptimizer` class is used to optimize the topic modeling process. The `name_topics_automatically_gpt3` function automatically names the topics using the OpenAI GPT-3 API. User can also use the `name_topics_manually` function to manually name the topics.

Finally, the data is saved for the interactive web app:
```python
save_data_for_app(model_optimizers, path='path/to/results', do_summaries=True)
```
The `save_data_for_app` function saves the data for the interactive web app. The `do_summaries` parameter is set to `True` to generate summaries for each topic.

When the data is saved, the interactive web app can be launched:
```sh
hades run-app --config path/to/results/config.json
```

***

