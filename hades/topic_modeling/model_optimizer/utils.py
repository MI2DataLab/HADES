import json
import os
from typing import List, Optional, Union
import pandas as pd

from pyLDAvis import prepared_data_to_html
from hades.summaries.summarize import prepare_app_summaries

from hades.plots.topics import interactive_exploration
from hades.topic_analysis.sentence_topic_analyser import SentenceTopicAnalyser
from .model_optimizer import ModelOptimizer

def save_data_for_app(
    model_optimizers: List[ModelOptimizer],
    num_words: int = 10,
    column: str = "country",
    perplexity: int = 10,
    n_iter: int = 1000,
    init: str = "pca",
    learning_rate_tsne: Union[str, float] = "auto",
    n_neighbors: int = 7,
    metric: str = "euclidean",
    min_dist: float = 0.1,
    learning_rate_umap: float = 1,
    path: str = "",
    save_model: bool = False,
    n_extract_sentences: int = 6,
    df: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    openai_organization: Optional[str] = None
):
    os.makedirs(path, exist_ok=True)
    config_dict = {}
    config_dict['division_column'] = column
    config_dict['sections'] = {}
    for model_optimizer in model_optimizers:
        filter_name = "_".join([value.replace(":","").replace(" ","_")
                                    .replace(",","").replace("/","_")
                                    .replace("(","").replace(")","")
                                    .replace("&","").replace("-","_")
                                for value in model_optimizer.column_filter.values()])
        topic_words = model_optimizer.get_topics_df(num_words)
        topics_by_country = model_optimizer.get_topic_probs_averaged_over_column(column, show_names=True)
        if save_model:
            model_optimizer.save(path=path)
        topic_words.to_csv(path + filter_name + "_topic_words.csv")
        topics_by_country.to_csv(path + filter_name + "_probs.csv")
        tsne_mapping = model_optimizer.get_tsne_mapping(
            column,
            perplexity,
            n_iter,
            init,
            learning_rate_tsne,
        )
        umap_mapping = model_optimizer.get_umap_mapping(
            column,
            n_neighbors,
            metric,
            min_dist,
            learning_rate_umap,
        )
        mappings = tsne_mapping.join(umap_mapping)
        mappings.to_csv(path + filter_name + "_mapping.csv")
        if model_optimizer.best_model.model_type == "lda":
            vis = interactive_exploration(model_optimizer.best_model.int_model, model_optimizer.encoded_docs, model_optimizer.lemmas_dictionary)
            vis_html_string = prepared_data_to_html(vis)
        with open(path + filter_name + "_vis.txt", "w") as text_file:
            text_file.write(vis_html_string)
        
        sentence_topic_analyser = SentenceTopicAnalyser(model_optimizer)
        df_summarized = model_optimizer.data.groupby("country")['tokens'].sum()
        sentences_processed = sentence_topic_analyser.process_documents(df_summarized)
        country_sentence_dict = dict(zip(list(df_summarized.index), sentences_processed))
        with open(path + filter_name + "_essentials.json", 'w') as json_file:
            json.dump(country_sentence_dict, json_file)
        config_dict['sections'][filter_name] = {"probs": path + filter_name + "_probs.csv",
            "mapping": path + filter_name + "_mapping.csv",
            "topic_words": path + filter_name + "_topic_words.csv",
            "vis": path + filter_name + "_vis.txt",
            "essentials": path + filter_name + "_essentials.json"}
    config_dict["summaries_file"] = None
    config_dict["additional_files"] = []
    if api_key is not None and openai_organization is not None and df is not None:
        prepare_app_summaries(df, list(model_optimizers[0].column_filter.keys())[0], n_extract_sentences, path + "summaries.json", api_key, openai_organization)
        config_dict["summaries_file"] = path + "summaries.json"
    with open(path + "config.json", 'w') as json_file:
        json.dump(config_dict, json_file)
