import json
import os
from typing import List, Optional, Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import openai

from pyLDAvis import prepared_data_to_html
from hades.summaries import make_section_summaries

from hades.plots.topics import interactive_exploration
from hades.topic_analysis.sentence_topic_analyser import SentenceTopicAnalyser
from .model_optimizer import ModelOptimizer


def save_data_for_app(
    model_optimizers: List[ModelOptimizer],
    path: str,
    num_words: int = 10,
    perplexity: int = 10,
    n_iter: int = 1000,
    init: str = "pca",
    learning_rate_tsne: Union[str, float] = "auto",
    n_neighbors: int = 7,
    metric: str = "euclidean",
    min_dist: float = 0.1,
    learning_rate_umap: float = 1,
    save_model: bool = False,
    n_extract_sentences: int = 6,
    do_summaries: bool = True,
    verbose_summaries: bool = False,
):
    """
    Saves data for app in path. After saving, the app can be started with the command: hades run-app --config 'path + "config.json"'.
    
    Args:
        model_optimizers: List[ModelOptimizer]
            List of model_optimizers to save data for, each for a different section.
        path: str
            Path to save data to.
        num_words: int = 10
            Number of words to save for each topic.
        perplexity: int = 10
            Perplexity for tsne.
        n_iter: int = 1000
            Number of iterations for tsne.
        init: str = "pca"
            Initialization for tsne.
        learning_rate_tsne: Union[str, float] = "auto"
            Learning rate for tsne.
        n_neighbors: int = 7
            Number of neighbors for umap.
        metric: str = "euclidean"
            Metric for umap.
        min_dist: float = 0.1
            Minimum distance for umap.
        learning_rate_umap: float = 1
            Learning rate for umap.
        save_model: bool = False
            If True, the model is saved.
        n_extract_sentences: int = 6
            Number of sentences to extract for each document.
        do_summaries: bool = True
            If True, summaries are generated.
    """
    if len(model_optimizers) == 0:
        warnings.warn(
                """
                empty list of model_optimizers - aborting...
                
                """
            )
        return
    id_column = model_optimizers[0].id_column
    section_column = model_optimizers[0].section_column
    for model_optimizer in model_optimizers:
        if model_optimizer.id_column != id_column:
            warnings.warn(
                """
                id_columns don't mach - aborting...
                
                """
            )
            return
        elif model_optimizer.section_column != section_column:
            warnings.warn(
                """
                section_columns don't mach - aborting...
                
                """
            )
            return
        

    os.makedirs(path, exist_ok=True)
    config_dict = {}
    config_dict['id_column'] = id_column
    config_dict['sections'] = {}
    final_summaries = {}
    for model_optimizer in model_optimizers:
        if model_optimizer.name:
            clean_name = model_optimizer.name
            filter_name = model_optimizer.name.replace(":", "").replace(" ", "_").replace(
                ",", "").replace("/", "_").replace("(", "").replace(")", "").replace("&", "").replace("-", "_")
        else:
            clean_name = " ".join([
                value
                for value in model_optimizer.data[model_optimizer.section_column].unique()
            ])
            filter_name = "_".join([
                value.replace(":","").replace(" ","_")
                    .replace(",","").replace("/","_")
                    .replace("(","").replace(")","")
                    .replace("&","").replace("-","_")
                for value in model_optimizer.data[model_optimizer.section_column].unique()
            ])
        topic_words = model_optimizer.get_topics_df(num_words)
        topics_by_column = model_optimizer.get_topic_probs_averaged_over_column(id_column, show_names=True)
        if save_model:
            model_optimizer.save(path=path)
        topic_words.to_csv(path + filter_name + "_topic_words.csv")
        topics_by_column.to_csv(path + filter_name + "_probs.csv")
        tsne_mapping = model_optimizer.get_tsne_mapping(
            id_column,
            perplexity,
            n_iter,
            init,
            learning_rate_tsne,
        )
        umap_mapping = model_optimizer.get_umap_mapping(
            id_column,
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
        df_summarized = model_optimizer.data.groupby(id_column)['tokens'].sum()
        sentences_processed = sentence_topic_analyser.process_documents(df_summarized)
        id_sentence_dict = dict(zip(list(df_summarized.index), sentences_processed))
        with open(path + filter_name + "_essentials.json", 'w') as json_file:
            json.dump(id_sentence_dict, json_file)

        config_dict['sections'][clean_name] = {"probs": path + filter_name + "_probs.csv",
            "mapping": path + filter_name + "_mapping.csv",
            "topic_words": path + filter_name + "_topic_words.csv",
            "vis": path + filter_name + "_vis.txt",
            "essentials": path + filter_name + "_essentials.json"}
        
        final_summaries[clean_name] = make_section_summaries(
            model_optimizer,
            n_extract_sentences,
            clean_name,
            do_summaries,
            verbose_summaries,
        )

    config_dict["additional_files"] = []
    with open(path + "summaries.json", 'w') as fp:
        json.dump(final_summaries, fp)
    config_dict["summaries_file"] = path + "summaries.json"
    with open(path + "config.json", 'w') as json_file:
        json.dump(config_dict, json_file)
