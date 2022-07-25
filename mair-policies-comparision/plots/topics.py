from collections import Counter
from typing import List, Optional, Tuple

import pandas as pd
import pyLDAvis.gensim_models
import seaborn as sns
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from matplotlib import colors
from matplotlib import pyplot as plt

pyLDAvis.enable_notebook()


def interactive_exploration(lda_model: LdaModel, encoded_docs: pd.Series, dictionary: Dictionary):
    return pyLDAvis.gensim_models.prepare(lda_model, encoded_docs, dictionary=dictionary)


def _topics_df(model: LdaModel, docs: pd.Series, num_words: int = 10) -> pd.DataFrame:
    topics = model.show_topics(formatted=False, num_words=num_words)
    counter = Counter(docs.sum())
    out = [[word, i, weight, counter[word]] for i, topic in topics for word, weight in topic]
    df = pd.DataFrame(out, columns=["word", "topic_id", "importance", "word_count"])
    df = df.sort_values(by=["importance"], ascending=False)
    return df


def plot_topics(
    model: LdaModel,
    docs: pd.Series,
    x: int,
    y: int,
    title: str,
    figsize: Tuple[int],
    num_words: int = 10,
    ylim_weight: Optional[int] = None,
    ylim_count: Optional[int] = None,
    topics_names: Optional[List[str]] = None,
):
    df = _topics_df(model, docs, num_words)
    fig, axes = plt.subplots(x, y, figsize=figsize, sharey=False)
    cols = [color for name, color in colors.TABLEAU_COLORS.items()]
    topics = model.show_topics(formatted=False, num_words=num_words)
    for i, ax in enumerate(axes.flatten()):
        ax.bar(
            x="word",
            height="word_count",
            data=df.loc[df.topic_id == i, :],
            color=cols[i],
            width=0.5,
            alpha=0.3,
            label="Word Count",
        )
        ax_twin = ax.twinx()
        ax_twin.bar(
            x="word",
            height="importance",
            data=df.loc[df.topic_id == i, :],
            color=cols[i],
            width=0.2,
            label="Word Weight",
        )
        ax.set_ylabel("Word Count", color=cols[i])
        ax_twin.set_ylabel("Word Weight", color=cols[i])
        if ylim_weight is not None:
            ax_twin.set_ylim(0, ylim_weight)
        if ylim_count is not None:
            ax.set_ylim(0, ylim_count)
        ax.set_title(
            "Topic: " + str(topics_names[i] if topics_names else i), color=cols[i], fontsize=12
        )
        ax.set_xticklabels(
            df.loc[df.topic_id == i, "word"], rotation=30, horizontalalignment="right"
        )
        ax.legend(loc="upper right")
        ax_twin.legend(loc="lower right")
        ax.grid(False)
        ax_twin.grid(False)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_similarities(similarities, topic_probs, linkage):
    plt.figure(figsize=(12, 8))
    sns.clustermap(
        1 - similarities,
        xticklabels=topic_probs.index,
        yticklabels=topic_probs.index,
        row_linkage=linkage,
        col_linkage=linkage,
    )
    plt.show()
