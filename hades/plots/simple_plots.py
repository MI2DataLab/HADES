from collections import Counter
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_counter(
    counter: Counter,
    orient: str = "h",
    color: str = "lightblue",
    figsize: tuple = (20, 13),
    save_path: Optional[str] = None,
):
    plt.figure(figsize=figsize)
    keys = [k[0] for k in counter]
    vals = [int(k[1]) for k in counter]
    ax = sns.barplot(x=vals, y=keys, orient=orient, color=color)
    ax.bar_label(ax.containers[0])
    if save_path is not None:
        plt.savefig(save_path)
    return ax


def plot_counter_lemmas(
    df: pd.DataFrame,
    filter_dict: Dict[str, str],
    number: int = 30,
    orient: str = "h",
    color: str = "lightblue",
    figsize: tuple = (20, 13),
    save_path: Optional[str] = None,
):
    filtered_lemmas = df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)][
        "lemmas"
    ]
    counter = Counter(filtered_lemmas.sum()).most_common(number)
    plot_counter(counter, orient, color, figsize)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
